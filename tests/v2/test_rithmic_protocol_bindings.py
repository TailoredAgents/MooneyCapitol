from __future__ import annotations

import base64
import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.v2.brokers.rithmic_protocol.adapter import (
    ObserverRuntimeConfig,
    RithmicReadOnlyObserver,
)
from app.v2.brokers.rithmic_protocol.bindings import (
    ExternalBindingRegistry,
    ExternalBindingsConfig,
    prepare_bindings,
)
from app.v2.brokers.rithmic_protocol.constants import (
    INBOUND_BINDINGS,
    OUTBOUND_BINDINGS,
    PROTOCOL_PACKAGE_VERSION,
    PROTOCOL_TEMPLATE_VERSION,
    Plant,
    Template,
)
from app.v2.brokers.rithmic_protocol.errors import BindingsUnavailable, UnauthorizedAccount
from app.v2.brokers.rithmic_protocol.factory import (
    AccountAllowlist,
    ReadOnlyMessageFactory,
)
from app.v2.brokers.rithmic_protocol.normalization import BrokerAccountKey
from app.v2.brokers.rithmic_protocol.transport import (
    ReadOnlyOutboundPolicy,
    create_client_ssl_context,
)


def _binding_files() -> set[str]:
    values = (*INBOUND_BINDINGS.values(), *OUTBOUND_BINDINGS.values())
    return {f"{module}.py" for module, _ in values}


def _make_binding_directory(path: Path) -> None:
    path.mkdir(parents=True)
    checksums: dict[str, str] = {}
    for filename in _binding_files():
        payload = f"# synthetic {filename}\n".encode()
        (path / filename).write_bytes(payload)
        checksums[filename] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "protocol_package": f"RProtocolAPI.{PROTOCOL_PACKAGE_VERSION}",
        "template_version": PROTOCOL_TEMPLATE_VERSION,
        "files": checksums,
    }
    (path / "rithmic_bindings_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _make_archive(source: Path, archive: Path) -> str:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for item in source.iterdir():
            package.write(item, item.name)
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def test_directory_bindings_require_exact_versioned_manifest_and_external_location(tmp_path):
    generated = tmp_path / "generated"
    _make_binding_directory(generated)
    prepared = prepare_bindings(ExternalBindingsConfig(generated_path=generated))
    assert prepared.path == generated.resolve()
    prepared.close()

    with pytest.raises(BindingsUnavailable, match="outside the repository"):
        prepare_bindings(
            ExternalBindingsConfig(
                generated_path=generated,
                forbid_workspace_root=tmp_path,
            )
        )

    manifest_path = generated / "rithmic_bindings_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["template_version"] = "old"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BindingsUnavailable, match="template version"):
        prepare_bindings(ExternalBindingsConfig(generated_path=generated))


def test_checksum_verified_zip_and_strict_base64_secret_file(tmp_path):
    generated = tmp_path / "generated"
    _make_binding_directory(generated)
    archive = tmp_path / "bindings.zip"
    digest = _make_archive(generated, archive)

    prepared = prepare_bindings(
        ExternalBindingsConfig(archive_path=archive, archive_sha256=digest)
    )
    extracted = prepared.path
    assert (extracted / "rithmic_bindings_manifest.json").is_file()
    prepared.close()
    assert not extracted.exists()

    encoded = tmp_path / "bindings.b64"
    encoded.write_bytes(base64.b64encode(archive.read_bytes()))
    prepared = prepare_bindings(
        ExternalBindingsConfig(archive_b64_file=encoded, archive_sha256=digest)
    )
    assert (prepared.path / "request_login_pb2.py").is_file()
    prepared.close()

    encoded.write_bytes(b"not valid base64!")
    with pytest.raises(BindingsUnavailable, match="strict base64"):
        prepare_bindings(
            ExternalBindingsConfig(archive_b64_file=encoded, archive_sha256=digest)
        )


def test_archive_rejects_zip_slip_unregistered_files_and_bad_checksum(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../request_login_pb2.py", "bad")
        package.writestr("rithmic_bindings_manifest.json", "{}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(BindingsUnavailable, match="unsafe path"):
        prepare_bindings(ExternalBindingsConfig(archive_path=archive, archive_sha256=digest))
    with pytest.raises(BindingsUnavailable, match="checksum mismatch"):
        prepare_bindings(
            ExternalBindingsConfig(archive_path=archive, archive_sha256="0" * 64)
        )


def test_registry_rejects_same_named_module_already_loaded_from_another_path(tmp_path):
    module_name = "synthetic_collision_pb2"
    (tmp_path / f"{module_name}.py").write_text("class Message: pass\n", encoding="utf-8")
    registry = ExternalBindingRegistry(
        tmp_path,
        inbound={},
        outbound={18: (module_name, "Message")},
    )
    collision = types.ModuleType(module_name)
    collision.__file__ = str(tmp_path.parent / "elsewhere" / f"{module_name}.py")
    sys.modules[module_name] = collision
    try:
        with pytest.raises(BindingsUnavailable, match="outside configured path"):
            registry.message_class(18, outbound=True)
    finally:
        sys.modules.pop(module_name, None)


class _FakeRegistry:
    def build(self, template_id: int, **fields):
        return SimpleNamespace(template_id=template_id, **fields)


def test_exact_account_allowlist_gates_every_account_scoped_factory_request():
    allowed = BrokerAccountKey("f", "i", "allowed")
    denied = BrokerAccountKey("f", "i", "denied")
    factory = ReadOnlyMessageFactory(
        _FakeRegistry(),  # type: ignore[arg-type]
        account_allowlist=AccountAllowlist([allowed]),
    )
    request = factory.subscribe_orders(allowed, correlation_id="c")
    assert request.template_id == Template.ORDER_UPDATES_REQUEST
    assert request.account_id == "allowed"
    with pytest.raises(UnauthorizedAccount):
        factory.subscribe_orders(denied)
    with pytest.raises(UnauthorizedAccount):
        factory.pnl_snapshot(denied)


def test_account_metadata_playback_request_uses_official_epoch_fields():
    factory = ReadOnlyMessageFactory(_FakeRegistry())  # type: ignore[arg-type]
    request = factory.playback_account_users(
        start_index=0,
        finish_index=1_791_382_800,
        playback="accounts",
        correlation_id="account-metadata",
    )
    assert request.template_id == Template.PLAYBACK_ACCOUNT_USERS_REQUEST
    assert request.user_msg == ["account-metadata"]
    assert request.playback == "accounts"
    assert request.start_index == 0
    assert request.finish_index == 1_791_382_800
    with pytest.raises(ValueError, match="proto int32"):
        factory.playback_account_users(start_index=2_147_483_648)


def test_fill_history_disables_unimplemented_server_flow_control():
    account = BrokerAccountKey("f", "i", "allowed")
    factory = ReadOnlyMessageFactory(
        _FakeRegistry(),  # type: ignore[arg-type]
        account_allowlist=AccountAllowlist([account]),
    )

    request = factory.fill_history(
        account,
        start_index=20261007,
        finish_index=20261007,
        correlation_id="fills-client",
    )

    assert request.flow_control == "disabled"
    assert request.user_msg == ["fills-client"]


def test_runtime_config_hides_credentials_and_rejects_wrong_template(monkeypatch):
    monkeypatch.setenv("RITHMIC_DISCOVERY_URI", "wss://discovery.test")
    monkeypatch.setenv("RITHMIC_SYSTEM_NAME", "Rithmic Test")
    monkeypatch.setenv("RITHMIC_USERNAME", "top-secret-user")
    monkeypatch.setenv("RITHMIC_PASSWORD", "top-secret-password")
    monkeypatch.setenv("RITHMIC_TEMPLATE_VERSION", PROTOCOL_TEMPLATE_VERSION)
    capture = SimpleNamespace(
        environment="TEST",
        connectivity_enabled=True,
        account_allowlist=frozenset({"account"}),
        reconcile_timeout_seconds=120,
    )
    runtime = ObserverRuntimeConfig.from_capture_config(capture)
    rendered = repr(runtime)
    assert "top-secret-user" not in rendered
    assert "top-secret-password" not in rendered
    assert runtime.order_uri is None and runtime.pnl_uri is None
    assert runtime.account_metadata_start_ssboe == 0

    monkeypatch.setenv("RITHMIC_TEMPLATE_VERSION", "0.49")
    with pytest.raises(ValueError, match=PROTOCOL_TEMPLATE_VERSION):
        ObserverRuntimeConfig.from_capture_config(capture)

    monkeypatch.setenv("RITHMIC_TEMPLATE_VERSION", PROTOCOL_TEMPLATE_VERSION)
    monkeypatch.setenv("RITHMIC_ACCOUNT_METADATA_START_SSBOE", "2147483648")
    with pytest.raises(ValueError, match="proto int32"):
        ObserverRuntimeConfig.from_capture_config(capture)


def test_gateway_selection_is_explicit_when_discovery_returns_multiple():
    gateways = {"primary": "wss://one.test", "backup": "wss://two.test"}
    assert RithmicReadOnlyObserver._select_gateway(gateways, "primary") == "wss://one.test"
    with pytest.raises(Exception, match="explicit gateway"):
        RithmicReadOnlyObserver._select_gateway(gateways, None)
    with pytest.raises(Exception, match="not discovered"):
        RithmicReadOnlyObserver._select_gateway(gateways, "missing")


def test_discovered_gateways_build_distinct_order_and_pnl_sessions():
    observer = object.__new__(RithmicReadOnlyObserver)
    observer.runtime = SimpleNamespace(
        order_uri=None,
        pnl_uri=None,
        ticker_uri=None,
        gateway_name=None,
        order_gateway_name="orders",
        pnl_gateway_name="pnl",
        ticker_gateway_name=None,
        login_timeout_seconds=30,
    )
    observer.capture_config = SimpleNamespace(enabled_plants=())
    observer.policy = ReadOnlyOutboundPolicy()
    observer._ssl_context = create_client_ssl_context()
    observer.registry = object()
    observer._sessions = {}

    observer._build_sessions(
        {
            "orders": "wss://orders.example.test/rithmic",
            "pnl": "wss://pnl.example.test/rithmic",
        }
    )

    order = observer._sessions[Plant.ORDER]
    pnl = observer._sessions[Plant.PNL]
    assert order is not pnl
    assert order.state is not pnl.state
    assert order.transport.endpoint.uri == "wss://orders.example.test/rithmic"
    assert pnl.transport.endpoint.uri == "wss://pnl.example.test/rithmic"
