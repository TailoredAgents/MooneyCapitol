from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path


# Only the schemas needed by the observation-only runtime are generated.  No
# vendor schema or generated derivative is written into this repository.
READ_ONLY_PROTO_FILES = (
    "request_login.proto",
    "response_login.proto",
    "request_logout.proto",
    "response_logout.proto",
    "request_heartbeat.proto",
    "response_heartbeat.proto",
    "request_rithmic_system_info.proto",
    "response_rithmic_system_info.proto",
    "request_rithmic_system_gateway_info.proto",
    "response_rithmic_system_gateway_info.proto",
    "request_flow_control.proto",
    "response_flow_control.proto",
    "request_session_config.proto",
    "response_session_config.proto",
    "reject.proto",
    "forced_logout.proto",
    "user_account_update.proto",
    "account_and_user_update.proto",
    "request_login_info.proto",
    "response_login_info.proto",
    "request_account_list.proto",
    "response_account_list.proto",
    "request_account_rms_info.proto",
    "response_account_rms_info.proto",
    "request_product_rms_info.proto",
    "response_product_rms_info.proto",
    "request_get_user_info.proto",
    "response_get_user_info.proto",
    "user_info_update.proto",
    "request_playback_acc_users.proto",
    "response_playback_acc_users.proto",
    "request_show_acc_user_history.proto",
    "response_show_acc_user_history.proto",
    "request_account_rms_updates.proto",
    "response_account_rms_updates.proto",
    "account_rms_updates.proto",
    "request_subscribe_for_order_updates.proto",
    "response_subscribe_for_order_updates.proto",
    "request_trade_routes.proto",
    "response_trade_routes.proto",
    "trade_route.proto",
    "request_show_order_history_dates.proto",
    "response_show_order_history_dates.proto",
    "request_show_orders.proto",
    "response_show_orders.proto",
    "request_show_order_history.proto",
    "response_show_order_history.proto",
    "request_show_order_history_summary.proto",
    "response_show_order_history_summary.proto",
    "request_show_order_history_detail.proto",
    "response_show_order_history_detail.proto",
    "request_replay_executions.proto",
    "response_replay_executions.proto",
    "request_show_fill_history.proto",
    "response_show_fill_history.proto",
    "request_subscribe_to_bracket_updates.proto",
    "response_subscribe_to_bracket_updates.proto",
    "request_show_brackets.proto",
    "response_show_brackets.proto",
    "request_show_bracket_stops.proto",
    "response_show_bracket_stops.proto",
    "rithmic_order_notification.proto",
    "exchange_order_notification.proto",
    "bracket_updates.proto",
    "request_pnl_position_updates.proto",
    "response_pnl_position_updates.proto",
    "request_pnl_position_snapshot.proto",
    "response_pnl_position_snapshot.proto",
    "instrument_pnl_position_update.proto",
    "account_pnl_position_update.proto",
    "request_reference_data.proto",
    "response_reference_data.proto",
    "request_get_instrument_by_underlying.proto",
    "response_get_instrument_by_underlying.proto",
    "response_get_instrument_by_underlying_keys.proto",
    "request_give_tick_size_type_table.proto",
    "response_give_tick_size_type_table.proto",
    "request_search_symbols.proto",
    "response_search_symbols.proto",
    "request_product_codes.proto",
    "response_product_codes.proto",
    "request_front_month_contract.proto",
    "response_front_month_contract.proto",
    "front_month_contract_update.proto",
    "request_auxilliary_reference_data.proto",
    "response_auxilliary_reference_data.proto",
    "request_list_exchange_permissions.proto",
    "response_list_exchange_permissions.proto",
    "request_easy_to_borrow_list.proto",
    "response_easy_to_borrow_list.proto",
    "request_user_entitlements.proto",
    "response_user_entitlements.proto",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _proto_directory(sdk_path: Path) -> Path:
    candidates = (sdk_path, sdk_path / "proto")
    for candidate in candidates:
        if (candidate / "request_login.proto").is_file():
            return candidate.resolve()
    raise ValueError("SDK path must be the official R|Protocol package or its proto directory")


def _require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(_repo_root())
    except ValueError:
        return resolved
    raise ValueError(f"{label} must remain outside the Git repository")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(
    sdk_path: Path,
    output_path: Path,
    archive_path: Path | None = None,
    archive_base64_path: Path | None = None,
) -> dict[str, object]:
    if importlib.util.find_spec("grpc_tools.protoc") is None:
        raise RuntimeError("install requirements-rithmic-build.txt in a private/local build environment")
    proto_dir = _proto_directory(sdk_path)
    output_dir = _require_external(output_path, "generated bindings output")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_generated_names = {
        f"{Path(name).stem}_pb2.py" for name in READ_ONLY_PROTO_FILES
    }
    unexpected_existing = {
        item.name for item in output_dir.glob("*_pb2.py")
    } - expected_generated_names
    if unexpected_existing:
        raise ValueError(
            "generated output contains unexpected bindings; use a clean external directory"
        )
    if archive_base64_path is not None and archive_path is None:
        raise ValueError("--archive-base64 requires --archive")

    missing = [name for name in READ_ONLY_PROTO_FILES if not (proto_dir / name).is_file()]
    if missing:
        raise ValueError(f"official SDK is missing required schemas: {', '.join(missing)}")

    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={output_dir}",
        *(str(proto_dir / name) for name in READ_ONLY_PROTO_FILES),
    ]
    subprocess.run(command, check=True)

    generated = sorted(
        output_dir / name for name in expected_generated_names if (output_dir / name).is_file()
    )
    if {item.name for item in generated} != expected_generated_names:
        raise RuntimeError("protobuf generation did not produce the complete read-only binding set")
    manifest: dict[str, object] = {
        "protocol_package": "RProtocolAPI.0.90.0.0",
        "template_version": "5.55",
        "files": {item.name: _sha256(item) for item in generated},
    }
    (output_dir / "rithmic_bindings_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if archive_path is not None:
        archive = _require_external(archive_path, "bindings archive")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for item in generated:
                bundle.write(item, item.name)
            bundle.write(
                output_dir / "rithmic_bindings_manifest.json",
                "rithmic_bindings_manifest.json",
            )
        manifest["archive_sha256"] = _sha256(archive)
        if archive_base64_path is not None:
            encoded_path = _require_external(
                archive_base64_path, "base64 bindings secret file"
            )
            encoded_path.parent.mkdir(parents=True, exist_ok=True)
            encoded_path.write_bytes(base64.b64encode(archive.read_bytes()))
            manifest["archive_base64_bytes"] = encoded_path.stat().st_size
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate licensed R|Protocol read-only bindings outside Git")
    parser.add_argument("--sdk-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--archive-base64",
        type=Path,
        help="write a single-line base64 archive for a plaintext secret-file service",
    )
    args = parser.parse_args()
    manifest = generate(
        args.sdk_path,
        args.output,
        args.archive,
        args.archive_base64,
    )
    print(
        json.dumps(
            {
                "generated_file_count": len(manifest["files"]),
                "archive_sha256": manifest.get("archive_sha256"),
                "archive_base64_bytes": manifest.get("archive_base64_bytes"),
            }
        )
    )


if __name__ == "__main__":
    main()
