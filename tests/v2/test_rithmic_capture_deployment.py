from __future__ import annotations

import ast
from pathlib import Path

from app.v2.capture.contracts import ReadOnlyObserver


def test_render_has_isolated_fail_closed_capture_web_service():
    text = Path("render.yaml").read_text(encoding="utf-8")
    service = text[text.index("name: mooney-rithmic-capture") :]

    assert "type: web" in text[text.rfind("- type: web", 0, text.index("name: mooney-rithmic-capture")) :]
    assert "startCommand: python -m app.v2.capture.main" in service
    assert "healthCheckPath: /health" in service
    assert "RITHMIC_CAPTURE_CONNECTIVITY_ENABLED" in service
    assert 'value: "0"' in service
    assert "RITHMIC_ENVIRONMENT" in service
    assert "value: DISABLED" in service
    assert "RITHMIC_USERNAME" in service and "sync: false" in service
    assert "RITHMIC_PASSWORD" in service
    assert "preDeployCommand: python -m app.tools.run_migrations" in service
    assert "maxShutdownDelaySeconds: 60" in service
    assert "numInstances: 1" in service
    assert (
        "value: app.v2.brokers.rithmic_protocol.adapter:create_observer" in service
    )
    assert "value: app.v2.capture.persistence:create_journal" in service
    assert "RITHMIC_GENERATED_BINDINGS_ARCHIVE_B64_FILE" in service
    assert "RITHMIC_GENERATED_BINDINGS_SHA256" in service
    assert "RITHMIC_BINDINGS_MODULE" not in service
    assert "OPENAI_API_KEY" not in service
    assert "COPIER_ENABLED" not in service


def test_capture_environment_examples_are_offline_and_contain_no_credentials():
    for name in (".env.example", ".env.sample"):
        text = Path(name).read_text(encoding="utf-8")
        assert "RITHMIC_CAPTURE_CONNECTIVITY_ENABLED=0" in text
        assert "RITHMIC_ENVIRONMENT=DISABLED" in text
        assert "RITHMIC_USERNAME=\n" in text
        assert "RITHMIC_PASSWORD=\n" in text
        assert "RITHMIC_ACCOUNT_ALLOWLIST=\n" in text


def test_capture_package_has_no_execution_copier_ml_or_openai_imports():
    forbidden_prefixes = (
        "app.api",
        "app.copier",
        "app.v2.execution",
        "app.v2.intelligence",
        "app.v2.learning",
        "app.v2.shadow",
        "openai",
    )
    for path in Path("app/v2/capture").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            module == prefix or module.startswith(f"{prefix}.")
            for module in imported
            for prefix in forbidden_prefixes
        ), f"forbidden dependency in {path}: {imported}"


def test_observer_contract_is_structurally_read_only():
    methods = {
        node.name
        for node in ast.walk(ast.parse(Path("app/v2/capture/contracts.py").read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not methods.intersection(
        {
            "submit",
            "modify",
            "cancel",
            "cancel_all",
            "flatten",
            "exit_position",
            "submit_bracket",
            "send_oco",
            "link_order",
            "modify_stop",
            "modify_target",
        }
    )


def test_vendor_material_has_narrow_gitignore_guards():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "/.vendor/rithmic/" in text
    assert "/generated/rithmic/" in text
    assert "/RProtocolAPI*/" in text
    assert "/RApiPlus*/" in text
    assert "*.proto" not in text
    assert "*.dll" not in text
