from pathlib import Path


def test_render_runs_migrations_before_api_and_worker_start():
    text = Path("render.yaml").read_text(encoding="utf-8")
    runtime = Path("runtime.txt").read_text(encoding="utf-8").strip()
    python_version = Path(".python-version").read_text(encoding="utf-8").strip()

    assert runtime.startswith("python-3.11.")
    assert python_version == "3.11.11"
    assert text.count("PYTHON_VERSION") == 2
    assert text.count('value: "3.11.11"') == 2
    assert text.count("preDeployCommand: python -m app.tools.run_migrations") == 2
    assert text.count("COPIER_ENABLED") == 2
    assert text.count("COPIER_MODE") == 2
    assert text.count("COPIER_GLOBAL_KILL_SWITCH") == 2
    assert text.count("SLACK_SIGNING_SECRET") == 2
    assert "WEBULL_MASTER_ACCOUNT_EQUITY" not in text
    assert "WEBULL_PERSONAL_ACCOUNT_EQUITY" not in text
    assert "IBKR_" not in text
    assert "ib_insync" not in Path("requirements.txt").read_text(encoding="utf-8")
    assert "COWORK_OPERATOR_USERNAME" in text
    assert "COWORK_OPERATOR_PASSWORD" in text
    assert "COWORK_OPERATOR_API_TOKEN" in text


def test_migration_runner_uses_postgres_advisory_lock():
    text = Path("app/tools/run_migrations.py").read_text(encoding="utf-8")

    assert "pg_advisory_lock" in text
    assert "command.upgrade" in text
    assert "DATABASE_URL is required" in text
