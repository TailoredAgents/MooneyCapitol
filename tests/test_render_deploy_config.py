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
    assert text.count("OPENAI_API_KEY") == 2
    assert text.count("OPENAI_AI_FEATURES_ENABLED") == 2
    assert text.count("OPENAI_DAILY_REQUEST_LIMIT") == 2
    assert text.count("OPENAI_RESEARCH_DAILY_REQUEST_LIMIT") == 2
    assert text.count("WEBULL_PERSONAL_DISPLAY_NAME") == 2
    assert text.count("AI_LAB_ENABLED") == 2
    assert text.count("AI_LAB_STARTING_EQUITY") == 2
    assert text.count("PAPER_TRADER_BROKER_MODE") == 2
    assert text.count("WEBULL_AI_PAPER_API_ENDPOINT") == 2
    assert text.count("WEBULL_AI_PAPER_ACCOUNT_ID") == 2
    assert text.count("AI_LIVE_TRADING_ENABLED") == 2
    assert text.count('value: "0"') >= 6
    assert text.count("SLACK_SIGNING_SECRET") == 2
    assert "WEBULL_MASTER_ACCOUNT_EQUITY" not in text
    assert "WEBULL_PERSONAL_ACCOUNT_EQUITY" not in text
    assert "IBKR_" not in text
    assert "ib_insync" not in Path("requirements.txt").read_text(encoding="utf-8")
    assert "COWORK_OPERATOR_USERNAME" in text
    assert "COWORK_OPERATOR_PASSWORD" in text
    assert "COWORK_OPERATOR_API_TOKEN" in text


def test_api_mounts_static_assets_for_favicon():
    text = Path("app/api/main.py").read_text(encoding="utf-8")

    assert "StaticFiles" in text
    assert 'app.mount("/static", StaticFiles(directory="app/static"), name="static")' in text


def test_migration_runner_uses_postgres_advisory_lock():
    text = Path("app/tools/run_migrations.py").read_text(encoding="utf-8")

    assert "pg_advisory_lock" in text
    assert "command.upgrade" in text
    assert "DATABASE_URL is required" in text


def test_alembic_revision_ids_fit_default_version_column():
    for path in Path("migrations/versions").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        revision_line = next(line for line in text.splitlines() if line.startswith("revision = "))
        revision = revision_line.split("=", 1)[1].strip().strip('"')

        assert len(revision) <= 32, f"{path.name} revision is too long for alembic_version.version_num"
