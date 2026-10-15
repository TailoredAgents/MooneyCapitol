from pathlib import Path


def test_render_runs_migrations_before_api_and_worker_start():
    text = Path("render.yaml").read_text(encoding="utf-8")
    runtime = Path("runtime.txt").read_text(encoding="utf-8").strip()
    python_version = Path(".python-version").read_text(encoding="utf-8").strip()

    assert runtime.startswith("python-3.11.")
    assert python_version == "3.11.11"
    # API, worker, and isolated read-only Rithmic capture use one pinned
    # runtime and run the additive migrations before startup.
    assert text.count("PYTHON_VERSION") == 3
    assert text.count('value: "3.11.11"') == 3
    assert text.count("preDeployCommand: python -m app.tools.run_migrations") == 3
    assert text.count("COPIER_ENABLED") == 2
    assert text.count("\n      - key: COPIER_MODE\n") == 2
    assert text.count("COPIER_GLOBAL_KILL_SWITCH") == 2
    assert text.count("OPENAI_API_KEY") == 2
    assert text.count("OPENAI_AI_FEATURES_ENABLED") == 2
    assert text.count("OPENAI_DAILY_REQUEST_LIMIT") == 2
    assert text.count("OPENAI_RESEARCH_DAILY_REQUEST_LIMIT") == 2
    assert "WEBULL_" not in text
    assert text.count("SHADOW_TRADER_ENABLED") == 2
    assert text.count("AI_LAB_ENABLED") == 2
    assert text.count("AI_LAB_STARTING_EQUITY") == 2
    assert text.count("AI_LAB_BUYING_POWER_MULTIPLIER") == 2
    assert text.count("PAPER_TRADER_BROKER_MODE") == 2
    assert "AI_LIVE_TRADING_ENABLED" not in text
    assert text.count('value: "0"') >= 6
    assert text.count("SLACK_SIGNING_SECRET") == 2
    assert "WEBULL_MASTER_ACCOUNT_EQUITY" not in text
    assert "WEBULL_PERSONAL_ACCOUNT_EQUITY" not in text
    assert "IBKR_" not in text
    assert "ib_insync" not in Path("requirements.txt").read_text(encoding="utf-8")
    assert "COWORK_OPERATOR_USERNAME" in text
    assert "COWORK_OPERATOR_PASSWORD" in text
    assert "COWORK_OPERATOR_API_TOKEN" in text
    assert text.count("RITHMIC_CAPTURE_BASE_URL") == 1
    assert text.count("RITHMIC_CAPTURE_HEALTH_TIMEOUT_SECONDS") == 1

    api_block = text.split("name: trading-agent-api", 1)[1].split(
        "  - type: worker", 1
    )[0]
    capture_block = text.split("name: mooney-rithmic-capture", 1)[1]
    copier_keys = (
        "V2_COPIER_MODE",
        "V2_COPIER_CONNECTIVITY_ENABLED",
        "V2_COPIER_ENVIRONMENT",
        "V2_RITHMIC_MASTER_ACCOUNT_ID",
        "V2_RITHMIC_MASTER_USERNAME",
        "V2_RITHMIC_MASTER_PASSWORD",
        "V2_COPIER_MAX_RISK_FRACTION",
        "V2_COPIER_MAX_CONTRACTS",
        "V2_COPIER_MAX_CONCURRENT_RISK_FRACTION",
        "V2_COPIER_DAILY_REALIZED_LOSS_FRACTION",
        "V2_COPIER_DAILY_TOTAL_LOSS_FRACTION",
        "V2_COPIER_MAX_RELATIVE_PARITY_ERROR",
        "V2_COPIER_MARGIN_HEADROOM_FRACTION",
        "V2_COPIER_ALLOW_OVER_TARGET",
        "V2_COPIER_NQ_INITIAL_MARGIN",
        "V2_COPIER_MNQ_INITIAL_MARGIN",
        "V2_COPIER_NQ_COST_BUFFER",
        "V2_COPIER_MNQ_COST_BUFFER",
    )
    assert all(text.count(key) == 1 for key in copier_keys)
    assert all(key in api_block for key in copier_keys)
    assert all(key not in capture_block for key in copier_keys)
    for slot in ("MASTER", "FOLLOWER_1", "FOLLOWER_2", "FOLLOWER_3", "FOLLOWER_4"):
        for suffix in ("DISCOVERY_URI", "SYSTEM_NAME", "GATEWAY_NAME"):
            assert f"V2_COPIER_{slot}_{suffix}" in api_block
        for suffix in ("ACCOUNT_ID", "USERNAME", "PASSWORD"):
            assert f"V2_RITHMIC_{slot}_{suffix}" in api_block


def test_shadow_copier_environment_templates_are_fail_closed():
    for filename in (".env.example", ".env.sample"):
        text = Path(filename).read_text(encoding="utf-8")

        assert "V2_COPIER_MODE=SHADOW" in text
        assert "V2_COPIER_CONNECTIVITY_ENABLED=0" in text
        assert "V2_COPIER_ENVIRONMENT=DISABLED" in text
        assert "V2_COPIER_ALLOW_OVER_TARGET=0" in text
        assert "V2_COPIER_MASTER_ACCOUNT_ID=" not in text
        assert "V2_COPIER_FOLLOWER_ACCOUNT_IDS=" not in text
        for slot in ("MASTER", "FOLLOWER_1", "FOLLOWER_2", "FOLLOWER_3", "FOLLOWER_4"):
            assert (
                f"V2_COPIER_{slot}_ACCOUNT_ID_SECRET_REF="
                f"V2_RITHMIC_{slot}_ACCOUNT_ID"
            ) in text
            assert f"V2_RITHMIC_{slot}_ACCOUNT_ID=\n" in text
            assert f"V2_RITHMIC_{slot}_USERNAME=\n" in text
            assert f"V2_RITHMIC_{slot}_PASSWORD=\n" in text
        for name in (
            "V2_COPIER_MAX_RISK_FRACTION",
            "V2_COPIER_MAX_CONTRACTS",
            "V2_COPIER_MAX_CONCURRENT_RISK_FRACTION",
            "V2_COPIER_DAILY_REALIZED_LOSS_FRACTION",
            "V2_COPIER_DAILY_TOTAL_LOSS_FRACTION",
            "V2_COPIER_NQ_INITIAL_MARGIN",
            "V2_COPIER_MNQ_INITIAL_MARGIN",
        ):
            assert f"{name}=0" in text


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
