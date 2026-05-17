from app.copier.state import get_copier_status
from app.core.config_store import CONFIG


def test_copier_status_defaults_to_safe_mode():
    original_kill_switch = CONFIG.copier.global_kill_switch
    original_enabled = CONFIG.copier.enabled
    try:
        CONFIG.copier.enabled = False
        CONFIG.copier.global_kill_switch = True

        status = get_copier_status()

        assert status["enabled"] is False
        assert status["mode"] == "test"
        assert status["global_kill_switch"] is True
        assert status["master"]["broker"] == "webull"
        assert [target["name"] for target in status["targets"]] == ["personal"]
        assert {target["broker"] for target in status["targets"]} == {"webull"}
    finally:
        CONFIG.copier.global_kill_switch = original_kill_switch
        CONFIG.copier.enabled = original_enabled
