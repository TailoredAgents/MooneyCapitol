import os
import pytest


@pytest.mark.skipif(os.getenv("RUN_DEMO_SMOKE") != "1", reason="Requires RUN_DEMO_SMOKE=1 with API/worker running")
def test_demo_smoke():
    # placeholder for manual integration smoke test
    assert True
