from app.services.live_state import get_lanes, set_lanes


def test_live_state_round_trip():
    set_lanes(
        armed=[{"symbol": "XYZ"}],
        primed=[{"symbol": "ABC"}],
        active=[{"symbol": "LMN"}],
        meta={"symbols": ["XYZ", "ABC", "LMN"]},
    )
    lanes = get_lanes()
    assert len(lanes["armed"]) == 1
    assert len(lanes["primed"]) == 1
    assert len(lanes["active"]) == 1
