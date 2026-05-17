from datetime import datetime, timezone

from app.copier.order_response import normalize_copy_order_response, normalize_copy_order_responses


def test_normalize_copy_order_response_maps_submit_ack():
    detail = normalize_copy_order_response(
        {
            "success": True,
            "data": {
                "orderId": "child-1",
                "clientOrderId": "mc123",
                "orderStatus": "SUBMITTED",
                "createdAt": "2026-05-16T14:30:01+00:00",
            },
        }
    )

    assert detail.status == "submitted"
    assert detail.broker_order_id == "child-1"
    assert detail.client_order_id == "mc123"
    assert detail.accepted_at == datetime(2026, 5, 16, 14, 30, 1, tzinfo=timezone.utc)


def test_normalize_copy_order_response_maps_filled_detail():
    detail = normalize_copy_order_response(
        {
            "data": {
                "entrustId": "child-2",
                "status": "FILLED",
                "filledQuantity": "3",
                "averagePrice": "12.50",
                "updatedAt": "2026-05-16T14:31:00+00:00",
            }
        }
    )

    assert detail.status == "filled"
    assert detail.broker_order_id == "child-2"
    assert detail.filled_qty == 3.0
    assert detail.avg_fill_price == 12.5
    assert detail.filled_at == datetime(2026, 5, 16, 14, 31, tzinfo=timezone.utc)


def test_normalize_copy_order_response_maps_reject_envelope():
    detail = normalize_copy_order_response(
        {"success": False, "code": "417", "message": "insufficient buying power"}
    )

    assert detail.status == "rejected"
    assert detail.reject_reason == "insufficient buying power"


def test_normalize_copy_order_responses_handles_batch_payload():
    details = normalize_copy_order_responses(
        {
            "data": {
                "orders": [
                    {"orderId": "child-1", "orderStatus": "SUBMITTED"},
                    {"orderId": "child-2", "orderStatus": "REJECTED", "rejectReason": "symbol blocked"},
                ]
            }
        }
    )

    assert [detail.status for detail in details] == ["submitted", "rejected"]
    assert [detail.broker_order_id for detail in details] == ["child-1", "child-2"]
    assert details[1].reject_reason == "symbol blocked"
