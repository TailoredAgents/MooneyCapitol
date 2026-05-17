from app.tools.copier_latency_benchmark import _pass_rate, _percentile, _stats, _summary


def test_latency_benchmark_stats_include_tail_percentiles():
    values = [1.0, 2.0, 3.0, 4.0]

    stats = _stats(values)

    assert stats["count"] == 4
    assert stats["p50"] == 2.5
    assert _percentile(values, 95) == 3.8499999999999996


def test_latency_benchmark_pass_rate_uses_300ms_threshold():
    assert _pass_rate([100.0, 250.0, 301.0], threshold_ms=300) == 2 / 3


def test_latency_benchmark_summary_counts_submits_and_blocks():
    class Args:
        iterations = 2
        targets = 1
        broker_delay_ms = 0.0

    rows = [
        {
            "submitted": 1,
            "blocked": 0,
            "event_to_submit_started_ms": 1.0,
            "broker_submit_ms": 2.0,
            "event_to_broker_response_ms": 3.0,
        },
        {
            "submitted": 0,
            "blocked": 1,
            "event_to_submit_started_ms": None,
            "broker_submit_ms": None,
            "event_to_broker_response_ms": 4.0,
        },
    ]

    summary = _summary(rows, Args())

    assert summary["submitted"] == 1
    assert summary["blocked"] == 1
    assert summary["latency"]["event_to_broker_response_ms"]["p50"] == 3.5
    assert summary["sub_300ms_pass_rate"] == 1.0
