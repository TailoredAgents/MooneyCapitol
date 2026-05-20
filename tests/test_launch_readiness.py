from app.api.routes.launch import _latency_stats, _read_only_counts, router


class FakeScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def scalar(self):
        return self.rows[0] if self.rows else None

    def scalars(self):
        return FakeScalarResult(self.rows)


class FakeSession:
    def __init__(self, *results):
        self.results = list(results)

    def execute(self, stmt):
        return FakeResult(self.results.pop(0))


def test_launch_readiness_route_is_registered():
    paths = {route.path for route in router.routes}

    assert "/launch/readiness" in paths


def test_launch_readiness_counts_read_only_decisions():
    counts = _read_only_counts(FakeSession([3], [2]))

    assert counts == {"would_copy": 3, "blocked": 2, "total": 5}


def test_launch_readiness_latency_stats_include_under_300ms_rate():
    stats = _latency_stats(FakeSession([100.0, 250.0, 450.0]))

    assert stats["count"] == 3
    assert stats["mean_ms"] == 266.67
    assert stats["max_ms"] == 450.0
    assert stats["under_300ms_rate"] == 0.6667
