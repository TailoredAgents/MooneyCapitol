from __future__ import annotations


def score_setup(
    htf_proximity: float,
    gap_edge_proximity: float,
    l2_persistence_s: float,
    rvol: float,
    spread_cents: float,
    box_height: float,
) -> int:
    """Compute ordering score (not gating).

    Base 60 + HTF proximity (+15) + gap edge proximity (+15) + L2 persistence ≥2s (+10)
    − penalties (RVOL <1.0, spread spikes, tiny box).
    """
    score = 60
    score += min(int(htf_proximity * 15), 15)
    score += min(int(gap_edge_proximity * 15), 15)
    if l2_persistence_s >= 2.0:
        score += 10
    if rvol < 1.0:
        score -= 10
    if spread_cents > 1.0:
        score -= 5
    if box_height < 0.02:
        score -= 5
    return max(0, min(score, 100))

