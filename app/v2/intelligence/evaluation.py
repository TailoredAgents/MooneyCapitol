from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Sequence

from app.v2.domain.models import PositionSide
from app.v2.intelligence.datasets import BehaviorLabelState, ObservationOrigin
from app.v2.intelligence.observation import require_aware
from app.v2.intelligence.ranking import ScoutRankingSnapshot


@dataclass(frozen=True)
class ShadowTradeEvaluation:
    evaluation_id: str
    ranking_id: str
    master_trade_id: str
    candidate_id: str | None
    origin: ObservationOrigin
    prediction_cutoff_at: datetime
    action_anchor_at: datetime
    rank_at_prediction: int | None
    lead_time: timedelta | None
    top_1: bool
    top_3: bool
    top_5: bool
    realized_r: Decimal | None
    mfe_r: Decimal | None
    mae_r: Decimal | None
    matching_version: str
    data_quality_eligible: bool
    predicted_direction: PositionSide | None = None
    actual_direction: PositionSide | None = None
    direction_match: bool | None = None

    def __post_init__(self) -> None:
        require_aware(self.prediction_cutoff_at, "prediction_cutoff_at")
        require_aware(self.action_anchor_at, "action_anchor_at")
        if self.action_anchor_at < self.prediction_cutoff_at:
            raise ValueError("trade action cannot predate prediction cutoff")
        if self.origin == ObservationOrigin.RETROSPECTIVE_TRADE_LEADUP and self.lead_time is not None:
            raise ValueError("retrospective snapshots cannot claim prospective lead time")
        if self.origin in {ObservationOrigin.LIVE, ObservationOrigin.BLIND_REPLAY}:
            expected = self.action_anchor_at - self.prediction_cutoff_at
            if self.lead_time != expected:
                raise ValueError("prospective lead time must equal action anchor minus prediction cutoff")
        if self.rank_at_prediction is None and any((self.top_1, self.top_3, self.top_5)):
            raise ValueError("top-k hits require an observed rank")
        if self.rank_at_prediction is not None:
            if self.rank_at_prediction <= 0:
                raise ValueError("observed rank must be positive")
            expected = (
                self.rank_at_prediction == 1,
                self.rank_at_prediction <= 3,
                self.rank_at_prediction <= 5,
            )
            if (self.top_1, self.top_3, self.top_5) != expected:
                raise ValueError("top-k indicators must agree with observed rank")
        if not self.matching_version:
            raise ValueError("shadow trade matching must be versioned")
        for value in (self.predicted_direction, self.actual_direction):
            if value == PositionSide.FLAT:
                raise ValueError("shadow trade directions must be LONG, SHORT, or unknown")
        expected_direction_match = None
        if self.predicted_direction is not None and self.actual_direction is not None:
            expected_direction_match = self.predicted_direction == self.actual_direction
        if self.direction_match != expected_direction_match:
            raise ValueError("direction_match must reflect predicted and actual directions")


class ShadowMatchStatus(str, Enum):
    PENDING = "PENDING"
    MATCHED_TRADE = "MATCHED_TRADE"
    NO_MATCH_OBSERVED_WINDOW = "NO_MATCH_OBSERVED_WINDOW"
    EXPLICIT_REJECTION = "EXPLICIT_REJECTION"
    DATA_INVALID = "DATA_INVALID"


@dataclass(frozen=True)
class ShadowCandidateEvaluation:
    """Evaluation of a surfaced candidate without inventing a negative label."""

    evaluation_id: str
    ranking_id: str
    candidate_id: str
    rank: int
    prediction_cutoff_at: datetime
    status: ShadowMatchStatus
    evaluated_at: datetime
    behavior_label_state: BehaviorLabelState = BehaviorLabelState.UNLABELED
    matched_master_trade_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_aware(self.prediction_cutoff_at, "prediction_cutoff_at")
        require_aware(self.evaluated_at, "evaluated_at")
        if self.evaluated_at < self.prediction_cutoff_at:
            raise ValueError("candidate cannot be evaluated before prediction")
        if self.rank <= 0:
            raise ValueError("candidate rank must be positive")
        if self.status == ShadowMatchStatus.MATCHED_TRADE:
            if not self.matched_master_trade_id or self.behavior_label_state != BehaviorLabelState.POSITIVE_TRADE:
                raise ValueError("matched trades require an observed-positive behavior label")
        elif self.status == ShadowMatchStatus.EXPLICIT_REJECTION:
            if self.behavior_label_state != BehaviorLabelState.EXPLICIT_PASS or not self.evidence:
                raise ValueError("explicit rejection requires explicit evidence")
        elif self.behavior_label_state != BehaviorLabelState.UNLABELED:
            raise ValueError("pending, unmatched, and invalid candidates remain unlabeled")
        if self.status == ShadowMatchStatus.NO_MATCH_OBSERVED_WINDOW and self.matched_master_trade_id is not None:
            raise ValueError("an unmatched candidate cannot link a master trade")


def evaluate_trade_against_ranking(
    ranking: ScoutRankingSnapshot,
    *,
    evaluation_id: str,
    master_trade_id: str,
    candidate_id: str | None,
    action_anchor_at: datetime,
    realized_r: Decimal | None,
    mfe_r: Decimal | None,
    mae_r: Decimal | None,
    matching_version: str,
    data_quality_eligible: bool,
    actual_direction: PositionSide | None = None,
) -> ShadowTradeEvaluation:
    require_aware(action_anchor_at, "action_anchor_at")
    ranked = next((item for item in ranking.opportunities if item.candidate_id == candidate_id), None)
    rank = ranked.rank if ranked else None
    predicted_direction = ranked.predicted_direction if ranked else None
    lead_time = None
    if ranking.origin in {ObservationOrigin.LIVE, ObservationOrigin.BLIND_REPLAY}:
        lead_time = action_anchor_at - ranking.feature_cutoff_at
    return ShadowTradeEvaluation(
        evaluation_id,
        ranking.ranking_id,
        master_trade_id,
        candidate_id,
        ranking.origin,
        ranking.feature_cutoff_at,
        action_anchor_at,
        rank,
        lead_time,
        rank == 1,
        rank is not None and rank <= 3,
        rank is not None and rank <= 5,
        realized_r,
        mfe_r,
        mae_r,
        matching_version,
        data_quality_eligible,
        predicted_direction,
        actual_direction,
        None if predicted_direction is None or actual_direction is None else predicted_direction == actual_direction,
    )


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    task_namespace: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    purge_trade_dates: int
    dataset_version: str

    def __post_init__(self) -> None:
        if not self.task_namespace.startswith("v2.futures."):
            raise ValueError("walk-forward fold requires a V2 task namespace")
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ValueError("walk-forward evaluation must be strictly chronological")
        if self.purge_trade_dates < 0:
            raise ValueError("purge_trade_dates cannot be negative")


class EvaluationMode(str, Enum):
    SHADOW_FORWARD = "SHADOW_FORWARD"
    WALK_FORWARD = "WALK_FORWARD"


@dataclass(frozen=True)
class EvaluationRunContract:
    evaluation_run_id: str
    task_namespace: str
    mode: EvaluationMode
    evaluation_start: date
    evaluation_end: date
    dataset_version: str
    metric_version: str
    training_end: date | None = None
    model_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_namespace.startswith("v2.futures."):
            raise ValueError("evaluation run requires a V2 futures task namespace")
        if self.evaluation_end < self.evaluation_start:
            raise ValueError("evaluation date range must be ordered")
        if self.mode == EvaluationMode.WALK_FORWARD:
            if self.training_end is None or self.training_end >= self.evaluation_start:
                raise ValueError("walk-forward training must end before evaluation starts")
        if not self.dataset_version or not self.metric_version:
            raise ValueError("evaluation data and metrics must be versioned")


@dataclass(frozen=True)
class ShadowMetricSummary:
    evaluation_run_id: str
    fold_id: str
    sample_count: int
    actual_trade_top_1_rate: Decimal | None
    actual_trade_top_3_rate: Decimal | None
    actual_trade_top_5_rate: Decimal | None
    mean_lead_time_seconds: Decimal | None
    mean_rank: Decimal | None
    rank_correlation_realized_r: Decimal | None
    calibration_metrics: Mapping[str, Decimal]
    observed_positive_precision: Mapping[str, Decimal]
    false_high_rank_count: int
    unlabeled_high_rank_count: int
    rank_stability: Decimal | None
    data_quality_failure_count: int


def summarize_shadow_evaluations(
    evaluation_run_id: str,
    fold_id: str,
    rows: Sequence[ShadowTradeEvaluation],
) -> ShadowMetricSummary:
    eligible = [row for row in rows if row.data_quality_eligible and row.origin != ObservationOrigin.RETROSPECTIVE_TRADE_LEADUP]
    count = len(eligible)
    ranks = [row.rank_at_prediction for row in eligible if row.rank_at_prediction is not None]
    lead_seconds = [Decimal(str(row.lead_time.total_seconds())) for row in eligible if row.lead_time is not None]
    ratio = lambda hits: None if count == 0 else Decimal(hits) / Decimal(count)
    return ShadowMetricSummary(
        evaluation_run_id,
        fold_id,
        count,
        ratio(sum(row.top_1 for row in eligible)),
        ratio(sum(row.top_3 for row in eligible)),
        ratio(sum(row.top_5 for row in eligible)),
        None if not lead_seconds else sum(lead_seconds) / Decimal(len(lead_seconds)),
        None if not ranks else Decimal(sum(ranks)) / Decimal(len(ranks)),
        None,
        {},
        {},
        0,
        0,
        None,
        sum(not row.data_quality_eligible for row in rows),
    )


@dataclass(frozen=True)
class ScoutMetricContract:
    """Named out-of-sample metrics; nullable values mean insufficient honest labels."""

    metric_version: str
    actual_trade_rank_distribution: Mapping[str, Decimal]
    observed_positive_precision_at_k: Mapping[int, Decimal | None]
    known_positive_recall_at_k: Mapping[int, Decimal | None]
    mean_prospective_lead_time_seconds: Decimal | None
    rank_correlation_realized_r: Decimal | None
    expected_r_calibration: Mapping[str, Decimal]
    positive_r_discrimination: Decimal | None
    two_r_discrimination: Decimal | None
    mean_rank_stability: Decimal | None
    explicit_false_high_rank_count: int
    unlabeled_high_rank_count: int
    data_quality_failure_count: int

    def __post_init__(self) -> None:
        if not self.metric_version:
            raise ValueError("metric contract requires a version")
        if any(key <= 0 for key in (*self.observed_positive_precision_at_k, *self.known_positive_recall_at_k)):
            raise ValueError("top-k metric keys must be positive")
        for value in (*self.observed_positive_precision_at_k.values(), *self.known_positive_recall_at_k.values()):
            if value is not None and not Decimal("0") <= value <= Decimal("1"):
                raise ValueError("precision/recall metrics must be in [0, 1]")
        for label, value in (
            ("positive_r_discrimination", self.positive_r_discrimination),
            ("two_r_discrimination", self.two_r_discrimination),
            ("mean_rank_stability", self.mean_rank_stability),
        ):
            if value is not None and not Decimal("0") <= value <= Decimal("1"):
                raise ValueError(f"{label} must be in [0, 1]")
        if min(
            self.explicit_false_high_rank_count,
            self.unlabeled_high_rank_count,
            self.data_quality_failure_count,
        ) < 0:
            raise ValueError("evaluation counts cannot be negative")
