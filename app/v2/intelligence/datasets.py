from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

from app.v2.domain.models import PositionSide
from app.v2.intelligence.observation import require_aware, require_exact_contract
from app.v2.learning.contracts import LearningTask


class SeenByConner(str, Enum):
    UNKNOWN = "UNKNOWN"
    CONFIRMED_SEEN = "CONFIRMED_SEEN"
    CONFIRMED_NOT_SEEN = "CONFIRMED_NOT_SEEN"


class ConsiderationStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    PRESENTED = "PRESENTED"
    CONFIRMED_CONSIDERED = "CONFIRMED_CONSIDERED"


class BehaviorLabelState(str, Enum):
    UNLABELED = "UNLABELED"
    POSITIVE_TRADE = "POSITIVE_TRADE"
    EXPLICIT_PASS = "EXPLICIT_PASS"


class ObservationOrigin(str, Enum):
    LIVE = "LIVE"
    BLIND_REPLAY = "BLIND_REPLAY"
    RETROSPECTIVE_TRADE_LEADUP = "RETROSPECTIVE_TRADE_LEADUP"


class PreferenceBasis(str, Enum):
    OBSERVED_CHOICE = "OBSERVED_CHOICE"
    EXPLICIT_TRADER_RANKING = "EXPLICIT_TRADER_RANKING"


class DatasetSourceKind(str, Enum):
    FUTURE_LIVE_MASTER = "FUTURE_LIVE_MASTER"
    BLIND_POINT_IN_TIME_REPLAY = "BLIND_POINT_IN_TIME_REPLAY"
    FINALIZED_HISTORICAL_MARKET = "FINALIZED_HISTORICAL_MARKET"
    EXPLICIT_TRADER_FEEDBACK = "EXPLICIT_TRADER_FEEDBACK"
    FOLLOWER_EXECUTION = "FOLLOWER_EXECUTION"


@dataclass(frozen=True)
class ScoutOpportunityCandidate:
    candidate_id: str
    trade_date: date
    nq_contract_id: str
    es_contract_id: str
    candidate_generator: str
    candidate_generator_version: str
    generator_parameters: Mapping[str, Any]
    first_observed_at: datetime
    latest_observed_at: datetime
    seen_by_conner: SeenByConner = SeenByConner.UNKNOWN
    consideration_status: ConsiderationStatus = ConsiderationStatus.UNKNOWN
    behavior_label_state: BehaviorLabelState = BehaviorLabelState.UNLABELED
    direction_hypothesis: PositionSide | None = None
    linked_master_trade_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    source_lineage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_exact_contract(self.nq_contract_id, "NQ")
        require_exact_contract(self.es_contract_id, "ES")
        require_aware(self.first_observed_at, "first_observed_at")
        require_aware(self.latest_observed_at, "latest_observed_at")
        if self.latest_observed_at < self.first_observed_at:
            raise ValueError("latest observation cannot precede first observation")
        if not self.candidate_generator or not self.candidate_generator_version:
            raise ValueError("candidate generator and version are required")
        if not self.source_lineage:
            raise ValueError("candidate generation requires source lineage")
        if self.behavior_label_state == BehaviorLabelState.POSITIVE_TRADE and not self.linked_master_trade_id:
            raise ValueError("positive behavior labels require an actual master trade")
        if self.behavior_label_state == BehaviorLabelState.POSITIVE_TRADE:
            if self.seen_by_conner != SeenByConner.CONFIRMED_SEEN:
                raise ValueError("an actual Conner trade is confirmed seen")
            if self.consideration_status != ConsiderationStatus.CONFIRMED_CONSIDERED or not self.evidence:
                raise ValueError("positive behavior labels require observed action evidence")
        if self.behavior_label_state == BehaviorLabelState.EXPLICIT_PASS:
            if self.seen_by_conner != SeenByConner.CONFIRMED_SEEN:
                raise ValueError("explicit pass requires evidence Conner saw the candidate")
            if self.consideration_status != ConsiderationStatus.CONFIRMED_CONSIDERED:
                raise ValueError("explicit pass requires confirmed consideration")
            if not self.evidence:
                raise ValueError("explicit pass requires evidence")


@dataclass(frozen=True)
class BehaviorLearningExample:
    example_id: str
    candidate_id: str
    snapshot_id: str
    query_group_id: str
    feature_cutoff_at: datetime
    label_state: BehaviorLabelState
    target_value: int | None
    target_observed_at: datetime | None
    seen_by_conner: SeenByConner
    consideration_status: ConsiderationStatus
    master_trade_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_aware(self.feature_cutoff_at, "feature_cutoff_at")
        if self.target_observed_at:
            require_aware(self.target_observed_at, "target_observed_at")
            if self.target_observed_at < self.feature_cutoff_at:
                raise ValueError("behavior target cannot predate feature cutoff")
        if self.label_state == BehaviorLabelState.UNLABELED:
            if self.target_value is not None or self.target_observed_at is not None:
                raise ValueError("unlabeled behavior examples must retain a null target")
        elif self.label_state == BehaviorLabelState.POSITIVE_TRADE:
            if self.target_value != 1 or not self.master_trade_id or self.target_observed_at is None:
                raise ValueError("positive behavior examples require target=1 and a real master trade")
            if (
                self.seen_by_conner != SeenByConner.CONFIRMED_SEEN
                or self.consideration_status != ConsiderationStatus.CONFIRMED_CONSIDERED
                or not self.evidence
            ):
                raise ValueError("positive behavior examples require observed action evidence")
        elif self.label_state == BehaviorLabelState.EXPLICIT_PASS:
            if self.target_value != 0 or self.target_observed_at is None:
                raise ValueError("explicit pass requires target=0")
            if self.master_trade_id is not None:
                raise ValueError("explicit pass cannot link a master trade")
            if self.seen_by_conner != SeenByConner.CONFIRMED_SEEN:
                raise ValueError("negative behavior labels require confirmed exposure")
            if self.consideration_status != ConsiderationStatus.CONFIRMED_CONSIDERED or not self.evidence:
                raise ValueError("negative behavior labels require confirmed consideration evidence")


@dataclass(frozen=True)
class PairwisePreference:
    preference_id: str
    query_group_id: str
    preferred_candidate_id: str
    other_candidate_id: str
    observed_at: datetime
    basis: PreferenceBasis
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_aware(self.observed_at, "observed_at")
        if self.preferred_candidate_id == self.other_candidate_id:
            raise ValueError("pairwise preference requires two candidates")
        if not self.evidence:
            raise ValueError("pairwise preferences cannot be inferred without evidence")


@dataclass(frozen=True)
class TaskDatasetExample:
    example_id: str
    task: LearningTask
    candidate_id: str | None
    snapshot_id: str | None
    query_group_id: str | None
    feature_cutoff_at: datetime
    target_name: str | None
    target_value: Any | None
    target_observed_at: datetime | None
    source_kind: DatasetSourceKind
    source_lineage: Mapping[str, Any]
    behavior_label_state: BehaviorLabelState | None = None
    eligible: bool = True
    exclusion_reason: str | None = None
    master_trade_id: str | None = None
    follower_trade_id: str | None = None

    def __post_init__(self) -> None:
        require_aware(self.feature_cutoff_at, "feature_cutoff_at")
        if not isinstance(self.source_kind, DatasetSourceKind):
            raise ValueError("dataset source_kind must use the closed V2 observed-data enum")
        if not self.source_lineage:
            raise ValueError("dataset examples require source lineage")
        if self.target_observed_at:
            require_aware(self.target_observed_at, "target_observed_at")
            if self.target_observed_at < self.feature_cutoff_at:
                raise ValueError("target cannot predate feature cutoff")
        namespace = str(self.source_lineage.get("namespace", "")).lower()
        if namespace.startswith("v1") or any(token in namespace for token in ("synthetic", "sandbox", "simulation", "equity")):
            raise ValueError("synthetic, sandbox, or V1 equity data cannot enter V2 futures datasets")
        if self.target_value is None and self.target_observed_at is not None:
            raise ValueError("a target observation timestamp requires an observed target")
        if self.target_value is not None and self.target_observed_at is None:
            raise ValueError("observed targets require a target observation timestamp")
        if self.task == LearningTask.BEHAVIOR_IMITATION:
            if self.behavior_label_state is None:
                raise ValueError("behavior examples require an explicit PU label state")
            if self.behavior_label_state == BehaviorLabelState.UNLABELED and self.target_value is not None:
                raise ValueError("unlabeled behavior examples cannot be coerced to zero")
            if self.behavior_label_state == BehaviorLabelState.POSITIVE_TRADE and not self.master_trade_id:
                raise ValueError("positive behavior examples require an actual master trade")
            if self.behavior_label_state == BehaviorLabelState.EXPLICIT_PASS and self.master_trade_id is not None:
                raise ValueError("explicit passes cannot link an executed master trade")
            if self.source_kind == DatasetSourceKind.FINALIZED_HISTORICAL_MARKET:
                raise ValueError("finalized market history cannot fabricate Conner behavior labels")
        elif self.behavior_label_state is not None:
            raise ValueError("behavior label state belongs only to the behavior task")
        if self.task == LearningTask.CONVICTION_BEHAVIOR and self.source_kind != DatasetSourceKind.FUTURE_LIVE_MASTER:
            raise ValueError("conviction labels require observed future live master actions")
        if self.task == LearningTask.CONVICTION_BEHAVIOR and not self.master_trade_id:
            raise ValueError("conviction labels require an observed master trade")
        if self.task == LearningTask.COPIER_EXECUTION_QUALITY:
            if self.source_kind != DatasetSourceKind.FOLLOWER_EXECUTION:
                raise ValueError("execution-quality examples require follower execution facts")
            if self.candidate_id is not None or self.snapshot_id is not None:
                raise ValueError("follower execution examples cannot populate strategy observations")
            if not self.follower_trade_id:
                raise ValueError("execution-quality examples require a follower trade")
        else:
            if self.source_kind == DatasetSourceKind.FOLLOWER_EXECUTION:
                raise ValueError("follower execution facts cannot label strategy tasks")
            if not self.candidate_id or not self.snapshot_id:
                raise ValueError("strategy examples require a candidate and point-in-time snapshot")
            if self.follower_trade_id is not None:
                raise ValueError("follower trades cannot link strategy learning examples")
        if not self.eligible and not self.exclusion_reason:
            raise ValueError("excluded examples require a reason")


@dataclass(frozen=True)
class ActualConnerTradeLabel:
    label_id: str
    master_trade_id: str
    nq_contract_id: str
    es_contract_id: str
    action_anchor_at: datetime
    action_anchor_kind: str
    direction: PositionSide
    original_stop_observed: bool
    original_stop_price: Decimal | None
    initial_quantity: int
    initial_risk_dollars: Decimal | None
    evidence: Mapping[str, Any]
    linked_candidate_id: str | None = None
    match_method: str | None = None
    match_version: str | None = None
    match_confidence: Decimal | None = None

    def __post_init__(self) -> None:
        require_exact_contract(self.nq_contract_id, "NQ")
        require_exact_contract(self.es_contract_id, "ES")
        require_aware(self.action_anchor_at, "action_anchor_at")
        if self.action_anchor_kind not in {"DECISION", "FIRST_ORDER_SUBMISSION", "FIRST_FILL"}:
            raise ValueError("trade label must state the actual action anchor semantics")
        if self.direction == PositionSide.FLAT:
            raise ValueError("an actual trade label must be LONG or SHORT")
        if self.initial_quantity <= 0:
            raise ValueError("initial quantity must be positive")
        if self.original_stop_observed != (self.original_stop_price is not None):
            raise ValueError("original stop price must reflect whether a stop was actually observed")
        if not self.original_stop_observed and self.initial_risk_dollars is not None:
            raise ValueError("initial R cannot be inferred when no original stop was observed")
        if self.initial_risk_dollars is not None and self.initial_risk_dollars <= Decimal("0"):
            raise ValueError("initial risk dollars must be positive when observed")
        if not self.evidence:
            raise ValueError("actual Conner trade labels require lifecycle evidence")
        if self.linked_candidate_id and not self.match_method:
            raise ValueError("candidate links require a versioned match method")
        if self.linked_candidate_id and (not self.match_version or self.match_confidence is None):
            raise ValueError("candidate links require match version and confidence")
        if self.match_confidence is not None and not Decimal("0") <= self.match_confidence <= Decimal("1"):
            raise ValueError("candidate match confidence must be in [0, 1]")


@dataclass(frozen=True)
class OutcomeLabel:
    label_id: str
    master_trade_id: str
    candidate_id: str | None
    feature_cutoff_at: datetime
    outcome_observed_at: datetime
    realized_r: Decimal | None
    mfe_r: Decimal | None
    mae_r: Decimal | None
    holding_time_seconds: Decimal | None
    original_risk_observed: bool
    source_lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_aware(self.feature_cutoff_at, "feature_cutoff_at")
        require_aware(self.outcome_observed_at, "outcome_observed_at")
        if self.outcome_observed_at < self.feature_cutoff_at:
            raise ValueError("outcome labels must be observed after the feature cutoff")
        if not self.original_risk_observed and any(value is not None for value in (self.realized_r, self.mfe_r, self.mae_r)):
            raise ValueError("R-denominated outcomes require an observed original risk denominator")
        if self.holding_time_seconds is not None and self.holding_time_seconds < Decimal("0"):
            raise ValueError("holding time cannot be negative")
        if not self.source_lineage:
            raise ValueError("outcome labels require source lineage")


@dataclass(frozen=True)
class ConvictionLabel:
    label_id: str
    master_trade_id: str
    observed_at: datetime
    initial_quantity: int
    initial_planned_risk_dollars: Decimal | None
    initial_risk_tier: str | None
    source_lineage: Mapping[str, Any]
    can_control_follower_risk: bool = False

    def __post_init__(self) -> None:
        require_aware(self.observed_at, "observed_at")
        if self.initial_quantity <= 0:
            raise ValueError("conviction labels require a positive observed initial quantity")
        if self.initial_planned_risk_dollars is not None and self.initial_planned_risk_dollars <= Decimal("0"):
            raise ValueError("planned risk must be positive when observed")
        if self.initial_planned_risk_dollars is None and not self.initial_risk_tier:
            raise ValueError("conviction labels require an honestly observed target")
        if not self.source_lineage:
            raise ValueError("conviction labels require source lineage")
        if self.can_control_follower_risk:
            raise ValueError("learned conviction can never control follower risk")


@dataclass(frozen=True)
class ExecutionQualityLabel:
    label_id: str
    master_trade_id: str
    follower_trade_id: str
    observed_at: datetime
    latency_ms: Decimal | None
    adverse_fill_ticks: Decimal | None
    fill_ratio: Decimal | None
    rejected_order_count: int
    bracket_failure_count: int
    reconciliation_seconds: Decimal | None
    source_lineage: Mapping[str, Any]
    strategy_label_eligible: bool = False

    def __post_init__(self) -> None:
        require_aware(self.observed_at, "observed_at")
        if self.fill_ratio is not None and not Decimal("0") <= self.fill_ratio <= Decimal("1"):
            raise ValueError("fill ratio must be in [0, 1]")
        if self.rejected_order_count < 0 or self.bracket_failure_count < 0:
            raise ValueError("execution failure counts cannot be negative")
        if self.latency_ms is not None and self.latency_ms < Decimal("0"):
            raise ValueError("execution latency cannot be negative")
        if self.adverse_fill_ticks is not None and self.adverse_fill_ticks < Decimal("0"):
            raise ValueError("adverse fill ticks cannot be negative")
        if self.reconciliation_seconds is not None and self.reconciliation_seconds < Decimal("0"):
            raise ValueError("reconciliation time cannot be negative")
        if not self.source_lineage:
            raise ValueError("execution-quality labels require source lineage")
        if self.strategy_label_eligible:
            raise ValueError("follower execution quality can never label strategy quality")
