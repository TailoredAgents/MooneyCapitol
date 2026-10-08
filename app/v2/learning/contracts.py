from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class LearningTask(str, Enum):
    BEHAVIOR_IMITATION = "behavior_imitation"
    SETUP_OUTCOME_QUALITY = "setup_outcome_quality"
    CONVICTION_BEHAVIOR = "conviction_behavior"
    COPIER_EXECUTION_QUALITY = "copier_execution_quality"


class PromotionStatus(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PROMOTED = "promoted"
    RETIRED = "retired"


class FeatureTemporalRole(str, Enum):
    """The point-in-time role a field plays in a learning dataset.

    Only values that existed at the decision cutoff may be predictors.  The
    other roles are explicit so a label or execution fact cannot be smuggled
    into a feature schema merely by giving it an innocuous name.
    """

    PREDICTOR = "predictor"
    AT_CUTOFF_CONTEXT = "at_cutoff_context"
    OUTCOME = "outcome"
    FUTURE = "future"
    FOLLOWER_EXECUTION = "follower_execution"


class FeatureSourceDomain(str, Enum):
    MARKET_STATE = "market_state"
    SESSION_CONTEXT = "session_context"
    MASTER_PRE_DECISION = "master_pre_decision"
    OUTCOME_ANALYSIS = "outcome_analysis"
    FOLLOWER_EXECUTION = "follower_execution"


class LearningSourceKind(str, Enum):
    """Closed set of honest V2 data origins.

    There is deliberately no synthetic, sandbox, or V1 member.
    """

    OBSERVED = "observed"  # Backwards-compatible name for captured facts.
    LIVE_OBSERVED = "live_observed"
    BLIND_POINT_IN_TIME_REPLAY = "blind_point_in_time_replay"
    FINALIZED_HISTORICAL_MARKET = "finalized_historical_market"
    HISTORICAL_REPLAY = "historical_replay"
    RETROSPECTIVE_TRADE_LEADUP = "retrospective_trade_leadup"
    BROKER_LIFECYCLE = "broker_lifecycle"


class LearningSourceNamespace(str, Enum):
    """Allowed source namespaces for V2 futures learning facts."""

    V2_FUTURES = "v2.futures"
    CONNER_NQ_OBSERVATION = "v2.futures.conner_nq_v1.observation"
    MARKET_DATA = "v2.futures.market_data"
    MASTER_TRADES = "v2.futures.master_trades"
    COPIER_EXECUTION = "v2.futures.copier_execution"


class LabelState(str, Enum):
    UNLABELED = "unlabeled"
    POSITIVE = "positive"
    EXPLICIT_NEGATIVE = "explicit_negative"


class SeenByConner(str, Enum):
    UNKNOWN = "unknown"
    YES = "yes"
    NO = "no"


def _lineage_contains_quarantined_source(value: Any) -> bool:
    """Small construction-time backstop for persisted model metadata."""

    if isinstance(value, Mapping):
        asset_class = value.get("asset_class")
        if asset_class is not None and str(asset_class).strip().lower() != "futures":
            return True
        return any(_lineage_contains_quarantined_source(item) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_lineage_contains_quarantined_source(item) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"equity", "stock", "sandbox", "synthetic", "v1", "v1_equity"} or any(
        marker in normalized
        for marker in ("equity_sandbox", "hybrid_target", "randomized_equity", "synthetic", "sandbox", "v1_equity")
    )


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    dtype: str
    imputation_rule: str | None = None
    temporal_role: FeatureTemporalRole = FeatureTemporalRole.PREDICTOR
    source_domain: FeatureSourceDomain = FeatureSourceDomain.MARKET_STATE
    definition_version: str = "1"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureObservation:
    name: str
    value: Any
    observed_at: datetime
    source: str
    was_missing: bool = False
    imputation_rule: str | None = None
    temporal_role: FeatureTemporalRole | str | None = None
    source_domain: FeatureSourceDomain | str | None = None
    event_at: datetime | None = None
    available_at: datetime | None = None
    computed_at: datetime | None = None
    source_kind: LearningSourceKind | str = LearningSourceKind.OBSERVED
    source_namespace: LearningSourceNamespace | str = LearningSourceNamespace.MARKET_DATA
    lineage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effective_event_at(self) -> datetime:
        """Market/event time, falling back to the legacy observed_at field."""

        return self.event_at or self.observed_at

    @property
    def effective_available_at(self) -> datetime:
        """Earliest time this exact data revision could have been known."""

        return self.available_at or self.observed_at

    @property
    def effective_computed_at(self) -> datetime:
        """Time the derived value was materialized.

        Computation may occur after a historical cutoff; its inputs still must
        have been available by that cutoff.
        """

        return self.computed_at or self.effective_available_at


@dataclass(frozen=True)
class FeatureSnapshot:
    opportunity_id: str
    contract_id: str
    feature_cutoff_at: datetime
    feature_schema_version: str
    observations: tuple[FeatureObservation, ...]
    materialized_at: datetime | None = None
    context_contract_ids: Mapping[str, str] = field(default_factory=dict)
    lineage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effective_materialized_at(self) -> datetime:
        return self.materialized_at or self.feature_cutoff_at


@dataclass(frozen=True)
class DatasetContract:
    task: LearningTask
    dataset_version: str
    feature_schema_version: str
    features: tuple[FeatureDefinition, ...]
    target_name: str
    source_lineage: Mapping[str, Any]
    traded_product_codes: frozenset[str] = field(default_factory=lambda: frozenset({"NQ"}))
    context_product_codes: frozenset[str] = field(default_factory=frozenset)
    source_namespace: LearningSourceNamespace | str = LearningSourceNamespace.V2_FUTURES

    @property
    def artifact_namespace(self) -> str:
        return f"v2.futures.{self.task.value}"


@dataclass(frozen=True)
class DatasetRow:
    task: LearningTask
    opportunity_id: str
    snapshot: FeatureSnapshot
    target_name: str
    target_value: Any
    target_observed_at: datetime | None
    conner_action_observed: bool = False
    strategy_quality_label: Any | None = None
    follower_execution_quality: Mapping[str, Any] | None = None
    source_kind: LearningSourceKind | str = LearningSourceKind.OBSERVED
    source_namespace: LearningSourceNamespace | str = LearningSourceNamespace.V2_FUTURES
    label_state: LabelState | str | None = None
    seen_by_conner: SeenByConner | str = SeenByConner.UNKNOWN
    label_evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelArtifactMetadata:
    artifact_id: str
    task: LearningTask
    feature_schema_version: str
    dataset_version: str
    artifact_namespace: str
    training_metrics: Mapping[str, Any]
    promotion_status: PromotionStatus
    source_lineage: Mapping[str, Any]
    asset_class: str = "futures"
    product_codes: frozenset[str] = field(default_factory=lambda: frozenset({"NQ", "MNQ"}))
    created_at: datetime | None = None
    context_product_codes: frozenset[str] = field(default_factory=frozenset)
    source_namespace: LearningSourceNamespace | str = LearningSourceNamespace.V2_FUTURES

    @property
    def traded_product_codes(self) -> frozenset[str]:
        """Compatibility-preserving explicit name for the prediction targets."""

        return self.product_codes

    def __post_init__(self) -> None:
        if self.asset_class.lower() != "futures":
            raise ValueError("V2 model artifacts must be futures-only")
        if self.artifact_namespace != f"v2.futures.{self.task.value}":
            raise ValueError("artifact namespace does not match its isolated learning task")
        if not self.product_codes or not self.product_codes.issubset({"NQ", "MNQ"}):
            raise ValueError("V2 foundation artifacts may only declare NQ/MNQ")
        if not self.context_product_codes.issubset({"ES"}):
            raise ValueError("V2 Conner-NQ context products may only declare ES")
        if self.product_codes & self.context_product_codes:
            raise ValueError("traded and contextual product roles must remain separate")
        if _lineage_contains_quarantined_source(self.source_lineage):
            raise ValueError("V1 equity/sandbox/synthetic lineage is quarantined from V2 model artifacts")
        try:
            LearningSourceNamespace(self.source_namespace)
        except ValueError as exc:
            raise ValueError("model artifact has an unapproved learning source namespace") from exc
