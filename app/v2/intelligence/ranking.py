from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping

from app.v2.domain.models import PositionSide
from app.v2.intelligence.datasets import ObservationOrigin
from app.v2.intelligence.observation import require_aware, require_exact_contract
from app.v2.learning.contracts import LearningTask


@dataclass(frozen=True)
class RankingComponent:
    component_name: str
    task_namespace: str
    raw_output: Mapping[str, Any]
    normalized_score: Decimal | None
    model_artifact_id: str | None
    confidence: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.task_namespace.startswith("v2.futures."):
            raise ValueError("ranking components must retain isolated V2 task namespaces")
        if self.task_namespace.endswith(LearningTask.COPIER_EXECUTION_QUALITY.value):
            raise ValueError("follower execution quality cannot rank strategy opportunities")
        if self.normalized_score is not None and not Decimal("0") <= self.normalized_score <= Decimal("1"):
            raise ValueError("normalized component score must be in [0, 1]")
        if self.confidence is not None and not Decimal("0") <= self.confidence <= Decimal("1"):
            raise ValueError("component confidence must be in [0, 1]")


@dataclass(frozen=True)
class RankedScoutOpportunity:
    candidate_id: str
    rank: int
    display_score: Decimal | None
    behavior: RankingComponent | None
    outcome: RankingComponent | None
    conviction: RankingComponent | None
    model_confidence: Decimal | None
    data_quality_score: Decimal | None
    display_explanation: Mapping[str, Any] = field(default_factory=dict)
    display_score_training_target: bool = False
    predicted_direction: PositionSide | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        for label, value in (("model_confidence", self.model_confidence), ("data_quality_score", self.data_quality_score)):
            if value is not None and not Decimal("0") <= value <= Decimal("1"):
                raise ValueError(f"{label} must be in [0, 1]")
        if self.display_score is not None and not Decimal("0") <= self.display_score <= Decimal("1"):
            raise ValueError("display score must be in [0, 1]")
        expected = (
            ("behavior", self.behavior, LearningTask.BEHAVIOR_IMITATION),
            ("outcome", self.outcome, LearningTask.SETUP_OUTCOME_QUALITY),
            ("conviction", self.conviction, LearningTask.CONVICTION_BEHAVIOR),
        )
        for label, component, task in expected:
            if component is not None and component.task_namespace != f"v2.futures.{task.value}":
                raise ValueError(f"{label} ranking output has the wrong isolated task namespace")
        if self.display_score_training_target:
            raise ValueError("combined display score is never a learning target")
        if self.predicted_direction == PositionSide.FLAT:
            raise ValueError("a ranked direction hypothesis must be LONG, SHORT, or unknown")


@dataclass(frozen=True)
class ScoutRankingSnapshot:
    ranking_id: str
    query_group_id: str
    trade_date: date
    feature_cutoff_at: datetime
    generated_at: datetime
    nq_contract_id: str
    es_contract_id: str
    ranking_policy_version: str
    candidate_generator_version: str
    origin: ObservationOrigin
    opportunities: tuple[RankedScoutOpportunity, ...]
    model_versions: Mapping[str, str]
    data_quality: Mapping[str, Any]
    observation_spec_version: str
    feature_schema_version: str
    source_lineage: Mapping[str, Any]
    execution_eligible: bool = False
    shadow_only: bool = True

    def __post_init__(self) -> None:
        require_aware(self.feature_cutoff_at, "feature_cutoff_at")
        require_aware(self.generated_at, "generated_at")
        require_exact_contract(self.nq_contract_id, "NQ")
        require_exact_contract(self.es_contract_id, "ES")
        if self.generated_at < self.feature_cutoff_at:
            raise ValueError("ranking cannot be generated before its cutoff")
        ranks = [item.rank for item in self.opportunities]
        candidates = [item.candidate_id for item in self.opportunities]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("ranking must contain unique consecutive ranks beginning at one")
        if len(candidates) != len(set(candidates)):
            raise ValueError("a candidate can appear only once in a ranking")
        if not self.ranking_policy_version or not self.candidate_generator_version:
            raise ValueError("ranking and candidate policies must be versioned")
        if not self.observation_spec_version or not self.feature_schema_version or not self.source_lineage:
            raise ValueError("ranking requires observation/feature versions and source lineage")
        if any("execution" in key.lower() or "copier" in key.lower() for key in self.model_versions):
            raise ValueError("copier execution models cannot enter strategy rankings")
        if self.execution_eligible:
            raise ValueError("Scout rankings are display/observation only and never execution eligible")
        if not self.shadow_only:
            raise ValueError("conner_nq_v1 rankings are shadow-only in this phase")
