from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Mapping, TypeVar

from app.v2.learning.contracts import (
    DatasetContract,
    DatasetRow,
    FeatureSnapshot,
    FeatureSourceDomain,
    FeatureTemporalRole,
    LabelState,
    LearningSourceKind,
    LearningSourceNamespace,
    LearningTask,
    SeenByConner,
)


class LeakageError(ValueError):
    pass


OUTCOME_ONLY_NAMES = frozenset(
    {
        "eventual_outcome",
        "exit_price",
        "holding_time",
        "mae_r",
        "mfe_r",
        "net_pnl",
        "realized_net_pnl",
        "realized_pnl",
        "realized_r",
        "trade_result",
    }
)
FOLLOWER_EXECUTION_NAMES = frozenset(
    {
        "adverse_fill_ticks",
        "bracket_failure",
        "copier_latency",
        "desynchronization",
        "fill_ratio",
        "follower_execution_quality",
        "follower_latency",
        "reconciliation_time",
        "rejected_order",
    }
)
OUTCOME_PREFIXES = ("eventual_", "future_", "post_entry_", "realized_")
FOLLOWER_PREFIXES = ("copier_", "follower_")

_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum(value: Any, enum_type: type[_EnumT], label: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise LeakageError(f"unapproved {label}: {value!s}") from exc


def _aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LeakageError(f"{label} must be timezone-aware")


def _feature_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _is_outcome_name(name: str) -> bool:
    normalized = _feature_name(name)
    return normalized in OUTCOME_ONLY_NAMES or normalized.startswith(OUTCOME_PREFIXES)


def _is_follower_name(name: str) -> bool:
    normalized = _feature_name(name)
    return normalized in FOLLOWER_EXECUTION_NAMES or normalized.startswith(FOLLOWER_PREFIXES)


def _validate_lineage(value: Any, path: str = "lineage") -> None:
    """Reject legacy/sandbox provenance without interpreting provider payloads.

    Namespace and source-kind fields are closed enums. Other strings receive a
    narrow deny check so ordinary concepts such as account equity are not
    accidentally rejected while known legacy sources still fail closed.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            child_path = f"{path}.{key}"
            if key == "asset_class" and str(child).strip().lower() != "futures":
                raise LeakageError(f"{child_path} must be futures, not {child!s}")
            if key == "source_kind":
                _enum(child, LearningSourceKind, child_path)
            elif key in {"source_namespace", "dataset_namespace"}:
                _enum(child, LearningSourceNamespace, child_path)
            _validate_lineage(child, child_path)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for index, child in enumerate(value):
            _validate_lineage(child, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    exact_blocked = {"equity", "stock", "sandbox", "synthetic", "v1", "v1_equity"}
    blocked_fragments = (
        "equity_sandbox",
        "hybrid_target",
        "randomized_equity",
        "stock_sandbox",
        "synthetic",
        "sandbox",
        "v1.equity",
        "v1_equity",
    )
    if normalized in exact_blocked or any(fragment in normalized for fragment in blocked_fragments):
        raise LeakageError(f"disallowed V1/equity/synthetic learning source at {path}: {value}")


def _internal_product(contract_id: str, label: str) -> str:
    parts = contract_id.split(":")
    if len(parts) < 3 or not parts[1].strip():
        raise LeakageError(f"{label} must use an exact internal futures contract id")
    return parts[1].upper()


def _validate_contract_products(snapshot: FeatureSnapshot, contract: DatasetContract) -> None:
    traded = frozenset(item.upper() for item in contract.traded_product_codes)
    context = frozenset(item.upper() for item in contract.context_product_codes)
    if not traded or not traded.issubset({"NQ", "MNQ"}):
        raise LeakageError("traded products must be explicitly limited to NQ/MNQ")
    if not context.issubset({"ES"}):
        raise LeakageError("Conner-NQ contextual products must be explicitly limited to ES")
    if traded & context:
        raise LeakageError("traded and contextual products cannot overlap")

    snapshot_product = _internal_product(snapshot.contract_id, "snapshot contract")
    if snapshot_product not in traded:
        raise LeakageError(f"snapshot product {snapshot_product} is not a declared traded product")
    for declared_product, context_contract_id in snapshot.context_contract_ids.items():
        product = declared_product.upper()
        if product not in context:
            raise LeakageError(f"undeclared contextual product: {product}")
        if _internal_product(context_contract_id, f"{product} context contract") != product:
            raise LeakageError(f"context contract does not match its {product} product role")


def _validate_source_boundary(
    *,
    task: LearningTask,
    source_kind: LearningSourceKind | str,
    source_namespace: LearningSourceNamespace | str,
) -> None:
    kind = _enum(source_kind, LearningSourceKind, "learning source kind")
    namespace = _enum(source_namespace, LearningSourceNamespace, "learning source namespace")
    if task != LearningTask.COPIER_EXECUTION_QUALITY and namespace == LearningSourceNamespace.COPIER_EXECUTION:
        raise LeakageError("copier/follower execution sources cannot enter strategy learning tasks")
    if task == LearningTask.BEHAVIOR_IMITATION and kind in {
        LearningSourceKind.FINALIZED_HISTORICAL_MARKET,
        LearningSourceKind.HISTORICAL_REPLAY,
        LearningSourceKind.RETROSPECTIVE_TRADE_LEADUP,
    }:
        raise LeakageError(
            "behavior learning requires live or explicitly blind point-in-time facts, not finalized/retrospective history"
        )
    if task == LearningTask.COPIER_EXECUTION_QUALITY and kind == LearningSourceKind.RETROSPECTIVE_TRADE_LEADUP:
        raise LeakageError("strategy lead-up snapshots cannot enter copier execution learning")


def _validate_temporal_lineage(snapshot: FeatureSnapshot, observation) -> None:
    cutoff = snapshot.feature_cutoff_at
    materialized_at = snapshot.effective_materialized_at
    observed_at = observation.observed_at
    event_at = observation.effective_event_at
    available_at = observation.effective_available_at
    computed_at = observation.effective_computed_at

    for label, value in (
        ("feature cutoff", cutoff),
        ("snapshot materialized_at", materialized_at),
        (f"{observation.name}.observed_at", observed_at),
        (f"{observation.name}.event_at", event_at),
        (f"{observation.name}.available_at", available_at),
        (f"{observation.name}.computed_at", computed_at),
    ):
        _aware(value, label)

    if materialized_at < cutoff:
        raise LeakageError("snapshot cannot be materialized before its feature cutoff")
    if observed_at > cutoff or event_at > cutoff:
        raise LeakageError(f"future/post-cutoff event data detected: {observation.name}")
    if available_at > cutoff:
        raise LeakageError(f"future knowledge/data revision detected: {observation.name}")
    if available_at < event_at:
        raise LeakageError(f"data cannot be available before its event time: {observation.name}")
    if computed_at < available_at:
        raise LeakageError(f"feature cannot be computed before its inputs are available: {observation.name}")
    if computed_at > materialized_at:
        raise LeakageError(f"feature computation follows snapshot materialization: {observation.name}")

    # Derived-feature builders may also provide aggregate input maxima. These
    # are checked independently so a safe-looking output timestamp cannot hide
    # one future input.
    for key in ("max_input_event_at", "max_input_available_at"):
        value = observation.lineage.get(key)
        if value is None:
            continue
        if not isinstance(value, datetime):
            raise LeakageError(f"{observation.name}.{key} must be a datetime")
        _aware(value, f"{observation.name}.{key}")
        if value > cutoff:
            raise LeakageError(f"future input lineage detected: {observation.name}.{key}")


def validate_feature_snapshot(snapshot: FeatureSnapshot, contract: DatasetContract) -> None:
    if snapshot.feature_schema_version != contract.feature_schema_version:
        raise LeakageError("snapshot does not match the contract feature schema version")
    _aware(snapshot.feature_cutoff_at, "feature cutoff")
    _aware(snapshot.effective_materialized_at, "snapshot materialized_at")
    _enum(contract.source_namespace, LearningSourceNamespace, "dataset contract source namespace")
    _validate_lineage(contract.source_lineage, "dataset_contract.source_lineage")
    _validate_lineage(snapshot.lineage, "snapshot.lineage")
    _validate_contract_products(snapshot, contract)

    definitions: dict[str, Any] = {}
    for definition in contract.features:
        if definition.name in definitions:
            raise LeakageError(f"duplicate feature definition: {definition.name}")
        definitions[definition.name] = definition

    observed_names: set[str] = set()
    for observation in snapshot.observations:
        if observation.name in observed_names:
            raise LeakageError(f"duplicate feature: {observation.name}")
        observed_names.add(observation.name)
        if observation.name not in definitions:
            raise LeakageError(f"feature is not declared in schema: {observation.name}")
        if observation.name == contract.target_name:
            raise LeakageError("prediction target cannot also be its own feature")

        definition = definitions[observation.name]
        role = _enum(definition.temporal_role, FeatureTemporalRole, f"{observation.name} temporal role")
        source_domain = _enum(definition.source_domain, FeatureSourceDomain, f"{observation.name} source domain")
        if observation.temporal_role is not None:
            observed_role = _enum(observation.temporal_role, FeatureTemporalRole, f"{observation.name} observation temporal role")
            if observed_role != role:
                raise LeakageError(f"feature temporal role differs from its schema: {observation.name}")
        if observation.source_domain is not None:
            observed_domain = _enum(observation.source_domain, FeatureSourceDomain, f"{observation.name} observation source domain")
            if observed_domain != source_domain:
                raise LeakageError(f"feature source domain differs from its schema: {observation.name}")

        if role in {FeatureTemporalRole.OUTCOME, FeatureTemporalRole.FUTURE}:
            raise LeakageError(f"outcome/future field cannot be a prediction feature: {observation.name}")
        if source_domain == FeatureSourceDomain.OUTCOME_ANALYSIS or _is_outcome_name(observation.name):
            raise LeakageError(f"post-outcome field cannot be an entry feature: {observation.name}")
        follower_feature = (
            role == FeatureTemporalRole.FOLLOWER_EXECUTION
            or source_domain == FeatureSourceDomain.FOLLOWER_EXECUTION
            or _is_follower_name(observation.name)
        )
        if follower_feature and contract.task != LearningTask.COPIER_EXECUTION_QUALITY:
            raise LeakageError(f"follower execution feature cannot enter {contract.task.value}: {observation.name}")
        if contract.task == LearningTask.BEHAVIOR_IMITATION and follower_feature:
            raise LeakageError(f"behavior inputs cannot contain follower execution facts: {observation.name}")

        _validate_source_boundary(
            task=contract.task,
            source_kind=observation.source_kind,
            source_namespace=observation.source_namespace,
        )
        _validate_lineage(observation.lineage, f"feature.{observation.name}.lineage")
        _validate_temporal_lineage(snapshot, observation)

        if observation.was_missing:
            rule = observation.imputation_rule or definition.imputation_rule
            if not rule:
                raise LeakageError(f"missing feature has no explicit imputation rule: {observation.name}")
            if observation.value == 0 and "zero" not in rule.lower():
                raise LeakageError(f"missing value silently became zero: {observation.name}")

    missing = set(definitions) - observed_names
    if missing:
        raise LeakageError(f"feature snapshot omitted declared fields: {', '.join(sorted(missing))}")


def validate_dataset_row(row: DatasetRow, contract: DatasetContract) -> None:
    if row.task != contract.task or row.target_name != contract.target_name:
        raise LeakageError("dataset row does not match its task contract")
    _validate_source_boundary(task=row.task, source_kind=row.source_kind, source_namespace=row.source_namespace)
    _validate_lineage(contract.source_lineage, "dataset_contract.source_lineage")

    label_state = None if row.label_state is None else _enum(row.label_state, LabelState, "label state")
    seen = _enum(row.seen_by_conner, SeenByConner, "seen_by_conner state")
    if label_state == LabelState.UNLABELED:
        if row.target_value is not None or row.target_observed_at is not None:
            raise LeakageError("unlabeled examples must not contain an artificial target")
    else:
        if row.target_observed_at is None:
            raise LeakageError("a labeled target requires target_observed_at")
        _aware(row.target_observed_at, "target_observed_at")
        if row.target_observed_at < row.snapshot.feature_cutoff_at:
            raise LeakageError("target cannot precede the feature cutoff")

    validate_feature_snapshot(row.snapshot, contract)

    if row.task == LearningTask.BEHAVIOR_IMITATION:
        if label_state is None and not row.conner_action_observed:
            raise LeakageError("unseen opportunities cannot be labeled as Conner passes")
        if label_state == LabelState.UNLABELED and row.conner_action_observed:
            raise LeakageError("an observed Conner action cannot remain behavior-unlabeled")
        if label_state == LabelState.POSITIVE and not row.conner_action_observed:
            raise LeakageError("a positive behavior label requires an observed Conner action")
        if label_state == LabelState.EXPLICIT_NEGATIVE:
            if seen != SeenByConner.YES or not row.label_evidence:
                raise LeakageError("an explicit behavior negative requires presentation/consideration evidence")
        if seen == SeenByConner.NO and row.conner_action_observed:
            raise LeakageError("Conner action evidence conflicts with seen_by_conner=no")

    if row.follower_execution_quality is not None and row.task != LearningTask.COPIER_EXECUTION_QUALITY:
        raise LeakageError("follower execution results cannot enter strategy learning tasks")
    if row.strategy_quality_label is not None and row.task != LearningTask.SETUP_OUTCOME_QUALITY:
        raise LeakageError("strategy quality labels belong only to the setup-outcome task")
    if row.strategy_quality_label is not None and row.follower_execution_quality is not None:
        raise LeakageError("follower execution results cannot label strategy quality")
    if row.target_name == "projected_rr" and row.task == LearningTask.SETUP_OUTCOME_QUALITY:
        raise LeakageError("projected R:R cannot substitute for realized outcome/R")
