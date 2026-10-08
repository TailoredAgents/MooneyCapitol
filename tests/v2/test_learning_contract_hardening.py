from datetime import datetime, timedelta, timezone

import pytest

from app.v2.learning.contracts import (
    DatasetContract,
    DatasetRow,
    FeatureDefinition,
    FeatureObservation,
    FeatureSnapshot,
    FeatureSourceDomain,
    FeatureTemporalRole,
    LabelState,
    LearningSourceKind,
    LearningSourceNamespace,
    LearningTask,
    ModelArtifactMetadata,
    PromotionStatus,
    SeenByConner,
)
from app.v2.learning.validation import LeakageError, validate_dataset_row, validate_feature_snapshot


CUTOFF = datetime(2026, 10, 5, 14, tzinfo=timezone.utc)


def _contract(
    *,
    task: LearningTask = LearningTask.BEHAVIOR_IMITATION,
    role: FeatureTemporalRole = FeatureTemporalRole.PREDICTOR,
    domain: FeatureSourceDomain = FeatureSourceDomain.MARKET_STATE,
    source_lineage=None,
) -> DatasetContract:
    return DatasetContract(
        task=task,
        dataset_version="conner-nq-d1",
        feature_schema_version="conner-nq-f1",
        features=(FeatureDefinition("candidate_measurement", "decimal", temporal_role=role, source_domain=domain),),
        target_name="target",
        source_lineage=source_lineage or {"source_kind": LearningSourceKind.OBSERVED.value, "asset_class": "futures"},
        traded_product_codes=frozenset({"NQ"}),
        context_product_codes=frozenset({"ES"}),
        source_namespace=LearningSourceNamespace.CONNER_NQ_OBSERVATION,
    )


def _observation(**changes) -> FeatureObservation:
    values = dict(
        name="candidate_measurement",
        value="1.25",
        observed_at=CUTOFF - timedelta(seconds=2),
        source="research-feed",
        event_at=CUTOFF - timedelta(seconds=2),
        available_at=CUTOFF - timedelta(seconds=1),
        computed_at=CUTOFF,
    )
    values.update(changes)
    return FeatureObservation(**values)


def _snapshot(observation: FeatureObservation, **changes) -> FeatureSnapshot:
    values = dict(
        opportunity_id="opp-1",
        contract_id="CME:NQ:2027-03-19",
        feature_cutoff_at=CUTOFF,
        feature_schema_version="conner-nq-f1",
        observations=(observation,),
        context_contract_ids={"ES": "CME:ES:2027-03-19"},
    )
    values.update(changes)
    return FeatureSnapshot(**values)


def test_late_revision_is_rejected_even_when_its_market_event_precedes_cutoff():
    observation = _observation(
        available_at=CUTOFF + timedelta(seconds=1),
        computed_at=CUTOFF + timedelta(seconds=2),
    )
    snapshot = _snapshot(observation, materialized_at=CUTOFF + timedelta(seconds=3))

    with pytest.raises(LeakageError, match="future knowledge"):
        validate_feature_snapshot(snapshot, _contract())


def test_historical_computation_after_cutoff_is_safe_when_all_inputs_were_available_by_cutoff():
    observation = _observation(computed_at=CUTOFF + timedelta(hours=1))
    snapshot = _snapshot(observation, materialized_at=CUTOFF + timedelta(hours=1, seconds=1))

    validate_feature_snapshot(snapshot, _contract())


def test_derived_feature_input_maxima_cannot_hide_future_data():
    observation = _observation(lineage={"max_input_available_at": CUTOFF + timedelta(microseconds=1)})

    with pytest.raises(LeakageError, match="future input lineage"):
        validate_feature_snapshot(_snapshot(observation), _contract())


@pytest.mark.parametrize(
    ("role", "domain"),
    [
        (FeatureTemporalRole.OUTCOME, FeatureSourceDomain.OUTCOME_ANALYSIS),
        (FeatureTemporalRole.FUTURE, FeatureSourceDomain.MARKET_STATE),
        (FeatureTemporalRole.FOLLOWER_EXECUTION, FeatureSourceDomain.FOLLOWER_EXECUTION),
    ],
)
def test_behavior_schema_rejects_outcome_future_and_follower_execution_inputs(role, domain):
    with pytest.raises(LeakageError):
        validate_feature_snapshot(_snapshot(_observation()), _contract(role=role, domain=domain))


def test_follower_execution_features_are_confined_to_the_execution_quality_task():
    contract = _contract(
        task=LearningTask.COPIER_EXECUTION_QUALITY,
        role=FeatureTemporalRole.FOLLOWER_EXECUTION,
        domain=FeatureSourceDomain.FOLLOWER_EXECUTION,
    )
    observation = _observation(
        temporal_role=FeatureTemporalRole.FOLLOWER_EXECUTION,
        source_domain=FeatureSourceDomain.FOLLOWER_EXECUTION,
        source_namespace=LearningSourceNamespace.COPIER_EXECUTION,
    )

    validate_feature_snapshot(_snapshot(observation), contract)


@pytest.mark.parametrize("source_kind", ["synthetic", "sandbox", "v1_equity"])
def test_learning_source_kind_is_a_closed_v2_enum(source_kind):
    contract = _contract(task=LearningTask.SETUP_OUTCOME_QUALITY)
    row = DatasetRow(
        task=contract.task,
        opportunity_id="opp-1",
        snapshot=_snapshot(_observation()),
        target_name="target",
        target_value="1",
        target_observed_at=CUTOFF + timedelta(minutes=30),
        source_kind=source_kind,
    )

    with pytest.raises(LeakageError, match=source_kind):
        validate_dataset_row(row, contract)


def test_v1_equity_lineage_cannot_masquerade_as_an_nq_dataset():
    contract = _contract(
        task=LearningTask.SETUP_OUTCOME_QUALITY,
        source_lineage={"asset_class": "equity", "ticker": "NQ", "source": "v1_equity"},
    )

    with pytest.raises(LeakageError, match="equity"):
        validate_feature_snapshot(_snapshot(_observation()), contract)

    with pytest.raises(ValueError, match="quarantined"):
        ModelArtifactMetadata(
            artifact_id="fake-nq",
            task=LearningTask.BEHAVIOR_IMITATION,
            feature_schema_version="conner-nq-f1",
            dataset_version="legacy",
            artifact_namespace="v2.futures.behavior_imitation",
            training_metrics={},
            promotion_status=PromotionStatus.CANDIDATE,
            source_lineage={"asset_class": "equity", "ticker": "NQ", "source": "v1_equity"},
            product_codes=frozenset({"NQ"}),
        )


def test_nq_prediction_target_and_es_context_are_declared_separately():
    contract = _contract(task=LearningTask.SETUP_OUTCOME_QUALITY)
    validate_feature_snapshot(_snapshot(_observation()), contract)

    bad_snapshot = _snapshot(_observation(), context_contract_ids={"ES": "CME:NQ:2027-03-19"})
    with pytest.raises(LeakageError, match="context contract"):
        validate_feature_snapshot(bad_snapshot, contract)

    artifact = ModelArtifactMetadata(
        artifact_id="behavior-1",
        task=LearningTask.BEHAVIOR_IMITATION,
        feature_schema_version="conner-nq-f1",
        dataset_version="conner-nq-d1",
        artifact_namespace="v2.futures.behavior_imitation",
        training_metrics={},
        promotion_status=PromotionStatus.SHADOW,
        source_lineage={"asset_class": "futures"},
        product_codes=frozenset({"NQ"}),
        context_product_codes=frozenset({"ES"}),
    )
    assert artifact.traded_product_codes == frozenset({"NQ"})
    assert artifact.context_product_codes == frozenset({"ES"})


def test_behavior_unlabeled_is_explicit_and_never_coerced_to_a_negative():
    contract = _contract()
    unlabeled = DatasetRow(
        task=contract.task,
        opportunity_id="opp-1",
        snapshot=_snapshot(_observation()),
        target_name="target",
        target_value=None,
        target_observed_at=None,
        label_state=LabelState.UNLABELED,
        seen_by_conner=SeenByConner.UNKNOWN,
    )
    validate_dataset_row(unlabeled, contract)

    artificial_negative = DatasetRow(
        **{**unlabeled.__dict__, "target_value": 0}
    )
    with pytest.raises(LeakageError, match="artificial target"):
        validate_dataset_row(artificial_negative, contract)


def test_explicit_behavior_negative_requires_exposure_evidence():
    contract = _contract()
    row = DatasetRow(
        task=contract.task,
        opportunity_id="opp-1",
        snapshot=_snapshot(_observation()),
        target_name="target",
        target_value=0,
        target_observed_at=CUTOFF + timedelta(minutes=1),
        label_state=LabelState.EXPLICIT_NEGATIVE,
        seen_by_conner=SeenByConner.YES,
    )

    with pytest.raises(LeakageError, match="evidence"):
        validate_dataset_row(row, contract)


def test_finalized_or_ambiguous_historical_data_cannot_train_behavior():
    contract = _contract()
    for source_kind in (
        LearningSourceKind.FINALIZED_HISTORICAL_MARKET,
        LearningSourceKind.HISTORICAL_REPLAY,
        LearningSourceKind.RETROSPECTIVE_TRADE_LEADUP,
    ):
        row = DatasetRow(
            task=contract.task,
            opportunity_id="opp-1",
            snapshot=_snapshot(_observation()),
            target_name="target",
            target_value=None,
            target_observed_at=None,
            source_kind=source_kind,
            label_state=LabelState.UNLABELED,
        )
        with pytest.raises(LeakageError, match="point-in-time"):
            validate_dataset_row(row, contract)

    blind = DatasetRow(
        task=contract.task,
        opportunity_id="opp-1",
        snapshot=_snapshot(_observation()),
        target_name="target",
        target_value=None,
        target_observed_at=None,
        source_kind=LearningSourceKind.BLIND_POINT_IN_TIME_REPLAY,
        label_state=LabelState.UNLABELED,
    )
    validate_dataset_row(blind, contract)
