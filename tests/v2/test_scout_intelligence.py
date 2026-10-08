from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from app.v2.domain.models import PositionSide
from app.v2.intelligence.datasets import (
    ActualConnerTradeLabel,
    BehaviorLabelState,
    ConsiderationStatus,
    ConvictionLabel,
    DatasetSourceKind,
    ExecutionQualityLabel,
    ObservationOrigin,
    OutcomeLabel,
    ScoutOpportunityCandidate,
    SeenByConner,
    TaskDatasetExample,
)
from app.v2.intelligence.evaluation import (
    EvaluationMode,
    EvaluationRunContract,
    ShadowCandidateEvaluation,
    ShadowMatchStatus,
    evaluate_trade_against_ranking,
)
from app.v2.intelligence.horizons import (
    HorizonDefinition,
    HorizonSet,
    LeadupSnapshotRecord,
    SnapshotSubjectKind,
    build_leadup_schedule,
)
from app.v2.intelligence.measurements import (
    CandidateLevelMeasurement,
    SwingCandidate,
    calculate_fibonacci_candidate,
    compare_swing_candidates,
    measure_candle_geometry,
    measure_candle_relationship,
)
from app.v2.intelligence.observation import (
    BarObservation,
    InstrumentMarketState,
    ObservationPurpose,
    ResearchContractPair,
    SynchronizedNqEsState,
)
from app.v2.intelligence.ranking import (
    RankedScoutOpportunity,
    RankingComponent,
    ScoutRankingSnapshot,
)
from app.v2.intelligence.replay import (
    MultiContractReplayRequest,
    ReplayAccessMode,
    prepare_replay_frames,
)
from app.v2.intelligence.specification import (
    CONNER_NQ_REQUIRED_CAPABILITIES,
    ObservationSpec,
    StrategyWindowCandidate,
)
from app.v2.intelligence.synchronization import (
    MarketSeriesRequirement,
    SynchronizationPolicy,
    synchronize_point_in_time,
)
from app.v2.learning.contracts import LearningTask
from app.v2.market_data import (
    AvailabilityMode,
    DataQuality,
    MarketDataCapability,
    MarketDataEvent,
    MarketEventKind,
)
from app.v2.reference import FuturesContractRegistry


CUTOFF = datetime(2026, 9, 18, 14, 0, tzinfo=timezone.utc)
NQ = "CME:NQ:2026-12-18"
ES = "CME:ES:2026-12-18"
LINEAGE = {"provider": "fixture", "dataset": "captured"}


def _bar(
    contract_id: str,
    event_id: str,
    *,
    minutes_before: int,
    duration_minutes: int = 1,
    timeframe: str = "1min",
    mode: AvailabilityMode = AvailabilityMode.POINT_IN_TIME,
    available_at: datetime | None = None,
) -> BarObservation:
    end = CUTOFF - timedelta(minutes=minutes_before)
    start = end - timedelta(minutes=duration_minutes)
    base = Decimal("20500") if ":NQ:" in contract_id else Decimal("6000")
    return BarObservation(
        event_id=event_id,
        contract_id=contract_id,
        timeframe=timeframe,
        window_start=start,
        window_end=end,
        open=base,
        high=base + Decimal("4"),
        low=base - Decimal("2"),
        close=base + Decimal("2"),
        volume=10,
        trade_count=5,
        source="fixture",
        source_timestamp=end,
        received_timestamp=available_at or end,
        session_end_date=date(2026, 9, 18),
        available_at=available_at,
        availability_mode=mode,
        window_start_ns=1,
        window_end_ns=2,
    )


def _event(
    contract_id: str,
    kind: MarketEventKind,
    event_at: datetime,
    *,
    available_at: datetime,
    resolution: str | None = None,
    mode: AvailabilityMode = AvailabilityMode.POINT_IN_TIME,
    event_id: str = "event",
) -> MarketDataEvent:
    payload = {} if resolution is None else {"resolution": resolution}
    return MarketDataEvent(
        contract_id=contract_id,
        kind=kind,
        source="fixture",
        source_timestamp=event_at,
        received_timestamp=available_at,
        sequence=1,
        payload=payload,
        available_timestamp=available_at,
        availability_mode=mode,
        source_timestamp_ns=int(event_at.timestamp() * 1_000_000_000),
        provider_event_id=event_id,
    )


def _ranking(origin: ObservationOrigin = ObservationOrigin.LIVE) -> ScoutRankingSnapshot:
    behavior = RankingComponent(
        "behavior_similarity",
        "v2.futures.behavior_imitation",
        {"probability": "0.8"},
        Decimal("0.8"),
        "behavior-1",
        Decimal("0.7"),
    )
    outcome = RankingComponent(
        "setup_outcome",
        "v2.futures.setup_outcome_quality",
        {"expected_r": "1.2"},
        Decimal("0.6"),
        "outcome-1",
        Decimal("0.5"),
    )
    opportunity = RankedScoutOpportunity(
        candidate_id="candidate-1",
        rank=1,
        display_score=Decimal("0.72"),
        behavior=behavior,
        outcome=outcome,
        conviction=None,
        model_confidence=Decimal("0.6"),
        data_quality_score=Decimal("1"),
        predicted_direction=PositionSide.LONG,
    )
    return ScoutRankingSnapshot(
        ranking_id="ranking-1",
        query_group_id="group-1",
        trade_date=date(2026, 9, 18),
        feature_cutoff_at=CUTOFF,
        generated_at=CUTOFF + timedelta(seconds=1),
        nq_contract_id=NQ,
        es_contract_id=ES,
        ranking_policy_version="display-v1",
        candidate_generator_version="candidate-v1",
        origin=origin,
        opportunities=(opportunity,),
        model_versions={"behavior": "behavior-1", "outcome": "outcome-1"},
        data_quality={"complete": True},
        observation_spec_version="1",
        feature_schema_version="1",
        source_lineage=LINEAGE,
    )


def test_observation_spec_is_provider_neutral_versioned_and_does_not_require_depth():
    window = StrategyWindowCandidate(
        "new-york-anchor-candidate",
        "1",
        "America/New_York",
        time(9, 30),
        timedelta(hours=3),
    )
    spec = ObservationSpec(
        spec_id="conner-observation-1",
        version="1",
        feature_schema_version="features-1",
        horizon_set_version="horizons-1",
        synchronization_policy_version="sync-1",
        timeframes=("1min", "5min", "1hour"),
        measurement_versions={"candle_geometry": ("raw-v1",), "smt_candidates": ("swing-v1", "swing-v2")},
        strategy_window_candidates=(window,),
    )
    assert spec.required_capabilities == CONNER_NQ_REQUIRED_CAPABILITIES
    assert MarketDataCapability.DEPTH not in spec.required_capabilities
    assert MarketDataCapability.MBO not in spec.required_capabilities
    assert "massive" not in repr(spec).lower()

    with pytest.raises(ValueError, match="not required"):
        ObservationSpec(
            **{**spec.__dict__, "required_capabilities": spec.required_capabilities | {MarketDataCapability.DEPTH}}
        )


def test_exact_nq_es_contract_pair_rejects_bare_and_wrong_product_ids():
    pair = ResearchContractPair("pair-1", date(2026, 9, 18), NQ, ES, "front-contract-v1", CUTOFF, LINEAGE)
    assert pair.nq_contract_id == NQ and pair.es_contract_id == ES
    with pytest.raises(ValueError, match="matching expiration-aware"):
        ResearchContractPair("bad", date(2026, 9, 18), ES, NQ, "v1", CUTOFF, LINEAGE)
    with pytest.raises(ValueError, match="matching expiration-aware"):
        ResearchContractPair("bad", date(2026, 9, 18), "NQ", ES, "v1", CUTOFF, LINEAGE)


def test_market_state_blocks_late_revisions_and_finalized_history_from_behavior():
    late = _bar(NQ, "late", minutes_before=1, available_at=CUTOFF + timedelta(seconds=1))
    with pytest.raises(ValueError, match="ineligible"):
        InstrumentMarketState("NQ", NQ, CUTOFF, CUTOFF, bars=(late,))

    finalized = _bar(
        NQ,
        "finalized",
        minutes_before=1,
        mode=AvailabilityMode.FINALIZED_HISTORICAL,
        available_at=CUTOFF + timedelta(days=1),
    )
    with pytest.raises(ValueError, match="ineligible"):
        InstrumentMarketState(
            "NQ",
            NQ,
            CUTOFF,
            CUTOFF,
            bars=(finalized,),
            purpose=ObservationPurpose.BEHAVIOR_TRAINING,
        )
    state = InstrumentMarketState(
        "NQ", NQ, CUTOFF, CUTOFF, bars=(finalized,), purpose=ObservationPurpose.OUTCOME_RESEARCH
    )
    assert state.bars == (finalized,)


def test_synchronization_is_backward_asof_and_keeps_each_timeframe_separate():
    requirements = (
        MarketSeriesRequirement("bar-1m", MarketEventKind.BAR, "1min"),
        MarketSeriesRequirement("bar-5m", MarketEventKind.BAR, "5min"),
        MarketSeriesRequirement("last-trade", MarketEventKind.TRADE),
    )
    policy = SynchronizationPolicy("sync-v1", requirements, timedelta(minutes=10), timedelta(seconds=5))
    events = []
    for contract in (NQ, ES):
        events.extend(
            [
                _event(
                    contract,
                    MarketEventKind.BAR,
                    CUTOFF - timedelta(minutes=1),
                    available_at=CUTOFF - timedelta(seconds=30),
                    resolution="1min",
                    event_id=f"{contract}-1m",
                ),
                _event(
                    contract,
                    MarketEventKind.BAR,
                    CUTOFF - timedelta(minutes=5),
                    available_at=CUTOFF - timedelta(seconds=30),
                    resolution="5min",
                    event_id=f"{contract}-5m",
                ),
                _event(
                    contract,
                    MarketEventKind.TRADE,
                    CUTOFF - timedelta(seconds=2),
                    available_at=CUTOFF - timedelta(seconds=1),
                    event_id=f"{contract}-trade",
                ),
                _event(
                    contract,
                    MarketEventKind.TRADE,
                    CUTOFF + timedelta(seconds=1),
                    available_at=CUTOFF + timedelta(seconds=2),
                    event_id=f"{contract}-future",
                ),
            ]
        )

    result = synchronize_point_in_time(events, nq_contract_id=NQ, es_contract_id=ES, cutoff_at=CUTOFF, policy=policy)
    assert result.is_complete
    assert set(result.events[NQ]) == {"bar-1m", "bar-5m", "last-trade"}
    assert result.events[NQ]["last-trade"].provider_event_id.endswith("trade")


def test_synchronized_state_requires_same_cutoff_and_within_leg_tolerance():
    nq = InstrumentMarketState("NQ", NQ, CUTOFF, CUTOFF - timedelta(seconds=1))
    es = InstrumentMarketState("ES", ES, CUTOFF, CUTOFF - timedelta(seconds=2))
    state = SynchronizedNqEsState(
        "state-1", date(2026, 9, 18), CUTOFF, nq, es, timedelta(seconds=2), DataQuality(), LINEAGE
    )
    assert state.nq.contract_id == NQ
    with pytest.raises(ValueError, match="tolerance"):
        SynchronizedNqEsState(
            "bad",
            date(2026, 9, 18),
            CUTOFF,
            nq,
            InstrumentMarketState("ES", ES, CUTOFF, CUTOFF - timedelta(seconds=10)),
            timedelta(seconds=2),
            DataQuality(),
            LINEAGE,
        )


def test_candle_outputs_are_raw_versioned_measurements_not_a_trade_rule():
    prior = _bar(NQ, "prior", minutes_before=2)
    current = _bar(NQ, "current", minutes_before=1)
    geometry = measure_candle_geometry(
        current,
        feature_cutoff_at=CUTOFF,
        purpose=ObservationPurpose.LIVE_SHADOW,
        definition_version="geometry-v1",
        source_lineage=LINEAGE,
        volatility_scale=Decimal("8"),
    )
    relationship = measure_candle_relationship(
        current,
        (prior,),
        feature_cutoff_at=CUTOFF,
        purpose=ObservationPurpose.LIVE_SHADOW,
        definition_version="relationship-v1",
        source_lineage=LINEAGE,
        volatility_scale=Decimal("8"),
    )
    assert geometry.body_size == Decimal("2")
    assert relationship.overlap_amount == Decimal("6")
    assert not hasattr(geometry, "is_rejection_block")
    assert geometry.definition_version == "geometry-v1" and geometry.source_lineage == LINEAGE


def test_relative_structure_and_fib_are_candidate_definitions_with_lineage():
    def swing(identifier, product, kind, price, seconds):
        timestamp = CUTOFF - timedelta(seconds=seconds)
        contract = NQ if product == "NQ" else ES
        return SwingCandidate(
            swing_id=identifier,
            contract_id=contract,
            product_code=product,
            kind=kind,
            price=Decimal(price),
            event_at=timestamp,
            confirmed_at=timestamp + timedelta(seconds=1),
            available_at=timestamp + timedelta(seconds=2),
            selector_version="local-extrema-v1",
            selector_parameters={"left": 2, "right": 2},
            source_event_ids=(f"bar-{identifier}",),
            source_lineage=LINEAGE,
        )

    relative = compare_swing_candidates(
        nq_prior=swing("nq-1", "NQ", "HIGH", "20500", 60),
        nq_current=swing("nq-2", "NQ", "HIGH", "20510", 30),
        es_prior=swing("es-1", "ES", "HIGH", "6000", 60),
        es_current=swing("es-2", "ES", "HIGH", "5999", 32),
        nq_scale=Decimal("20"),
        es_scale=Decimal("5"),
        definition_version="relative-high-v1",
        feature_cutoff_at=CUTOFF,
        source_lineage=LINEAGE,
    )
    assert relative.candidate_pattern == "NQ_HIGHER_HIGH_ES_NOT"
    assert relative.source_lineage == LINEAGE

    fib_args = dict(
        candidate_id="fib-1",
        contract_id=NQ,
        selector_version="passed-anchors-v1",
        selector_parameters={"ratio_family": "retracement"},
        anchor_a_id="a",
        anchor_a_price=Decimal("20400"),
        anchor_a_at=CUTOFF - timedelta(minutes=10),
        anchor_b_id="b",
        anchor_b_price=Decimal("20600"),
        anchor_b_at=CUTOFF - timedelta(minutes=2),
        ratio=Decimal("0.5"),
        current_price=Decimal("20510"),
        feature_cutoff_at=CUTOFF,
        source_lineage=LINEAGE,
    )
    fib = calculate_fibonacci_candidate(**fib_args)
    assert fib.level_price == Decimal("20500.0")
    with pytest.raises(ValueError, match="after cutoff"):
        calculate_fibonacci_candidate(**{**fib_args, "anchor_b_at": CUTOFF + timedelta(seconds=1)})


def test_candidate_levels_reject_post_cutoff_availability():
    values = dict(
        level_id="level-1",
        contract_id=NQ,
        source_algorithm="prior-high",
        source_version="1",
        price=Decimal("20500"),
        current_price=Decimal("20504"),
        signed_distance=Decimal("4"),
        first_observed_at=CUTOFF - timedelta(minutes=10),
        available_at=CUTOFF,
        feature_cutoff_at=CUTOFF,
        parameters={},
        source_lineage=LINEAGE,
    )
    assert CandidateLevelMeasurement(**values).signed_distance == Decimal("4")
    with pytest.raises(ValueError, match="unavailable"):
        CandidateLevelMeasurement(**{**values, "available_at": CUTOFF + timedelta(microseconds=1)})


def test_configurable_horizons_materialize_master_trade_leadup_without_a_candidate():
    horizons = HorizonSet(
        "leadup-v1",
        (HorizonDefinition("twenty-minutes", timedelta(minutes=20)), HorizonDefinition("decision", timedelta(0))),
    )
    slots = build_leadup_schedule("master-trade-1", CUTOFF, horizons)
    assert [slot.horizon_id for slot in slots] == ["twenty-minutes", "decision"]
    record = LeadupSnapshotRecord(
        snapshot_id="leadup-1",
        subject_id="master-trade-1",
        subject_kind=SnapshotSubjectKind.MASTER_TRADE,
        horizon_set_version="leadup-v1",
        horizon_id="twenty-minutes",
        decision_at=CUTOFF,
        target_cutoff_at=CUTOFF - timedelta(minutes=20),
        actual_cutoff_at=CUTOFF - timedelta(minutes=20, seconds=1),
        market_state_id="state-1",
        origin=ObservationOrigin.RETROSPECTIVE_TRADE_LEADUP,
        captured_at=CUTOFF + timedelta(days=1),
        source_lineage=LINEAGE,
    )
    assert record.candidate_id is None
    assert record.alignment_error == timedelta(seconds=1)
    assert not record.prospective_lead_time_eligible


def test_opportunities_default_unknown_unlabeled_and_only_evidenced_passes_are_negative():
    base = dict(
        candidate_id="candidate-1",
        trade_date=date(2026, 9, 18),
        nq_contract_id=NQ,
        es_contract_id=ES,
        candidate_generator="objective-candidate-enumerator",
        candidate_generator_version="1",
        generator_parameters={},
        first_observed_at=CUTOFF,
        latest_observed_at=CUTOFF,
        source_lineage=LINEAGE,
    )
    candidate = ScoutOpportunityCandidate(**base)
    assert candidate.seen_by_conner == SeenByConner.UNKNOWN
    assert candidate.behavior_label_state == BehaviorLabelState.UNLABELED

    with pytest.raises(ValueError, match="saw"):
        ScoutOpportunityCandidate(**base, behavior_label_state=BehaviorLabelState.EXPLICIT_PASS)
    explicit = ScoutOpportunityCandidate(
        **base,
        behavior_label_state=BehaviorLabelState.EXPLICIT_PASS,
        seen_by_conner=SeenByConner.CONFIRMED_SEEN,
        consideration_status=ConsiderationStatus.CONFIRMED_CONSIDERED,
        evidence={"source": "explicit_feedback"},
    )
    assert explicit.behavior_label_state == BehaviorLabelState.EXPLICIT_PASS


def test_task_datasets_reject_synthetic_and_keep_execution_facts_separate():
    common = dict(
        example_id="example-1",
        task=LearningTask.BEHAVIOR_IMITATION,
        candidate_id="candidate-1",
        snapshot_id="snapshot-1",
        query_group_id="group-1",
        feature_cutoff_at=CUTOFF,
        target_name=None,
        target_value=None,
        target_observed_at=None,
        source_lineage={"namespace": "v2.futures.conner_nq_v1"},
        behavior_label_state=BehaviorLabelState.UNLABELED,
    )
    with pytest.raises(ValueError, match="closed"):
        TaskDatasetExample(**common, source_kind="synthetic")
    with pytest.raises(ValueError, match="finalized"):
        TaskDatasetExample(**common, source_kind=DatasetSourceKind.FINALIZED_HISTORICAL_MARKET)

    execution = TaskDatasetExample(
        example_id="execution-1",
        task=LearningTask.COPIER_EXECUTION_QUALITY,
        candidate_id=None,
        snapshot_id=None,
        query_group_id=None,
        feature_cutoff_at=CUTOFF,
        target_name="latency_ms",
        target_value=Decimal("15"),
        target_observed_at=CUTOFF + timedelta(seconds=1),
        source_kind=DatasetSourceKind.FOLLOWER_EXECUTION,
        source_lineage={"namespace": "v2.futures.copier_execution"},
        follower_trade_id="follower-1",
    )
    assert execution.candidate_id is None
    with pytest.raises(ValueError, match="cannot label strategy"):
        TaskDatasetExample(**{**common, "source_kind": DatasetSourceKind.FOLLOWER_EXECUTION})


def test_actual_trade_outcome_conviction_and_execution_labels_remain_distinct():
    trade = ActualConnerTradeLabel(
        "label-1",
        "master-1",
        NQ,
        ES,
        CUTOFF,
        "FIRST_FILL",
        PositionSide.LONG,
        True,
        Decimal("20490"),
        2,
        Decimal("400"),
        {"orders": ["order-1"], "fills": ["fill-1"]},
    )
    assert trade.initial_risk_dollars == Decimal("400")
    with pytest.raises(ValueError, match="R-denominated"):
        OutcomeLabel(
            "outcome-1",
            "master-1",
            "candidate-1",
            CUTOFF,
            CUTOFF + timedelta(hours=1),
            Decimal("1"),
            None,
            None,
            Decimal("3600"),
            False,
            LINEAGE,
        )
    conviction = ConvictionLabel(
        "conviction-1", "master-1", CUTOFF, 2, Decimal("400"), None, LINEAGE
    )
    execution = ExecutionQualityLabel(
        "execution-1",
        "master-1",
        "follower-1",
        CUTOFF + timedelta(seconds=1),
        Decimal("25"),
        Decimal("1"),
        Decimal("1"),
        0,
        0,
        Decimal("0"),
        LINEAGE,
    )
    assert not conviction.can_control_follower_risk
    assert not execution.strategy_label_eligible


def test_ranking_keeps_task_outputs_separate_and_is_never_execution_eligible():
    ranking = _ranking()
    row = ranking.opportunities[0]
    assert row.behavior.raw_output != row.outcome.raw_output
    assert not row.display_score_training_target
    assert not ranking.execution_eligible and ranking.shadow_only
    with pytest.raises(ValueError, match="never execution"):
        ScoutRankingSnapshot(**{**ranking.__dict__, "execution_eligible": True})


def test_shadow_nonmatches_stay_unlabeled_and_only_prospective_runs_claim_lead_time():
    unmatched = ShadowCandidateEvaluation(
        "eval-1",
        "ranking-1",
        "candidate-1",
        1,
        CUTOFF,
        ShadowMatchStatus.NO_MATCH_OBSERVED_WINDOW,
        CUTOFF + timedelta(hours=1),
    )
    assert unmatched.behavior_label_state == BehaviorLabelState.UNLABELED
    with pytest.raises(ValueError, match="remain unlabeled"):
        ShadowCandidateEvaluation(
            **{**unmatched.__dict__, "behavior_label_state": BehaviorLabelState.EXPLICIT_PASS}
        )

    prospective = evaluate_trade_against_ranking(
        _ranking(),
        evaluation_id="trade-eval-1",
        master_trade_id="master-1",
        candidate_id="candidate-1",
        action_anchor_at=CUTOFF + timedelta(minutes=4),
        realized_r=Decimal("1"),
        mfe_r=Decimal("2"),
        mae_r=Decimal("-0.5"),
        matching_version="match-v1",
        data_quality_eligible=True,
        actual_direction=PositionSide.LONG,
    )
    assert prospective.lead_time == timedelta(minutes=4)
    assert prospective.direction_match is True
    retrospective = evaluate_trade_against_ranking(
        _ranking(ObservationOrigin.RETROSPECTIVE_TRADE_LEADUP),
        evaluation_id="trade-eval-2",
        master_trade_id="master-1",
        candidate_id="candidate-1",
        action_anchor_at=CUTOFF + timedelta(minutes=4),
        realized_r=None,
        mfe_r=None,
        mae_r=None,
        matching_version="match-v1",
        data_quality_eligible=True,
    )
    assert retrospective.lead_time is None


def test_replay_is_multi_contract_deterministic_and_finalized_history_is_not_behavior_data():
    request = MultiContractReplayRequest(
        NQ,
        ES,
        CUTOFF - timedelta(minutes=1),
        CUTOFF + timedelta(minutes=1),
        frozenset({MarketEventKind.TRADE}),
        ReplayAccessMode.BLIND_POINT_IN_TIME,
        "replay-v1",
    )
    nq = _event(
        NQ,
        MarketEventKind.TRADE,
        CUTOFF - timedelta(seconds=2),
        available_at=CUTOFF,
        event_id="nq",
    )
    es = _event(
        ES,
        MarketEventKind.TRADE,
        CUTOFF - timedelta(seconds=1),
        available_at=CUTOFF - timedelta(seconds=1),
        event_id="es",
    )
    frames = prepare_replay_frames(request, (nq, es))
    assert [frame.event.provider_event_id for frame in frames] == ["es", "nq"]
    assert all(frame.behavior_learning_eligible for frame in frames)

    finalized = _event(
        NQ,
        MarketEventKind.TRADE,
        CUTOFF - timedelta(seconds=2),
        available_at=CUTOFF + timedelta(days=1),
        mode=AvailabilityMode.FINALIZED_HISTORICAL,
        event_id="finalized",
    )
    with pytest.raises(ValueError, match="blind replay"):
        prepare_replay_frames(request, (finalized,))


def test_walk_forward_evaluation_is_strictly_out_of_sample():
    run = EvaluationRunContract(
        "run-1",
        "v2.futures.behavior_imitation",
        EvaluationMode.WALK_FORWARD,
        date(2026, 9, 1),
        date(2026, 9, 30),
        "dataset-1",
        "metrics-1",
        training_end=date(2026, 8, 31),
    )
    assert run.training_end < run.evaluation_start
    with pytest.raises(ValueError, match="must end before"):
        EvaluationRunContract(**{**run.__dict__, "training_end": run.evaluation_start})


def test_es_reference_is_context_only_and_not_an_nq_mnq_execution_mapping():
    registry = FuturesContractRegistry()
    expiry = date(2026, 12, 18)
    nq = registry.register_contract(contract_id=NQ, product_code="NQ", expiration=expiry)
    es = registry.register_contract(contract_id=ES, product_code="ES", expiration=expiry)
    assert es.specification.point_value == Decimal("50")
    with pytest.raises(ValueError, match="NQ/MNQ"):
        registry.map_same_expiry(nq.contract_id, es.contract_id)
