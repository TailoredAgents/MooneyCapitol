from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.db.models import Base


MONEY = Numeric(24, 10)


class V2FuturesContract(Base):
    __tablename__ = "v2_futures_contracts"
    contract_id = Column(String(96), primary_key=True)
    product_code = Column(String(16), nullable=False, index=True)
    exchange = Column(String(16), nullable=False)
    expiration = Column(Date, nullable=False, index=True)
    point_value = Column(MONEY, nullable=False)
    tick_size = Column(MONEY, nullable=False)
    tick_value = Column(MONEY, nullable=False)
    currency = Column(String(8), nullable=False, default="USD")
    first_trade_date = Column(Date, nullable=True)
    last_trade_date = Column(Date, nullable=True)
    provider_symbols = Column(JSONB, nullable=False, default=dict)


class V2ContractMapping(Base):
    __tablename__ = "v2_contract_mappings"
    id = Column(BigInteger, primary_key=True)
    source_contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    target_contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    expiration = Column(Date, nullable=False)
    __table_args__ = (UniqueConstraint("source_contract_id", "target_contract_id", name="uq_v2_contract_mapping"),)


class V2BrokerConnection(Base):
    __tablename__ = "v2_broker_connections"
    connection_id = Column(String(96), primary_key=True)
    broker = Column(String(32), nullable=False)
    environment = Column(String(32), nullable=False)
    credential_ref = Column(String(256), nullable=True)
    enabled = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class V2BrokerAccount(Base):
    __tablename__ = "v2_broker_accounts"
    account_id = Column(String(96), primary_key=True)
    connection_id = Column(String(96), ForeignKey("v2_broker_connections.connection_id"), nullable=False)
    broker_account_ref = Column(String(256), nullable=False)
    display_name = Column(String(128), nullable=False, default="")
    enabled = Column(Boolean, nullable=False, default=False)


class V2RiskProfile(Base):
    __tablename__ = "v2_risk_profiles"
    profile_id = Column(String(96), primary_key=True)
    account_id = Column(String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True)
    sizing_mode = Column(String(40), nullable=False)
    sizing_value = Column(MONEY, nullable=False)
    max_risk_per_trade = Column(MONEY, nullable=False)
    max_contracts = Column(Integer, nullable=False)
    max_concurrent_open_risk = Column(MONEY, nullable=False)
    daily_realized_loss_ceiling = Column(MONEY, nullable=False)
    daily_total_loss_ceiling = Column(MONEY, nullable=False)
    margin_headroom_fraction = Column(MONEY, nullable=False)
    limits_json = Column(JSONB, nullable=False, default=dict)
    global_kill_switch = Column(Boolean, nullable=False, default=True)
    account_kill_switch = Column(Boolean, nullable=False, default=True)


class V2SessionSchedule(Base):
    __tablename__ = "v2_session_schedules"
    id = Column(BigInteger, primary_key=True)
    contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    trade_date = Column(Date, nullable=False)
    session_start = Column(DateTime(timezone=True), nullable=False)
    session_end = Column(DateTime(timezone=True), nullable=False)
    maintenance_start = Column(DateTime(timezone=True), nullable=True)
    maintenance_end = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(32), nullable=False)
    source = Column(String(64), nullable=False)
    __table_args__ = (UniqueConstraint("contract_id", "trade_date", name="uq_v2_schedule_contract_date"),)


class V2OpportunitySnapshot(Base):
    __tablename__ = "v2_opportunity_snapshots"
    opportunity_id = Column(String(96), primary_key=True)
    contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    feature_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    trade_date = Column(Date, nullable=False)
    rule_version = Column(String(64), nullable=False)
    feature_schema_version = Column(String(64), nullable=False)
    market_data_lineage = Column(JSONB, nullable=False)


class V2TradePlanVersion(Base):
    __tablename__ = "v2_trade_plan_versions"
    plan_id = Column(String(96), primary_key=True)
    trade_id = Column(String(96), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    direction = Column(String(8), nullable=False)
    planned_entry = Column(MONEY, nullable=True)
    original_stop = Column(MONEY, nullable=True)
    planned_targets = Column(JSONB, nullable=False, default=list)
    planned_quantity = Column(Integer, nullable=True)
    planned_risk_dollars = Column(MONEY, nullable=True)
    source = Column(String(32), nullable=False)
    __table_args__ = (UniqueConstraint("trade_id", "version", name="uq_v2_trade_plan_version"),)


class V2MasterOrder(Base):
    __tablename__ = "v2_master_orders"
    master_order_id = Column(String(96), primary_key=True)
    account_id = Column(String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=False)
    contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    plan_id = Column(String(96), ForeignKey("v2_trade_plan_versions.plan_id"), nullable=True)
    parent_order_id = Column(String(96), nullable=True)
    side = Column(String(8), nullable=False)
    quantity = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False)
    latest_revision = Column(Integer, nullable=False)


class V2OrderEvent(Base):
    __tablename__ = "v2_order_events"
    event_id = Column(String(96), primary_key=True)
    master_order_id = Column(String(96), ForeignKey("v2_master_orders.master_order_id"), nullable=True)
    follower_order_id = Column(String(96), nullable=True)
    broker = Column(String(32), nullable=False)
    account_id = Column(String(96), nullable=False)
    broker_order_id = Column(String(128), nullable=False)
    contract_id = Column(String(96), nullable=False)
    revision = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    quantity = Column(Integer, nullable=False)
    filled_quantity = Column(Integer, nullable=False)
    average_fill_price = Column(MONEY, nullable=True)
    event_at = Column(DateTime(timezone=True), nullable=False)
    received_at = Column(DateTime(timezone=True), nullable=False)
    raw_reference = Column(String(256), nullable=True)
    __table_args__ = (UniqueConstraint("broker", "account_id", "broker_order_id", "revision", name="uq_v2_order_event_revision"),)


class V2MasterExecution(Base):
    __tablename__ = "v2_master_executions"
    execution_id = Column(String(96), primary_key=True)
    master_order_id = Column(String(96), ForeignKey("v2_master_orders.master_order_id"), nullable=False)
    account_id = Column(String(96), nullable=False)
    contract_id = Column(String(96), nullable=False)
    side = Column(String(8), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(MONEY, nullable=False)
    fee = Column(MONEY, nullable=False)
    executed_at = Column(DateTime(timezone=True), nullable=False)
    received_at = Column(DateTime(timezone=True), nullable=False)


class V2TradeManagementEvent(Base):
    __tablename__ = "v2_trade_management_events"
    management_event_id = Column(String(96), primary_key=True)
    trade_id = Column(String(96), nullable=False, index=True)
    event_type = Column(String(32), nullable=False)
    occurred_at = Column(DateTime(timezone=True), nullable=False)
    quantity = Column(Integer, nullable=True)
    previous_price = Column(MONEY, nullable=True)
    new_price = Column(MONEY, nullable=True)
    reason = Column(Text, nullable=True)


class V2MasterTrade(Base):
    __tablename__ = "v2_master_trades"
    trade_id = Column(String(96), primary_key=True)
    account_id = Column(String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=False)
    contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    direction = Column(String(8), nullable=False)
    original_plan_id = Column(String(96), ForeignKey("v2_trade_plan_versions.plan_id"), nullable=True)
    original_stop = Column(MONEY, nullable=True)
    original_risk_dollars = Column(MONEY, nullable=True)
    entry_vwap = Column(MONEY, nullable=True)
    exit_vwap = Column(MONEY, nullable=True)
    initial_quantity = Column(Integer, nullable=False)
    maximum_quantity = Column(Integer, nullable=False)
    opened_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    realized_gross_pnl = Column(MONEY, nullable=True)
    realized_net_pnl = Column(MONEY, nullable=True)
    realized_r = Column(MONEY, nullable=True)
    mfe_r = Column(MONEY, nullable=True)
    mae_r = Column(MONEY, nullable=True)


class V2CopyIntent(Base):
    __tablename__ = "v2_copy_intents"
    copy_intent_id = Column(String(96), primary_key=True)
    master_order_id = Column(String(96), ForeignKey("v2_master_orders.master_order_id"), nullable=False)
    follower_account_id = Column(String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=False)
    source_contract_id = Column(String(96), nullable=False)
    target_contract_id = Column(String(96), nullable=False)
    intended_quantity = Column(Integer, nullable=False)
    planned_risk_dollars = Column(MONEY, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)


class V2FollowerOrder(Base):
    __tablename__ = "v2_follower_orders"
    follower_order_id = Column(String(96), primary_key=True)
    copy_intent_id = Column(String(96), ForeignKey("v2_copy_intents.copy_intent_id"), nullable=False)
    account_id = Column(String(96), nullable=False)
    contract_id = Column(String(96), nullable=False)
    side = Column(String(8), nullable=False)
    quantity = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False)
    latest_revision = Column(Integer, nullable=False)


class V2OrderLink(Base):
    __tablename__ = "v2_order_links"
    link_id = Column(String(96), primary_key=True)
    parent_order_id = Column(String(96), nullable=False)
    child_order_id = Column(String(96), nullable=False)
    link_type = Column(String(32), nullable=False)
    oco_group_id = Column(String(96), nullable=True)


class V2FollowerExecution(Base):
    __tablename__ = "v2_follower_executions"
    execution_id = Column(String(96), primary_key=True)
    follower_order_id = Column(String(96), ForeignKey("v2_follower_orders.follower_order_id"), nullable=False)
    copy_intent_id = Column(String(96), nullable=False)
    account_id = Column(String(96), nullable=False)
    contract_id = Column(String(96), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(MONEY, nullable=False)
    fee = Column(MONEY, nullable=False)
    executed_at = Column(DateTime(timezone=True), nullable=False)
    received_at = Column(DateTime(timezone=True), nullable=False)


class V2FollowerTrade(Base):
    __tablename__ = "v2_follower_trades"
    follower_trade_id = Column(String(96), primary_key=True)
    master_trade_id = Column(String(96), ForeignKey("v2_master_trades.trade_id"), nullable=False)
    account_id = Column(String(96), nullable=False)
    contract_id = Column(String(96), nullable=False)
    entry_vwap = Column(MONEY, nullable=True)
    exit_vwap = Column(MONEY, nullable=True)
    realized_net_pnl = Column(MONEY, nullable=True)
    realized_r = Column(MONEY, nullable=True)
    execution_quality = Column(JSONB, nullable=False, default=dict)


class V2PositionSnapshot(Base):
    __tablename__ = "v2_position_snapshots"
    id = Column(BigInteger, primary_key=True)
    account_id = Column(String(96), nullable=False)
    contract_id = Column(String(96), nullable=False)
    captured_at = Column(DateTime(timezone=True), nullable=False)
    quantity = Column(Integer, nullable=False)
    average_price = Column(MONEY, nullable=True)
    mark_price = Column(MONEY, nullable=True)
    unrealized_pnl = Column(MONEY, nullable=False)


class V2AccountRiskSnapshot(Base):
    __tablename__ = "v2_account_risk_snapshots"
    id = Column(BigInteger, primary_key=True)
    account_id = Column(String(96), nullable=False)
    captured_at = Column(DateTime(timezone=True), nullable=False)
    equity = Column(MONEY, nullable=False)
    available_margin = Column(MONEY, nullable=False)
    open_risk = Column(MONEY, nullable=False)
    realized_pnl_trade_date = Column(MONEY, nullable=False)
    total_pnl_trade_date = Column(MONEY, nullable=False)
    drawdown_trade_date = Column(MONEY, nullable=False)
    global_kill_switch = Column(Boolean, nullable=False)
    account_kill_switch = Column(Boolean, nullable=False)


class V2ReconciliationIncident(Base):
    __tablename__ = "v2_reconciliation_incidents"
    incident_id = Column(String(96), primary_key=True)
    account_id = Column(String(96), nullable=False)
    detected_at = Column(DateTime(timezone=True), nullable=False)
    incident_type = Column(String(64), nullable=False)
    severity = Column(String(16), nullable=False)
    expected = Column(JSONB, nullable=False)
    actual = Column(JSONB, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)


class V2ModelArtifact(Base):
    __tablename__ = "v2_model_artifacts"
    artifact_id = Column(String(96), primary_key=True)
    task = Column(String(64), nullable=False)
    feature_schema_version = Column(String(64), nullable=False)
    dataset_version = Column(String(64), nullable=False)
    artifact_namespace = Column(String(128), nullable=False)
    training_metrics = Column(JSONB, nullable=False)
    promotion_status = Column(String(32), nullable=False)
    source_lineage = Column(JSONB, nullable=False)
    asset_class = Column(String(16), nullable=False)
    product_codes = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class V2ExecutionLease(Base):
    __tablename__ = "v2_execution_leases"
    lease_name = Column(String(96), primary_key=True)
    owner_id = Column(String(96), nullable=False)
    fence = Column(BigInteger, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class V2ObservationSpec(Base):
    __tablename__ = "v2_observation_specs"
    observation_spec_id = Column(String(96), primary_key=True)
    strategy_namespace = Column(String(64), nullable=False)
    version = Column(String(64), nullable=False)
    feature_schema_version = Column(String(64), nullable=False)
    horizon_set_version = Column(String(64), nullable=False)
    synchronization_policy_version = Column(String(64), nullable=False)
    timeframes_json = Column(JSONB, nullable=False, default=list)
    measurement_versions = Column(JSONB, nullable=False)
    required_capabilities = Column(JSONB, nullable=False, default=list)
    strategy_window_candidates = Column(JSONB, nullable=False, default=list)
    synchronization_tolerance_ms = Column(Integer, nullable=False)
    parameters_json = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("strategy_namespace", "version", name="uq_v2_observation_spec_version"),
        CheckConstraint("synchronization_tolerance_ms >= 0", name="ck_v2_observation_spec_sync"),
        CheckConstraint(
            "jsonb_typeof(timeframes_json) = 'array' AND timeframes_json <> '[]'::jsonb "
            "AND jsonb_typeof(required_capabilities) = 'array' "
            "AND required_capabilities <> '[]'::jsonb "
            "AND jsonb_typeof(measurement_versions) = 'object' "
            "AND measurement_versions <> '{}'::jsonb "
            "AND jsonb_typeof(strategy_window_candidates) = 'array'",
            name="ck_v2_observation_spec_shapes",
        ),
    )


class V2ObservationHorizon(Base):
    __tablename__ = "v2_observation_horizons"
    horizon_id = Column(String(96), primary_key=True)
    observation_spec_id = Column(String(96), ForeignKey("v2_observation_specs.observation_spec_id"), nullable=False)
    name = Column(String(64), nullable=False)
    offset_seconds = Column(Integer, nullable=False)
    ordinal = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("observation_spec_id", "name", name="uq_v2_observation_horizon_name"),
        UniqueConstraint("observation_spec_id", "offset_seconds", name="uq_v2_observation_horizon_offset"),
        CheckConstraint("offset_seconds >= 0", name="ck_v2_observation_horizon_offset"),
        CheckConstraint("ordinal >= 0", name="ck_v2_observation_horizon_ordinal"),
    )


class V2SynchronizedMarketState(Base):
    __tablename__ = "v2_synchronized_market_states"
    market_state_id = Column(String(96), primary_key=True)
    observation_spec_id = Column(String(96), ForeignKey("v2_observation_specs.observation_spec_id"), nullable=False)
    nq_contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    es_contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    trade_date = Column(Date, nullable=False)
    schema_name = Column(String(64), nullable=False)
    schema_version = Column(String(64), nullable=False)
    feature_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    nq_source_max_at = Column(DateTime(timezone=True), nullable=False)
    es_source_max_at = Column(DateTime(timezone=True), nullable=False)
    nq_available_max_at = Column(DateTime(timezone=True), nullable=False)
    es_available_max_at = Column(DateTime(timezone=True), nullable=False)
    availability_mode = Column(String(32), nullable=False)
    captured_at = Column(DateTime(timezone=True), nullable=False)
    observation_purpose = Column(String(32), nullable=False)
    provider = Column(String(64), nullable=False)
    provider_dataset = Column(String(128), nullable=True)
    replay_reference = Column(String(256), nullable=True)
    synchronization_delta_ms = Column(Integer, nullable=False)
    nq_state = Column(JSONB, nullable=False)
    es_state = Column(JSONB, nullable=False)
    source_lineage = Column(JSONB, nullable=False)
    data_quality = Column(JSONB, nullable=False, default=dict)
    is_complete = Column(Boolean, nullable=False, default=False)
    __table_args__ = (
        UniqueConstraint(
            "observation_spec_id",
            "nq_contract_id",
            "es_contract_id",
            "feature_cutoff_at",
            name="uq_v2_market_state_cutoff",
        ),
        CheckConstraint("nq_contract_id <> es_contract_id", name="ck_v2_market_state_contracts"),
        CheckConstraint("synchronization_delta_ms >= 0", name="ck_v2_market_state_sync"),
        CheckConstraint(
            "nq_source_max_at <= feature_cutoff_at AND es_source_max_at <= feature_cutoff_at",
            name="ck_v2_market_state_source_cutoff",
        ),
        CheckConstraint(
            "nq_source_max_at <= nq_available_max_at "
            "AND es_source_max_at <= es_available_max_at",
            name="ck_v2_market_state_temporal_order",
        ),
        CheckConstraint(
            "availability_mode IN ('POINT_IN_TIME', 'FINALIZED_HISTORICAL')",
            name="ck_v2_market_state_availability_mode",
        ),
        CheckConstraint(
            "(availability_mode = 'POINT_IN_TIME' "
            "AND nq_available_max_at <= feature_cutoff_at "
            "AND es_available_max_at <= feature_cutoff_at) OR "
            "(availability_mode = 'FINALIZED_HISTORICAL' "
            "AND observation_purpose IN ('OUTCOME_RESEARCH', 'HISTORICAL_REPLAY'))",
            name="ck_v2_market_state_available_cutoff",
        ),
        CheckConstraint(
            "feature_cutoff_at <= captured_at "
            "AND nq_available_max_at <= captured_at "
            "AND es_available_max_at <= captured_at",
            name="ck_v2_market_state_capture",
        ),
        CheckConstraint(
            "observation_purpose IN ('LIVE_SHADOW', 'BEHAVIOR_TRAINING', "
            "'OUTCOME_RESEARCH', 'HISTORICAL_REPLAY')",
            name="ck_v2_market_state_purpose",
        ),
        CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_market_state_lineage"),
        Index("ix_v2_market_states_trade_cutoff", "trade_date", "feature_cutoff_at"),
    )


class V2CandidateFeatureDefinition(Base):
    __tablename__ = "v2_candidate_feature_definitions"
    feature_definition_id = Column(String(96), primary_key=True)
    strategy_namespace = Column(String(64), nullable=False)
    family = Column(String(64), nullable=False)
    name = Column(String(128), nullable=False)
    version = Column(String(64), nullable=False)
    value_type = Column(String(32), nullable=False)
    source_algorithm = Column(String(128), nullable=False)
    parameters_json = Column(JSONB, nullable=False)
    description = Column(Text, nullable=False)
    research_status = Column(String(32), nullable=False, default="CANDIDATE")
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "strategy_namespace", "family", "name", "version", name="uq_v2_feature_definition_version"
        ),
        CheckConstraint(
            "research_status IN ('CANDIDATE', 'EMPIRICALLY_SUPPORTED', 'RETIRED')",
            name="ck_v2_feature_definition_research_status",
        ),
    )


class V2CandidateFeatureValue(Base):
    __tablename__ = "v2_candidate_feature_values"
    measurement_id = Column(String(96), primary_key=True)
    market_state_id = Column(
        String(96), ForeignKey("v2_synchronized_market_states.market_state_id"), nullable=False
    )
    feature_definition_id = Column(
        String(96), ForeignKey("v2_candidate_feature_definitions.feature_definition_id"), nullable=False
    )
    candidate_key = Column(String(96), nullable=False, default="default")
    instrument_role = Column(String(16), nullable=False)
    feature_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    source_window_start_at = Column(DateTime(timezone=True), nullable=True)
    source_window_end_at = Column(DateTime(timezone=True), nullable=True)
    max_input_event_at = Column(DateTime(timezone=True), nullable=False)
    max_input_available_at = Column(DateTime(timezone=True), nullable=True)
    definition_available_at = Column(DateTime(timezone=True), nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)
    availability_mode = Column(String(32), nullable=False)
    value_json = Column(JSONB, nullable=True)
    units = Column(String(32), nullable=True)
    parameters_json = Column(JSONB, nullable=False, default=dict)
    source_event_ids = Column(JSONB, nullable=False, default=list)
    is_missing = Column(Boolean, nullable=False, default=False)
    missing_reason = Column(Text, nullable=True)
    source_lineage = Column(JSONB, nullable=False)
    data_quality = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint(
            "market_state_id",
            "feature_definition_id",
            "candidate_key",
            "instrument_role",
            name="uq_v2_candidate_feature_value",
        ),
        CheckConstraint("instrument_role IN ('NQ', 'ES', 'CROSS')", name="ck_v2_feature_value_role"),
        CheckConstraint(
            "max_input_event_at <= feature_cutoff_at "
            "AND definition_available_at <= feature_cutoff_at "
            "AND computed_at >= max_input_event_at "
            "AND computed_at >= definition_available_at "
            "AND (max_input_available_at IS NULL OR computed_at >= max_input_available_at)",
            name="ck_v2_feature_value_cutoff",
        ),
        CheckConstraint(
            "availability_mode IN ('POINT_IN_TIME', 'FINALIZED_HISTORICAL')",
            name="ck_v2_feature_value_availability_mode",
        ),
        CheckConstraint(
            "availability_mode <> 'POINT_IN_TIME' OR "
            "(max_input_available_at IS NOT NULL AND max_input_available_at <= feature_cutoff_at)",
            name="ck_v2_feature_value_available_at",
        ),
        CheckConstraint(
            "(source_window_start_at IS NULL AND source_window_end_at IS NULL) OR "
            "(source_window_start_at IS NOT NULL AND source_window_end_at IS NOT NULL "
            "AND source_window_start_at <= source_window_end_at "
            "AND source_window_end_at <= feature_cutoff_at)",
            name="ck_v2_feature_value_window",
        ),
        CheckConstraint(
            "(is_missing AND value_json IS NULL AND missing_reason IS NOT NULL) OR "
            "(NOT is_missing AND value_json IS NOT NULL)",
            name="ck_v2_feature_value_missing",
        ),
        CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_feature_value_lineage"),
        Index(
            "ix_v2_feature_values_state_definition", "market_state_id", "feature_definition_id"
        ),
    )


class V2ScoutOpportunity(Base):
    __tablename__ = "v2_scout_opportunities"
    opportunity_id = Column(
        String(96), ForeignKey("v2_opportunity_snapshots.opportunity_id"), primary_key=True
    )
    observation_spec_id = Column(String(96), ForeignKey("v2_observation_specs.observation_spec_id"), nullable=False)
    market_state_id = Column(
        String(96), ForeignKey("v2_synchronized_market_states.market_state_id"), nullable=False
    )
    nq_contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    es_contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    strategy_namespace = Column(String(64), nullable=False)
    candidate_series_id = Column(String(96), nullable=False)
    candidate_key = Column(String(96), nullable=False)
    candidate_generator = Column(String(128), nullable=False)
    candidate_generator_version = Column(String(64), nullable=False)
    generator_parameters = Column(JSONB, nullable=False)
    first_observed_at = Column(DateTime(timezone=True), nullable=False)
    latest_observed_at = Column(DateTime(timezone=True), nullable=False)
    candidate_direction = Column(String(8), nullable=True)
    seen_by_conner = Column(String(24), nullable=False, default="UNKNOWN")
    consideration_status = Column(String(32), nullable=False, default="UNKNOWN")
    behavior_label_state = Column(String(32), nullable=False, default="UNLABELED")
    linked_master_trade_id = Column(String(96), ForeignKey("v2_master_trades.trade_id"), nullable=True)
    evidence_json = Column(JSONB, nullable=False, default=dict)
    source_lineage = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "strategy_namespace",
            "market_state_id",
            "candidate_key",
            "candidate_generator_version",
            name="uq_v2_scout_opportunity_candidate",
        ),
        CheckConstraint("nq_contract_id <> es_contract_id", name="ck_v2_scout_opportunity_contracts"),
        CheckConstraint(
            "first_observed_at <= latest_observed_at", name="ck_v2_scout_opportunity_observed_at"
        ),
        CheckConstraint(
            "candidate_direction IS NULL OR candidate_direction IN ('LONG', 'SHORT')",
            name="ck_v2_scout_opportunity_direction",
        ),
        CheckConstraint(
            "seen_by_conner IN ('UNKNOWN', 'CONFIRMED_SEEN', 'CONFIRMED_NOT_SEEN')",
            name="ck_v2_scout_opportunity_seen",
        ),
        CheckConstraint(
            "consideration_status IN ('UNKNOWN', 'PRESENTED', 'CONFIRMED_CONSIDERED')",
            name="ck_v2_scout_opportunity_consideration",
        ),
        CheckConstraint(
            "behavior_label_state IN ('UNLABELED', 'POSITIVE_TRADE', 'EXPLICIT_PASS')",
            name="ck_v2_scout_opportunity_behavior_state",
        ),
        CheckConstraint(
            "(behavior_label_state = 'UNLABELED' AND linked_master_trade_id IS NULL) OR "
            "(behavior_label_state = 'POSITIVE_TRADE' AND linked_master_trade_id IS NOT NULL "
            "AND seen_by_conner = 'CONFIRMED_SEEN' "
            "AND consideration_status = 'CONFIRMED_CONSIDERED' "
            "AND evidence_json <> '{}'::jsonb) OR "
            "(behavior_label_state = 'EXPLICIT_PASS' AND linked_master_trade_id IS NULL "
            "AND seen_by_conner = 'CONFIRMED_SEEN' "
            "AND consideration_status = 'CONFIRMED_CONSIDERED' "
            "AND evidence_json <> '{}'::jsonb)",
            name="ck_v2_scout_opportunity_behavior_evidence",
        ),
        CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_scout_opportunity_lineage"),
        Index(
            "ix_v2_scout_opportunities_namespace_series",
            "strategy_namespace",
            "candidate_series_id",
        ),
    )


class V2OpportunityBehaviorLabel(Base):
    __tablename__ = "v2_opportunity_behavior_labels"
    label_id = Column(String(96), primary_key=True)
    opportunity_id = Column(String(96), ForeignKey("v2_scout_opportunities.opportunity_id"), nullable=False)
    revision = Column(Integer, nullable=False)
    label_kind = Column(String(32), nullable=False)
    master_trade_id = Column(String(96), ForeignKey("v2_master_trades.trade_id"), nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    evidence_source = Column(String(64), nullable=False)
    evidence_json = Column(JSONB, nullable=False)
    supersedes_label_id = Column(
        String(96), ForeignKey("v2_opportunity_behavior_labels.label_id"), nullable=True
    )
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("opportunity_id", "revision", name="uq_v2_opportunity_behavior_revision"),
        CheckConstraint("revision >= 1", name="ck_v2_opportunity_behavior_revision"),
        CheckConstraint(
            "label_kind IN ('POSITIVE_TRADE', 'EXPLICIT_PASS')",
            name="ck_v2_opportunity_behavior_kind",
        ),
        CheckConstraint(
            "(label_kind = 'POSITIVE_TRADE' AND master_trade_id IS NOT NULL "
            "AND evidence_json <> '{}'::jsonb) OR "
            "(label_kind = 'EXPLICIT_PASS' AND master_trade_id IS NULL "
            "AND evidence_json <> '{}'::jsonb)",
            name="ck_v2_opportunity_behavior_evidence",
        ),
        CheckConstraint(
            "supersedes_label_id IS NULL OR supersedes_label_id <> label_id",
            name="ck_v2_opportunity_behavior_supersedes",
        ),
    )


class V2TradeLeadupSnapshot(Base):
    __tablename__ = "v2_trade_leadup_snapshots"
    trade_leadup_id = Column(String(96), primary_key=True)
    subject_id = Column(String(96), nullable=False)
    subject_kind = Column(String(32), nullable=False)
    master_trade_id = Column(String(96), ForeignKey("v2_master_trades.trade_id"), nullable=True)
    horizon_id = Column(String(96), ForeignKey("v2_observation_horizons.horizon_id"), nullable=False)
    horizon_set_version = Column(String(64), nullable=False)
    market_state_id = Column(
        String(96), ForeignKey("v2_synchronized_market_states.market_state_id"), nullable=False
    )
    opportunity_id = Column(String(96), ForeignKey("v2_scout_opportunities.opportunity_id"), nullable=True)
    decision_at = Column(DateTime(timezone=True), nullable=False)
    target_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    actual_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    origin = Column(String(32), nullable=False)
    captured_at = Column(DateTime(timezone=True), nullable=False)
    lead_time_ms = Column(BigInteger, nullable=True)
    alignment_error_ms = Column(Integer, nullable=False)
    source_lineage = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "subject_kind",
            "subject_id",
            "horizon_set_version",
            "horizon_id",
            name="uq_v2_trade_leadup_horizon",
        ),
        CheckConstraint(
            "subject_kind IN ('MASTER_TRADE', 'SCOUT_CANDIDATE')",
            name="ck_v2_trade_leadup_subject_kind",
        ),
        CheckConstraint(
            "(subject_kind = 'MASTER_TRADE' AND master_trade_id = subject_id "
            "AND opportunity_id IS NULL) OR "
            "(subject_kind = 'SCOUT_CANDIDATE' AND opportunity_id = subject_id "
            "AND master_trade_id IS NULL)",
            name="ck_v2_trade_leadup_subject",
        ),
        CheckConstraint(
            "actual_cutoff_at <= target_cutoff_at AND target_cutoff_at <= decision_at",
            name="ck_v2_trade_leadup_cutoff",
        ),
        CheckConstraint(
            "origin IN ('LIVE', 'BLIND_REPLAY', 'RETROSPECTIVE_TRADE_LEADUP')",
            name="ck_v2_trade_leadup_origin",
        ),
        CheckConstraint(
            "(origin = 'RETROSPECTIVE_TRADE_LEADUP' AND lead_time_ms IS NULL) OR "
            "(origin IN ('LIVE', 'BLIND_REPLAY') AND lead_time_ms IS NOT NULL AND lead_time_ms >= 0)",
            name="ck_v2_trade_leadup_lead_time",
        ),
        CheckConstraint("alignment_error_ms >= 0", name="ck_v2_trade_leadup_alignment"),
        CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_trade_leadup_lineage"),
        Index("ix_v2_trade_leadup_trade_cutoff", "subject_kind", "subject_id", "actual_cutoff_at"),
    )


class V2LearningExample(Base):
    __tablename__ = "v2_learning_examples"
    example_id = Column(String(96), primary_key=True)
    strategy_namespace = Column(String(64), nullable=True)
    task = Column(String(64), nullable=False)
    market_state_id = Column(
        String(96), ForeignKey("v2_synchronized_market_states.market_state_id"), nullable=True
    )
    opportunity_id = Column(String(96), ForeignKey("v2_scout_opportunities.opportunity_id"), nullable=True)
    behavior_label_id = Column(
        String(96), ForeignKey("v2_opportunity_behavior_labels.label_id"), nullable=True
    )
    master_trade_id = Column(String(96), ForeignKey("v2_master_trades.trade_id"), nullable=True)
    follower_trade_id = Column(
        String(96), ForeignKey("v2_follower_trades.follower_trade_id"), nullable=True
    )
    trade_date = Column(Date, nullable=False)
    feature_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    label_state = Column(String(32), nullable=False)
    seen_by_conner = Column(String(24), nullable=False, default="UNKNOWN")
    consideration_status = Column(String(32), nullable=False, default="UNKNOWN")
    target_name = Column(String(128), nullable=True)
    target_numeric = Column(MONEY, nullable=True)
    target_boolean = Column(Boolean, nullable=True)
    target_json = Column(JSONB, nullable=True)
    target_observed_at = Column(DateTime(timezone=True), nullable=True)
    ranking_group_id = Column(String(96), nullable=True)
    relevance_grade = Column(MONEY, nullable=True)
    sampling_policy_version = Column(String(64), nullable=False)
    feature_schema_version = Column(String(64), nullable=False)
    dataset_version = Column(String(64), nullable=False)
    source_kind = Column(String(32), nullable=False)
    source_lineage = Column(JSONB, nullable=False)
    eligible = Column(Boolean, nullable=False, default=True)
    exclusion_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        CheckConstraint(
            "task IN ('behavior_imitation', 'setup_outcome_quality', "
            "'conviction_behavior', 'copier_execution_quality')",
            name="ck_v2_learning_example_task",
        ),
        CheckConstraint(
            "label_state IN ('UNLABELED', 'POSITIVE_TRADE', 'EXPLICIT_PASS', 'OBSERVED_TARGET')",
            name="ck_v2_learning_example_label_state",
        ),
        CheckConstraint(
            "seen_by_conner IN ('UNKNOWN', 'CONFIRMED_SEEN', 'CONFIRMED_NOT_SEEN')",
            name="ck_v2_learning_example_seen",
        ),
        CheckConstraint(
            "consideration_status IN ('UNKNOWN', 'PRESENTED', 'CONFIRMED_CONSIDERED')",
            name="ck_v2_learning_example_consideration",
        ),
        CheckConstraint(
            "source_kind IN ('FUTURE_LIVE_MASTER', 'BLIND_POINT_IN_TIME_REPLAY', "
            "'FINALIZED_HISTORICAL_MARKET', 'EXPLICIT_TRADER_FEEDBACK', 'FOLLOWER_EXECUTION')",
            name="ck_v2_learning_example_source_kind",
        ),
        CheckConstraint(
            "(label_state = 'UNLABELED' AND target_name IS NULL AND target_observed_at IS NULL "
            "AND num_nonnulls(target_numeric, target_boolean, target_json) = 0) OR "
            "(label_state <> 'UNLABELED' AND target_name IS NOT NULL "
            "AND target_observed_at IS NOT NULL AND target_observed_at >= feature_cutoff_at "
            "AND num_nonnulls(target_numeric, target_boolean, target_json) = 1)",
            name="ck_v2_learning_example_target",
        ),
        CheckConstraint(
            "label_state <> 'POSITIVE_TRADE' OR "
            "(task = 'behavior_imitation' AND master_trade_id IS NOT NULL "
            "AND behavior_label_id IS NOT NULL AND seen_by_conner = 'CONFIRMED_SEEN' "
            "AND consideration_status = 'CONFIRMED_CONSIDERED')",
            name="ck_v2_learning_example_positive",
        ),
        CheckConstraint(
            "label_state <> 'EXPLICIT_PASS' OR "
            "(task = 'behavior_imitation' AND master_trade_id IS NULL "
            "AND behavior_label_id IS NOT NULL AND seen_by_conner = 'CONFIRMED_SEEN' "
            "AND consideration_status = 'CONFIRMED_CONSIDERED')",
            name="ck_v2_learning_example_rejection",
        ),
        CheckConstraint(
            "label_state <> 'OBSERVED_TARGET' OR task <> 'behavior_imitation'",
            name="ck_v2_learning_example_observed_target",
        ),
        CheckConstraint(
            "label_state <> 'UNLABELED' OR behavior_label_id IS NULL",
            name="ck_v2_learning_example_unlabeled",
        ),
        CheckConstraint(
            "(task = 'copier_execution_quality' AND follower_trade_id IS NOT NULL "
            "AND source_kind = 'FOLLOWER_EXECUTION' "
            "AND strategy_namespace IS NULL AND market_state_id IS NULL "
            "AND opportunity_id IS NULL AND behavior_label_id IS NULL AND master_trade_id IS NULL) OR "
            "(task <> 'copier_execution_quality' AND follower_trade_id IS NULL "
            "AND source_kind <> 'FOLLOWER_EXECUTION' "
            "AND strategy_namespace IS NOT NULL AND market_state_id IS NOT NULL)",
            name="ck_v2_learning_example_task_boundary",
        ),
        CheckConstraint(
            "task <> 'conviction_behavior' OR source_kind = 'FUTURE_LIVE_MASTER'",
            name="ck_v2_learning_example_conviction_source",
        ),
        CheckConstraint(
            "task <> 'behavior_imitation' OR source_kind <> 'FINALIZED_HISTORICAL_MARKET'",
            name="ck_v2_learning_example_behavior_source",
        ),
        CheckConstraint(
            "eligible OR exclusion_reason IS NOT NULL", name="ck_v2_learning_example_exclusion"
        ),
        CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_learning_example_lineage"),
        CheckConstraint(
            "relevance_grade IS NULL OR relevance_grade >= 0",
            name="ck_v2_learning_example_relevance",
        ),
        CheckConstraint(
            "relevance_grade IS NULL OR ranking_group_id IS NOT NULL",
            name="ck_v2_learning_example_ranking_group",
        ),
        Index("ix_v2_learning_examples_task_date_label", "task", "trade_date", "label_state"),
    )


class V2ScoutRun(Base):
    __tablename__ = "v2_scout_runs"
    run_id = Column(String(96), primary_key=True)
    strategy_namespace = Column(String(64), nullable=False)
    query_group_id = Column(String(96), nullable=False)
    trade_date = Column(Date, nullable=False)
    prediction_at = Column(DateTime(timezone=True), nullable=False)
    feature_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    nq_contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    es_contract_id = Column(String(96), ForeignKey("v2_futures_contracts.contract_id"), nullable=False)
    origin = Column(String(32), nullable=False)
    observation_spec_version = Column(String(64), nullable=False)
    candidate_set_version = Column(String(64), nullable=False)
    feature_schema_version = Column(String(64), nullable=False)
    ranking_policy_version = Column(String(64), nullable=False)
    behavior_artifact_id = Column(String(96), ForeignKey("v2_model_artifacts.artifact_id"), nullable=True)
    outcome_artifact_id = Column(String(96), ForeignKey("v2_model_artifacts.artifact_id"), nullable=True)
    conviction_artifact_id = Column(String(96), ForeignKey("v2_model_artifacts.artifact_id"), nullable=True)
    model_versions = Column(JSONB, nullable=False, default=dict)
    data_quality = Column(JSONB, nullable=False)
    source_lineage = Column(JSONB, nullable=False)
    execution_eligible = Column(Boolean, nullable=False, default=False)
    shadow_only = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        CheckConstraint("feature_cutoff_at <= prediction_at", name="ck_v2_scout_run_cutoff"),
        CheckConstraint("nq_contract_id <> es_contract_id", name="ck_v2_scout_run_contracts"),
        CheckConstraint(
            "origin IN ('LIVE', 'BLIND_REPLAY', 'RETROSPECTIVE_TRADE_LEADUP')",
            name="ck_v2_scout_run_origin",
        ),
        CheckConstraint("execution_eligible = false", name="ck_v2_scout_run_execution"),
        CheckConstraint("shadow_only = true", name="ck_v2_scout_run_shadow_only"),
        CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_scout_run_lineage"),
        Index("ix_v2_scout_runs_date_cutoff", "trade_date", "feature_cutoff_at"),
    )


class V2ScoutRanking(Base):
    __tablename__ = "v2_scout_rankings"
    ranking_id = Column(String(96), primary_key=True)
    run_id = Column(String(96), ForeignKey("v2_scout_runs.run_id"), nullable=False)
    opportunity_id = Column(String(96), ForeignKey("v2_scout_opportunities.opportunity_id"), nullable=False)
    market_state_id = Column(
        String(96), ForeignKey("v2_synchronized_market_states.market_state_id"), nullable=False
    )
    rank = Column(Integer, nullable=False)
    predicted_direction = Column(String(8), nullable=True)
    behavior_score = Column(MONEY, nullable=True)
    outcome_prediction = Column(JSONB, nullable=True)
    conviction_prediction = Column(JSONB, nullable=True)
    model_confidence = Column(MONEY, nullable=True)
    data_quality_score = Column(MONEY, nullable=True)
    ranking_score = Column(MONEY, nullable=True)
    ranking_components = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("run_id", "rank", name="uq_v2_scout_ranking_rank"),
        UniqueConstraint("run_id", "opportunity_id", name="uq_v2_scout_ranking_opportunity"),
        CheckConstraint("rank >= 1", name="ck_v2_scout_ranking_rank"),
        CheckConstraint(
            "predicted_direction IS NULL OR predicted_direction IN ('LONG', 'SHORT')",
            name="ck_v2_scout_ranking_direction",
        ),
        CheckConstraint(
            "behavior_score IS NULL OR behavior_score BETWEEN 0 AND 1",
            name="ck_v2_scout_ranking_behavior",
        ),
        CheckConstraint(
            "model_confidence IS NULL OR model_confidence BETWEEN 0 AND 1",
            name="ck_v2_scout_ranking_confidence",
        ),
        CheckConstraint(
            "data_quality_score IS NULL OR data_quality_score BETWEEN 0 AND 1",
            name="ck_v2_scout_ranking_data_quality",
        ),
        CheckConstraint(
            "ranking_score IS NULL OR ranking_score BETWEEN 0 AND 1",
            name="ck_v2_scout_ranking_display_score",
        ),
    )


class V2ShadowEvaluation(Base):
    __tablename__ = "v2_shadow_evaluations"
    shadow_evaluation_id = Column(String(96), primary_key=True)
    ranking_id = Column(String(96), ForeignKey("v2_scout_rankings.ranking_id"), nullable=False)
    evaluation_version = Column(String(64), nullable=False)
    match_status = Column(String(32), nullable=False)
    origin = Column(String(32), nullable=False)
    prediction_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    behavior_label_state = Column(String(32), nullable=False, default="UNLABELED")
    evidence_json = Column(JSONB, nullable=False, default=dict)
    matched_master_trade_id = Column(String(96), ForeignKey("v2_master_trades.trade_id"), nullable=True)
    conner_entry_at = Column(DateTime(timezone=True), nullable=True)
    actual_direction = Column(String(8), nullable=True)
    evaluated_at = Column(DateTime(timezone=True), nullable=False)
    lead_time_ms = Column(BigInteger, nullable=True)
    direction_match = Column(Boolean, nullable=True)
    realized_r = Column(MONEY, nullable=True)
    mfe_r = Column(MONEY, nullable=True)
    mae_r = Column(MONEY, nullable=True)
    outcome_observed_at = Column(DateTime(timezone=True), nullable=True)
    behavior_metrics = Column(JSONB, nullable=False, default=dict)
    outcome_metrics = Column(JSONB, nullable=False, default=dict)
    data_quality_failures = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("ranking_id", "evaluation_version", name="uq_v2_shadow_evaluation_version"),
        CheckConstraint(
            "match_status IN ('PENDING', 'MATCHED_TRADE', 'NO_MATCH_OBSERVED_WINDOW', "
            "'EXPLICIT_REJECTION', 'DATA_INVALID')",
            name="ck_v2_shadow_evaluation_status",
        ),
        CheckConstraint(
            "origin IN ('LIVE', 'BLIND_REPLAY', 'RETROSPECTIVE_TRADE_LEADUP')",
            name="ck_v2_shadow_evaluation_origin",
        ),
        CheckConstraint(
            "behavior_label_state IN ('UNLABELED', 'POSITIVE_TRADE', 'EXPLICIT_PASS')",
            name="ck_v2_shadow_evaluation_behavior",
        ),
        CheckConstraint(
            "actual_direction IS NULL OR actual_direction IN ('LONG', 'SHORT')",
            name="ck_v2_shadow_evaluation_direction",
        ),
        CheckConstraint(
            "(match_status = 'MATCHED_TRADE' AND matched_master_trade_id IS NOT NULL "
            "AND conner_entry_at IS NOT NULL AND actual_direction IS NOT NULL "
            "AND behavior_label_state = 'POSITIVE_TRADE') OR "
            "(match_status = 'EXPLICIT_REJECTION' AND matched_master_trade_id IS NULL "
            "AND conner_entry_at IS NULL AND behavior_label_state = 'EXPLICIT_PASS' "
            "AND actual_direction IS NULL AND lead_time_ms IS NULL "
            "AND direction_match IS NULL AND outcome_observed_at IS NULL "
            "AND evidence_json <> '{}'::jsonb) OR "
            "(match_status IN ('PENDING', 'NO_MATCH_OBSERVED_WINDOW', 'DATA_INVALID') "
            "AND behavior_label_state = 'UNLABELED' AND matched_master_trade_id IS NULL "
            "AND conner_entry_at IS NULL AND actual_direction IS NULL "
            "AND lead_time_ms IS NULL AND direction_match IS NULL "
            "AND outcome_observed_at IS NULL)",
            name="ck_v2_shadow_evaluation_match",
        ),
        CheckConstraint(
            "(origin = 'RETROSPECTIVE_TRADE_LEADUP' AND lead_time_ms IS NULL) OR "
            "(origin IN ('LIVE', 'BLIND_REPLAY') "
            "AND (match_status <> 'MATCHED_TRADE' OR lead_time_ms IS NOT NULL))",
            name="ck_v2_shadow_evaluation_origin_lead_time",
        ),
        CheckConstraint(
            "lead_time_ms IS NULL OR lead_time_ms >= 0", name="ck_v2_shadow_evaluation_lead_time"
        ),
        CheckConstraint(
            "outcome_observed_at IS NULL OR "
            "(conner_entry_at IS NOT NULL AND outcome_observed_at >= conner_entry_at)",
            name="ck_v2_shadow_evaluation_outcome_time",
        ),
        CheckConstraint(
            "matched_master_trade_id IS NOT NULL OR "
            "(realized_r IS NULL AND mfe_r IS NULL AND mae_r IS NULL)",
            name="ck_v2_shadow_evaluation_outcomes",
        ),
        Index("ix_v2_shadow_evaluations_master_trade", "matched_master_trade_id"),
    )


class V2ScoutEvaluationRun(Base):
    __tablename__ = "v2_scout_evaluation_runs"
    evaluation_run_id = Column(String(96), primary_key=True)
    strategy_namespace = Column(String(64), nullable=False)
    task_namespace = Column(String(128), nullable=False)
    evaluation_version = Column(String(64), nullable=False)
    evaluation_mode = Column(String(32), nullable=False)
    dataset_version = Column(String(64), nullable=False)
    metric_version = Column(String(64), nullable=False)
    training_end_trade_date = Column(Date, nullable=True)
    evaluation_start_trade_date = Column(Date, nullable=False)
    evaluation_end_trade_date = Column(Date, nullable=False)
    model_versions = Column(JSONB, nullable=False)
    behavior_metrics = Column(JSONB, nullable=False)
    outcome_metrics = Column(JSONB, nullable=False)
    usefulness_metrics = Column(JSONB, nullable=False)
    data_quality_metrics = Column(JSONB, nullable=False)
    source_lineage = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        CheckConstraint(
            "evaluation_mode IN ('SHADOW_FORWARD', 'WALK_FORWARD')",
            name="ck_v2_scout_evaluation_mode",
        ),
        CheckConstraint(
            "task_namespace LIKE 'v2.futures.%'", name="ck_v2_scout_evaluation_task_namespace"
        ),
        CheckConstraint(
            "evaluation_start_trade_date <= evaluation_end_trade_date",
            name="ck_v2_scout_evaluation_dates",
        ),
        CheckConstraint(
            "training_end_trade_date IS NULL OR "
            "training_end_trade_date < evaluation_start_trade_date",
            name="ck_v2_scout_evaluation_chronology",
        ),
        CheckConstraint(
            "evaluation_mode <> 'WALK_FORWARD' OR training_end_trade_date IS NOT NULL",
            name="ck_v2_scout_evaluation_walk_forward",
        ),
        CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_scout_evaluation_lineage"),
        Index(
            "ix_v2_scout_evaluations_dates",
            "evaluation_start_trade_date",
            "evaluation_end_trade_date",
        ),
    )


# Rithmic capture persistence is intentionally additive to the broker-neutral V2
# foundation above.  These tables journal broker observations; none represents a
# command queue or grants permission to mutate a broker account.


class V2RithmicConnectionGeneration(Base):
    __tablename__ = "v2_rithmic_connection_generations"
    generation_id = Column(String(96), primary_key=True)
    connection_id = Column(
        String(96), ForeignKey("v2_broker_connections.connection_id"), nullable=False
    )
    generation_ordinal = Column(BigInteger, nullable=False)
    plant = Column(String(16), nullable=False)
    system_name = Column(String(128), nullable=False)
    state = Column(String(32), nullable=False)
    reconnect_attempt = Column(Integer, nullable=False, default=0)
    heartbeat_interval_ms = Column(BigInteger, nullable=True)
    connected_at = Column(DateTime(timezone=True), nullable=False)
    authenticated_at = Column(DateTime(timezone=True), nullable=True)
    reconciled_at = Column(DateTime(timezone=True), nullable=True)
    disconnected_at = Column(DateTime(timezone=True), nullable=True)
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    forced_logout = Column(Boolean, nullable=False, default=False)
    ready = Column(Boolean, nullable=False, default=False)
    disconnect_reason = Column(Text, nullable=True)
    state_details = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint(
            "connection_id",
            "plant",
            "generation_ordinal",
            name="uq_v2_rithmic_generation",
        ),
        CheckConstraint(
            "plant IN ('ORDER', 'PNL', 'TICKER')",
            name="ck_v2_rithmic_generation_plant",
        ),
        CheckConstraint(
            "generation_ordinal >= 0 AND reconnect_attempt >= 0",
            name="ck_v2_rithmic_generation_counts",
        ),
        CheckConstraint(
            "heartbeat_interval_ms IS NULL OR heartbeat_interval_ms > 0",
            name="ck_v2_rithmic_generation_heartbeat",
        ),
        Index(
            "ix_v2_rithmic_generation_connection_plant",
            "connection_id",
            "plant",
            "connected_at",
        ),
    )


class V2RithmicReplayBatch(Base):
    __tablename__ = "v2_rithmic_replay_batches"
    replay_batch_id = Column(String(96), primary_key=True)
    generation_id = Column(
        String(96),
        ForeignKey("v2_rithmic_connection_generations.generation_id"),
        nullable=False,
    )
    generation_map = Column(JSONB, nullable=False, default=dict)
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    broker_account_id = Column(String(256), nullable=True)
    batch_kind = Column(String(48), nullable=False)
    source_kind = Column(String(24), nullable=False)
    request_key = Column(String(256), nullable=True)
    user_message = Column(String(256), nullable=True)
    status = Column(String(32), nullable=False)
    requested_at = Column(DateTime(timezone=True), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    terminal_response_received = Column(Boolean, nullable=False, default=False)
    event_count = Column(BigInteger, nullable=False, default=0)
    boundary_facts = Column(JSONB, nullable=False, default=dict)
    failure_reason = Column(Text, nullable=True)
    __table_args__ = (
        CheckConstraint("event_count >= 0", name="ck_v2_rithmic_replay_event_count"),
        Index(
            "ix_v2_rithmic_replay_generation_account",
            "generation_id",
            "account_id",
            "requested_at",
        ),
    )


class V2RithmicBrokerEvent(Base):
    """Immutable, append-only journal row for one inbound protocol message."""

    __tablename__ = "v2_rithmic_broker_events"
    event_id = Column(String(96), primary_key=True)
    generation_id = Column(
        String(96),
        ForeignKey("v2_rithmic_connection_generations.generation_id"),
        nullable=False,
    )
    replay_batch_id = Column(
        String(96), ForeignKey("v2_rithmic_replay_batches.replay_batch_id"), nullable=True
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    plant = Column(String(16), nullable=False)
    template_id = Column(Integer, nullable=False)
    template_name = Column(String(128), nullable=False)
    source_kind = Column(String(24), nullable=False)
    local_ingest_sequence = Column(BigInteger, nullable=False)
    request_key = Column(String(256), nullable=True)
    user_message = Column(String(256), nullable=True)
    fcm_id = Column(String(256), nullable=True)
    ib_id = Column(String(256), nullable=True)
    broker_account_id = Column(String(256), nullable=True)
    basket_id = Column(String(256), nullable=True)
    original_basket_id = Column(String(256), nullable=True)
    linked_basket_ids = Column(JSONB, nullable=False, default=list)
    exchange_order_id = Column(String(256), nullable=True)
    ticker_plant_exchange_order_id = Column(String(256), nullable=True)
    fill_id = Column(String(256), nullable=True)
    sequence_number = Column(String(256), nullable=True)
    original_sequence_number = Column(String(256), nullable=True)
    correlation_sequence_number = Column(String(256), nullable=True)
    source_at = Column(DateTime(timezone=True), nullable=True)
    server_at = Column(DateTime(timezone=True), nullable=True)
    exchange_at = Column(DateTime(timezone=True), nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False)
    payload_fingerprint = Column(String(128), nullable=False)
    deduplication_key = Column(String(256), nullable=True)
    event_facts = Column(JSONB, nullable=False, default=dict)
    unknown_fields = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint(
            "generation_id",
            "local_ingest_sequence",
            name="uq_v2_rithmic_event_ingest",
        ),
        UniqueConstraint(
            "deduplication_key",
            name="uq_v2_rithmic_event_dedup",
        ),
        CheckConstraint(
            "local_ingest_sequence >= 0",
            name="ck_v2_rithmic_event_ingest_sequence",
        ),
        CheckConstraint(
            "payload_fingerprint <> ''",
            name="ck_v2_rithmic_event_fingerprint",
        ),
        Index(
            "ix_v2_rithmic_event_account_received",
            "account_id",
            "received_at",
        ),
        Index(
            "ix_v2_rithmic_event_basket_received",
            "basket_id",
            "received_at",
        ),
        Index(
            "ix_v2_rithmic_event_fill",
            "fill_id",
            "received_at",
        ),
    )


class V2RithmicAccountObservation(Base):
    __tablename__ = "v2_rithmic_account_observations"
    observation_id = Column(String(96), primary_key=True)
    event_id = Column(
        String(96), ForeignKey("v2_rithmic_broker_events.event_id"), nullable=True
    )
    generation_id = Column(
        String(96),
        ForeignKey("v2_rithmic_connection_generations.generation_id"),
        nullable=False,
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    fcm_id = Column(String(256), nullable=False)
    ib_id = Column(String(256), nullable=False)
    broker_account_id = Column(String(256), nullable=False)
    account_name = Column(String(256), nullable=True)
    currency = Column(String(16), nullable=True)
    access_type = Column(String(32), nullable=True)
    account_status = Column(String(64), nullable=True)
    user_id = Column(String(256), nullable=True)
    user_type = Column(String(64), nullable=True)
    user_status = Column(String(64), nullable=True)
    order_copy_status = Column(String(64), nullable=True)
    country_code = Column(String(16), nullable=True)
    state_code = Column(String(32), nullable=True)
    max_order_sessions = Column(Integer, nullable=True)
    max_ticker_sessions = Column(Integer, nullable=True)
    allowlisted = Column(Boolean, nullable=False, default=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    account_facts = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_v2_rithmic_account_event"),
        Index(
            "ix_v2_rithmic_account_generation_observed",
            "generation_id",
            "broker_account_id",
            "observed_at",
        ),
        CheckConstraint(
            "max_order_sessions IS NULL OR max_order_sessions >= 0",
            name="ck_v2_rithmic_account_order_sessions",
        ),
        CheckConstraint(
            "max_ticker_sessions IS NULL OR max_ticker_sessions >= 0",
            name="ck_v2_rithmic_account_ticker_sessions",
        ),
    )


class V2RithmicOrderObservation(Base):
    __tablename__ = "v2_rithmic_order_observations"
    order_observation_id = Column(String(96), primary_key=True)
    event_id = Column(
        String(96), ForeignKey("v2_rithmic_broker_events.event_id"), nullable=False
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    fcm_id = Column(String(256), nullable=True)
    ib_id = Column(String(256), nullable=True)
    broker_account_id = Column(String(256), nullable=False)
    basket_id = Column(String(256), nullable=False)
    original_basket_id = Column(String(256), nullable=True)
    linked_basket_ids = Column(JSONB, nullable=False, default=list)
    exchange_order_id = Column(String(256), nullable=True)
    ticker_plant_exchange_order_id = Column(String(256), nullable=True)
    symbol = Column(String(128), nullable=True)
    exchange = Column(String(64), nullable=True)
    normalized_state = Column(String(48), nullable=False, default="UNKNOWN")
    broker_status = Column(String(128), nullable=True)
    notification_type = Column(String(128), nullable=True)
    completion_reason = Column(String(256), nullable=True)
    report_type = Column(String(128), nullable=True)
    command_outcome = Column(String(48), nullable=False, default="NOT_APPLICABLE")
    side = Column(String(32), nullable=True)
    order_type = Column(String(64), nullable=True)
    duration = Column(String(64), nullable=True)
    quantity = Column(BigInteger, nullable=True)
    fill_size = Column(BigInteger, nullable=True)
    total_fill_size = Column(BigInteger, nullable=True)
    total_unfilled_size = Column(BigInteger, nullable=True)
    limit_price = Column(MONEY, nullable=True)
    trigger_price = Column(MONEY, nullable=True)
    fill_price = Column(MONEY, nullable=True)
    average_fill_price = Column(MONEY, nullable=True)
    fill_id = Column(String(256), nullable=True)
    sequence_number = Column(String(256), nullable=True)
    original_sequence_number = Column(String(256), nullable=True)
    correlation_sequence_number = Column(String(256), nullable=True)
    user_id = Column(String(256), nullable=True)
    application = Column(String(256), nullable=True)
    application_version = Column(String(128), nullable=True)
    originator_application = Column(String(256), nullable=True)
    originator_version = Column(String(128), nullable=True)
    window_name = Column(String(256), nullable=True)
    originator_window_name = Column(String(256), nullable=True)
    manual_or_auto = Column(String(32), nullable=True)
    user_tag = Column(String(256), nullable=True)
    mooney_owned = Column(Boolean, nullable=False, default=False)
    unknown_state = Column(Boolean, nullable=False, default=False)
    terminal = Column(Boolean, nullable=False, default=False)
    source_kind = Column(String(24), nullable=False)
    broker_at = Column(DateTime(timezone=True), nullable=True)
    server_at = Column(DateTime(timezone=True), nullable=True)
    exchange_at = Column(DateTime(timezone=True), nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False)
    order_facts = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_v2_rithmic_order_event"),
        Index(
            "ix_v2_rithmic_order_account_basket",
            "broker_account_id",
            "basket_id",
            "received_at",
        ),
        Index(
            "ix_v2_rithmic_order_fill",
            "fill_id",
            "received_at",
        ),
    )


class V2RithmicExecutionObservation(Base):
    __tablename__ = "v2_rithmic_execution_observations"
    execution_observation_id = Column(String(96), primary_key=True)
    event_id = Column(
        String(96), ForeignKey("v2_rithmic_broker_events.event_id"), nullable=False
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    broker_account_id = Column(String(256), nullable=False)
    basket_id = Column(String(256), nullable=False)
    fill_id = Column(String(256), nullable=False)
    execution_key = Column(String(512), nullable=False)
    exchange_order_id = Column(String(256), nullable=True)
    ticker_plant_exchange_order_id = Column(String(256), nullable=True)
    execution_kind = Column(String(32), nullable=False)
    corrects_execution_id = Column(
        String(96),
        ForeignKey("v2_rithmic_execution_observations.execution_observation_id"),
        nullable=True,
    )
    side = Column(String(32), nullable=True)
    quantity = Column(BigInteger, nullable=True)
    effective_quantity_delta = Column(BigInteger, nullable=True)
    price = Column(MONEY, nullable=True)
    commission = Column(MONEY, nullable=True)
    sequence_number = Column(String(256), nullable=True)
    source_kind = Column(String(24), nullable=False)
    executed_at = Column(DateTime(timezone=True), nullable=True)
    server_at = Column(DateTime(timezone=True), nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False)
    execution_facts = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_v2_rithmic_execution_event"),
        UniqueConstraint("execution_key", name="uq_v2_rithmic_execution_key"),
        Index(
            "ix_v2_rithmic_execution_account_fill",
            "broker_account_id",
            "fill_id",
        ),
        Index(
            "ix_v2_rithmic_execution_basket_time",
            "basket_id",
            "received_at",
        ),
    )


class V2RithmicBracketObservation(Base):
    __tablename__ = "v2_rithmic_bracket_observations"
    bracket_observation_id = Column(String(96), primary_key=True)
    event_id = Column(
        String(96), ForeignKey("v2_rithmic_broker_events.event_id"), nullable=False
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    broker_account_id = Column(String(256), nullable=False)
    parent_basket_id = Column(String(256), nullable=False)
    linked_basket_ids = Column(JSONB, nullable=False, default=list)
    bracket_type = Column(String(64), nullable=True)
    operation_type = Column(String(64), nullable=True)
    status = Column(String(64), nullable=True)
    target_total_quantity = Column(BigInteger, nullable=True)
    target_released_quantity = Column(BigInteger, nullable=True)
    stop_total_quantity = Column(BigInteger, nullable=True)
    stop_released_quantity = Column(BigInteger, nullable=True)
    target_tiers = Column(JSONB, nullable=False, default=list)
    stop_tiers = Column(JSONB, nullable=False, default=list)
    trailing_facts = Column(JSONB, nullable=False, default=dict)
    source_kind = Column(String(24), nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False)
    bracket_facts = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "parent_basket_id",
            name="uq_v2_rithmic_bracket_event_parent",
        ),
        Index(
            "ix_v2_rithmic_bracket_account_parent",
            "broker_account_id",
            "parent_basket_id",
            "received_at",
        ),
    )


class V2RithmicReferenceObservation(Base):
    __tablename__ = "v2_rithmic_reference_observations"
    reference_observation_id = Column(String(96), primary_key=True)
    event_id = Column(
        String(96), ForeignKey("v2_rithmic_broker_events.event_id"), nullable=False
    )
    generation_id = Column(
        String(96),
        ForeignKey("v2_rithmic_connection_generations.generation_id"),
        nullable=False,
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    broker_account_id = Column(String(256), nullable=True)
    observation_kind = Column(String(32), nullable=False)
    symbol = Column(String(128), nullable=True)
    exchange = Column(String(64), nullable=True)
    exchange_symbol = Column(String(128), nullable=True)
    symbol_name = Column(String(256), nullable=True)
    trading_symbol = Column(String(128), nullable=True)
    trading_exchange = Column(String(64), nullable=True)
    product_code = Column(String(64), nullable=True)
    instrument_type = Column(String(64), nullable=True)
    underlying_symbol = Column(String(128), nullable=True)
    expiration_date = Column(Date, nullable=True)
    currency = Column(String(16), nullable=True)
    tick_size_type = Column(String(64), nullable=True)
    price_display_format = Column(String(64), nullable=True)
    is_tradable = Column(Boolean, nullable=True)
    minimum_quoted_price_change = Column(MONEY, nullable=True)
    minimum_feed_price_change = Column(MONEY, nullable=True)
    single_point_value = Column(MONEY, nullable=True)
    quote_to_feed_price_factor = Column(MONEY, nullable=True)
    feed_to_quote_price_factor = Column(MONEY, nullable=True)
    tick_table_first_price = Column(MONEY, nullable=True)
    tick_table_last_price = Column(MONEY, nullable=True)
    tick_table_first_price_operator = Column(String(32), nullable=True)
    tick_table_last_price_operator = Column(String(32), nullable=True)
    presence_bits = Column(BigInteger, nullable=True)
    source_kind = Column(String(24), nullable=False)
    source_at = Column(DateTime(timezone=True), nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False)
    reference_facts = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_v2_rithmic_reference_event"),
        Index(
            "ix_v2_rithmic_reference_symbol_exchange",
            "symbol",
            "exchange",
            "received_at",
        ),
        Index(
            "ix_v2_rithmic_reference_product_expiration",
            "product_code",
            "expiration_date",
            "received_at",
        ),
    )


class V2RithmicPnLObservation(Base):
    __tablename__ = "v2_rithmic_pnl_observations"
    pnl_observation_id = Column(String(96), primary_key=True)
    event_id = Column(
        String(96), ForeignKey("v2_rithmic_broker_events.event_id"), nullable=False
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    broker_account_id = Column(String(256), nullable=False)
    scope = Column(String(24), nullable=False)
    symbol = Column(String(128), nullable=True)
    exchange = Column(String(64), nullable=True)
    currency = Column(String(16), nullable=True)
    trade_date = Column(Date, nullable=True)
    net_quantity = Column(BigInteger, nullable=True)
    long_quantity = Column(BigInteger, nullable=True)
    short_quantity = Column(BigInteger, nullable=True)
    open_quantity = Column(BigInteger, nullable=True)
    closed_quantity = Column(BigInteger, nullable=True)
    working_buy_quantity = Column(BigInteger, nullable=True)
    working_sell_quantity = Column(BigInteger, nullable=True)
    average_open_fill_price = Column(MONEY, nullable=True)
    open_position_pnl = Column(MONEY, nullable=True)
    closed_position_pnl = Column(MONEY, nullable=True)
    day_open_pnl = Column(MONEY, nullable=True)
    day_closed_pnl = Column(MONEY, nullable=True)
    day_total_pnl = Column(MONEY, nullable=True)
    day_open_pnl_offset = Column(MONEY, nullable=True)
    day_closed_pnl_offset = Column(MONEY, nullable=True)
    account_balance = Column(MONEY, nullable=True)
    cash_on_hand = Column(MONEY, nullable=True)
    margin_balance = Column(MONEY, nullable=True)
    available_buying_power = Column(MONEY, nullable=True)
    used_buying_power = Column(MONEY, nullable=True)
    reserved_buying_power = Column(MONEY, nullable=True)
    excess_buy_margin = Column(MONEY, nullable=True)
    excess_sell_margin = Column(MONEY, nullable=True)
    commission = Column(MONEY, nullable=True)
    source_kind = Column(String(24), nullable=False)
    is_snapshot = Column(Boolean, nullable=False, default=False)
    freshness = Column(String(24), nullable=False, default="UNKNOWN")
    source_at = Column(DateTime(timezone=True), nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False)
    pnl_facts = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_v2_rithmic_pnl_event"),
        CheckConstraint(
            "scope IN ('ACCOUNT', 'INSTRUMENT', 'UNKNOWN')",
            name="ck_v2_rithmic_pnl_scope",
        ),
        Index(
            "ix_v2_rithmic_pnl_account_received",
            "broker_account_id",
            "received_at",
        ),
        Index(
            "ix_v2_rithmic_pnl_instrument_received",
            "symbol",
            "exchange",
            "received_at",
        ),
    )


class V2RithmicRmsObservation(Base):
    __tablename__ = "v2_rithmic_rms_observations"
    rms_observation_id = Column(String(96), primary_key=True)
    event_id = Column(
        String(96), ForeignKey("v2_rithmic_broker_events.event_id"), nullable=False
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    broker_account_id = Column(String(256), nullable=False)
    scope = Column(String(24), nullable=False)
    product_code = Column(String(64), nullable=True)
    currency = Column(String(16), nullable=True)
    status = Column(String(64), nullable=True)
    algorithm = Column(String(128), nullable=True)
    loss_limit = Column(MONEY, nullable=True)
    minimum_account_balance = Column(MONEY, nullable=True)
    minimum_margin_balance = Column(MONEY, nullable=True)
    account_balance = Column(MONEY, nullable=True)
    current_auto_liquidate_threshold = Column(MONEY, nullable=True)
    peak_account_balance = Column(MONEY, nullable=True)
    peak_account_balance_at = Column(DateTime(timezone=True), nullable=True)
    auto_liquidate = Column(Boolean, nullable=True)
    auto_liquidate_criteria = Column(String(128), nullable=True)
    disable_on_auto_liquidate = Column(Boolean, nullable=True)
    max_order_quantity = Column(BigInteger, nullable=True)
    buy_limit = Column(BigInteger, nullable=True)
    sell_limit = Column(BigInteger, nullable=True)
    buy_margin_rate = Column(MONEY, nullable=True)
    sell_margin_rate = Column(MONEY, nullable=True)
    commission_rate = Column(MONEY, nullable=True)
    source_kind = Column(String(24), nullable=False)
    source_at = Column(DateTime(timezone=True), nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False)
    rms_facts = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_v2_rithmic_rms_event"),
        CheckConstraint(
            "scope IN ('ACCOUNT', 'PRODUCT', 'UNKNOWN')",
            name="ck_v2_rithmic_rms_scope",
        ),
        Index(
            "ix_v2_rithmic_rms_account_product",
            "broker_account_id",
            "product_code",
            "received_at",
        ),
    )


class V2RithmicReconciliationCheckpoint(Base):
    __tablename__ = "v2_rithmic_reconciliation_checkpoints"
    checkpoint_id = Column(String(96), primary_key=True)
    generation_id = Column(
        String(96),
        ForeignKey("v2_rithmic_connection_generations.generation_id"),
        nullable=False,
    )
    account_id = Column(
        String(96), ForeignKey("v2_broker_accounts.account_id"), nullable=True
    )
    broker_account_id = Column(String(256), nullable=True)
    plant = Column(String(16), nullable=False)
    phase = Column(String(48), nullable=False)
    status = Column(String(32), nullable=False)
    live_subscription_active = Column(Boolean, nullable=False, default=False)
    live_buffer_started_at = Column(DateTime(timezone=True), nullable=True)
    snapshot_requested_at = Column(DateTime(timezone=True), nullable=True)
    snapshot_completed_at = Column(DateTime(timezone=True), nullable=True)
    buffered_events_applied_at = Column(DateTime(timezone=True), nullable=True)
    highest_ingest_sequence = Column(BigInteger, nullable=True)
    order_cursor = Column(String(256), nullable=True)
    execution_cursor = Column(String(256), nullable=True)
    fill_cursor = Column(String(256), nullable=True)
    pnl_cursor = Column(String(256), nullable=True)
    discrepancy_count = Column(BigInteger, nullable=False, default=0)
    ready = Column(Boolean, nullable=False, default=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    failure_reason = Column(Text, nullable=True)
    checkpoint_facts = Column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        CheckConstraint(
            "highest_ingest_sequence IS NULL OR highest_ingest_sequence >= 0",
            name="ck_v2_rithmic_checkpoint_ingest",
        ),
        CheckConstraint(
            "discrepancy_count >= 0",
            name="ck_v2_rithmic_checkpoint_discrepancy",
        ),
        CheckConstraint(
            "NOT ready OR (status = 'COMPLETE' AND discrepancy_count = 0)",
            name="ck_v2_rithmic_checkpoint_ready",
        ),
        Index(
            "ix_v2_rithmic_checkpoint_generation_account",
            "generation_id",
            "account_id",
            "recorded_at",
        ),
    )
