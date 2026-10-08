import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.slack_security import require_valid_slack_signature
from app.api.routes.config import _validate_execution_transition
from app.copier.environment_safety import environment_safety_blocks
from app.core.config import AppConfig
from app.v2.brokers.base import BrokerSubmissionDisabled, OrderRequest
from app.v2.brokers.rithmic import RithmicAdapterPlaceholder
from app.v2.execution.config import ExecutionConfigSnapshot
from app.v2.execution.service import FuturesExecutionService
from app.v2.learning.contracts import (
    DatasetContract,
    DatasetRow,
    FeatureDefinition,
    FeatureObservation,
    FeatureSnapshot,
    LearningTask,
    ModelArtifactMetadata,
    PromotionStatus,
)
from app.v2.learning.registry import FuturesModelRegistry, ModelSelectionError
from app.v2.learning.validation import LeakageError, validate_dataset_row
from app.v2.security import redact_sensitive


NOW = datetime(2026, 10, 5, 14, tzinfo=timezone.utc)


def _contract(task=LearningTask.SETUP_OUTCOME_QUALITY, target="realized_r"):
    return DatasetContract(task, "dataset-1", "features-1", (FeatureDefinition("entry_context", "decimal"),), target, {"source": "observed"})


def _row(contract, **changes):
    snapshot = FeatureSnapshot(
        "opp-1",
        "CME:NQ:2027-03-19",
        NOW,
        "features-1",
        (FeatureObservation("entry_context", "x", NOW - timedelta(seconds=1), "feed"),),
    )
    values = dict(
        task=contract.task,
        opportunity_id="opp-1",
        snapshot=snapshot,
        target_name=contract.target_name,
        target_value=1,
        target_observed_at=NOW + timedelta(hours=1),
    )
    values.update(changes)
    return DatasetRow(**values)


def test_registry_cannot_load_v1_equity_artifact():
    @dataclass
    class LegacyArtifact:
        artifact_id: str = "legacy"
        asset_class: str = "equity"
        artifact_namespace: str = "hybrid_target"

    registry = FuturesModelRegistry()
    with pytest.raises(ModelSelectionError, match="quarantined"):
        registry.register(LegacyArtifact())


def test_learning_tasks_have_independent_namespaces_and_no_hybrid_target():
    registry = FuturesModelRegistry()
    for task in LearningTask:
        artifact = ModelArtifactMetadata(
            task.value,
            task,
            "features-1",
            "dataset-1",
            f"v2.futures.{task.value}",
            {},
            PromotionStatus.PROMOTED,
            {"source": "observed"},
        )
        registry.register(artifact)
        assert registry.select(task, product_code="NQ", feature_schema_version="features-1") == artifact
        assert "hybrid_target" not in artifact.artifact_namespace


@pytest.mark.parametrize(
    "contract,row,match",
    [
        (_contract(), lambda c: _row(c, source_kind="synthetic"), "synthetic"),
        (_contract(LearningTask.BEHAVIOR_IMITATION, "conner_action"), lambda c: _row(c), "unseen"),
        (_contract(), lambda c: _row(c, strategy_quality_label="good", follower_execution_quality={"slippage": 1}), "follower"),
        (_contract(LearningTask.SETUP_OUTCOME_QUALITY, "projected_rr"), lambda c: _row(c), "projected"),
    ],
)
def test_dataset_guardrails(contract, row, match):
    with pytest.raises(LeakageError, match=match):
        validate_dataset_row(row(contract), contract)


def test_target_feature_future_data_and_silent_zero_are_rejected():
    for feature in (
        FeatureObservation("realized_r", 1, NOW, "outcome"),
        FeatureObservation("entry_context", 1, NOW + timedelta(seconds=1), "feed"),
        FeatureObservation("entry_context", 0, NOW, "feed", was_missing=True),
    ):
        contract = DatasetContract(
            LearningTask.SETUP_OUTCOME_QUALITY,
            "d1",
            "f1",
            (FeatureDefinition(feature.name, "decimal"),),
            "realized_r",
            {},
        )
        snapshot = FeatureSnapshot("o", "CME:NQ:2027-03-19", NOW, "f1", (feature,))
        with pytest.raises(LeakageError):
            validate_dataset_row(
                DatasetRow(contract.task, "o", snapshot, contract.target_name, 1, NOW + timedelta(hours=1)),
                contract,
            )


def test_execution_service_is_foundation_only_and_order_submission_is_impossible():
    with pytest.raises(ValueError, match="cannot be enabled"):
        ExecutionConfigSnapshot("bad", submission_enabled=True)
    adapter = RithmicAdapterPlaceholder()
    with pytest.raises(BrokerSubmissionDisabled):
        asyncio.run(adapter.submit(OrderRequest("a", "c", "BUY", 1, "MARKET", "client")))
    service = FuturesExecutionService(ExecutionConfigSnapshot("v1"), adapter=adapter)
    health = asyncio.run(service.start())
    assert health.live and not health.ready and not health.submission_enabled
    asyncio.run(service.stop())


def test_execution_plane_has_no_ai_or_ml_imports():
    text = "\n".join(path.read_text(encoding="utf-8") for path in Path("app/v2/execution").glob("*.py"))
    for forbidden in ("import openai", "from openai", "import xgboost", "import sklearn", "app.learning"):
        assert forbidden not in text.lower()


def test_slack_mutations_fail_closed_without_secret(monkeypatch):
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    with pytest.raises(HTTPException) as exc:
        require_valid_slack_signature("1", b"payload=x", "v0=nope")
    assert exc.value.status_code == 503


def test_sensitive_values_are_redacted_recursively():
    result = redact_sensitive({"account_id": "123", "nested": {"apiKey": "abc", "safe": "ok"}})
    assert result == {"account_id": "[REDACTED]", "nested": {"apiKey": "[REDACTED]", "safe": "ok"}}


def test_whole_config_cannot_bypass_execution_confirmations_or_live_ceiling():
    current = AppConfig()
    proposed = current.model_copy(deep=True)
    proposed.copier.enabled = True
    proposed.copier.mode = "live"
    proposed.copier.live_trading_enabled = True
    proposed.copier.global_kill_switch = False
    with pytest.raises(HTTPException) as confirmation_error:
        _validate_execution_transition(current, proposed, None)
    assert "Confirmation required" in confirmation_error.value.detail
    confirmations = "ENABLE_COPIER,SET_LIVE_MODE,DISABLE_KILL_SWITCH"
    with pytest.raises(HTTPException) as ceiling_error:
        _validate_execution_transition(current, proposed, confirmations)
    assert "notional ceiling" in ceiling_error.value.detail
    proposed.copier.live_max_notional_per_order = 1000
    _validate_execution_transition(current, proposed, confirmations)


def test_deployment_flags_are_enforced_as_deny_only_gates(monkeypatch):
    config = AppConfig().copier
    config.enabled = True
    monkeypatch.setenv("COPIER_ENABLED", "0")
    monkeypatch.setenv("COPIER_MODE", "live")
    monkeypatch.setenv("COPIER_GLOBAL_KILL_SWITCH", "1")
    blockers = environment_safety_blocks(config)
    assert len(blockers) == 3
    # No environment value can mutate persisted configuration into an enabled state.
    assert config.enabled and config.mode == "test"
