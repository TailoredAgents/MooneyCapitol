"""Explicit V1 quarantine markers used by migration and registry tests."""

V1_ASSET_CLASS = "equity"
V1_ARTIFACT_NAMESPACES = frozenset({"learning", "shadow-v1", "paper-v1", "hybrid_target"})
V2_MAY_NOT_IMPORT = frozenset({"app.learning", "app.ai", "openai", "xgboost", "sklearn"})
