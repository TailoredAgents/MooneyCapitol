from __future__ import annotations

from app.v2.learning.contracts import LearningTask, ModelArtifactMetadata, PromotionStatus


class ModelSelectionError(LookupError):
    pass


class FuturesModelRegistry:
    """Metadata registry that cannot address V1 equity namespaces."""

    def __init__(self) -> None:
        self._artifacts: dict[str, ModelArtifactMetadata] = {}

    def register(self, artifact: ModelArtifactMetadata) -> None:
        # Repeat the boundary checks even when an untyped legacy object is passed.
        if getattr(artifact, "asset_class", "").lower() != "futures":
            raise ModelSelectionError("legacy/equity artifacts are quarantined from V2")
        namespace = getattr(artifact, "artifact_namespace", "")
        if not namespace.startswith("v2.futures."):
            raise ModelSelectionError("artifact is outside the V2 futures namespace")
        self._artifacts[artifact.artifact_id] = artifact

    def select(
        self,
        task: LearningTask,
        *,
        product_code: str,
        feature_schema_version: str,
        promotion_status: PromotionStatus = PromotionStatus.PROMOTED,
    ) -> ModelArtifactMetadata:
        product = product_code.upper()
        matches = [
            artifact
            for artifact in self._artifacts.values()
            if artifact.task == task
            and artifact.asset_class == "futures"
            and artifact.artifact_namespace == f"v2.futures.{task.value}"
            and product in artifact.product_codes
            and artifact.feature_schema_version == feature_schema_version
            and artifact.promotion_status == promotion_status
        ]
        if not matches:
            raise ModelSelectionError("no compatible V2 futures artifact")
        if len(matches) > 1:
            raise ModelSelectionError("artifact selection is ambiguous; use an explicit version")
        return matches[0]

    def get(self, artifact_id: str) -> ModelArtifactMetadata:
        try:
            artifact = self._artifacts[artifact_id]
        except KeyError as exc:
            raise ModelSelectionError("unknown V2 futures artifact") from exc
        if artifact.asset_class != "futures" or not artifact.artifact_namespace.startswith("v2.futures."):
            raise ModelSelectionError("artifact failed V2 quarantine check")
        return artifact
