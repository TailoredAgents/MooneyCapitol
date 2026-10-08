from __future__ import annotations

import base64
import hashlib
import hmac
import importlib
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from .constants import (
    INBOUND_BINDINGS,
    OUTBOUND_BINDINGS,
    PROTOCOL_PACKAGE_VERSION,
    PROTOCOL_TEMPLATE_VERSION,
)
from .errors import BindingsUnavailable, UnsupportedTemplate
from .framing import extract_template_id


@dataclass(frozen=True)
class ExternalBindingsConfig:
    """Location of locally generated vendor bindings; never copied into Git.

    A zip archive is supported for secret-file deployment. Its checksum is
    mandatory and extraction is constrained to generated ``*_pb2.py`` files.
    """

    generated_path: Path | None = None
    archive_path: Path | None = None
    archive_b64_file: Path | None = None
    archive_sha256: str | None = None
    sdk_path: Path | None = None
    forbid_workspace_root: Path | None = None

    @classmethod
    def from_environment(cls, *, forbid_workspace_root: Path | None = None) -> "ExternalBindingsConfig":
        generated = os.getenv("RITHMIC_GENERATED_BINDINGS_PATH")
        archive = os.getenv("RITHMIC_GENERATED_BINDINGS_ARCHIVE")
        archive_b64 = os.getenv("RITHMIC_GENERATED_BINDINGS_ARCHIVE_B64_FILE")
        sdk = os.getenv("RITHMIC_PROTOCOL_SDK_PATH")
        return cls(
            generated_path=Path(generated) if generated else None,
            archive_path=Path(archive) if archive else None,
            archive_b64_file=Path(archive_b64) if archive_b64 else None,
            archive_sha256=os.getenv("RITHMIC_GENERATED_BINDINGS_SHA256"),
            sdk_path=Path(sdk) if sdk else None,
            forbid_workspace_root=forbid_workspace_root,
        )


class PreparedBindings:
    def __init__(self, path: Path, temporary: tempfile.TemporaryDirectory[str] | None = None) -> None:
        self.path = path
        self._temporary = temporary

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> "PreparedBindings":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_external_path(path: Path, workspace_root: Path | None) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise BindingsUnavailable("generated bindings path is not a directory")
    if workspace_root is not None and _is_relative_to(resolved, workspace_root.resolve()):
        raise BindingsUnavailable("generated Rithmic bindings must remain outside the repository")
    return resolved


def _validate_external_file(path: Path, workspace_root: Path | None, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise BindingsUnavailable(f"{label} is not a file")
    if workspace_root is not None and _is_relative_to(resolved, workspace_root.resolve()):
        raise BindingsUnavailable(f"{label} must remain outside the repository")
    return resolved


def _archive_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_generated_archive(archive: Path, destination: Path) -> Path:
    max_members = 512
    max_uncompressed_bytes = 64 * 1024 * 1024
    total = 0
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        if not members or len(members) > max_members:
            raise BindingsUnavailable("bindings archive has an invalid member count")
        manifests = [
            item
            for item in members
            if not item.is_dir()
            and Path(item.filename.replace("\\", "/")).name
            == "rithmic_bindings_manifest.json"
        ]
        if len(manifests) != 1:
            raise BindingsUnavailable("bindings archive must contain exactly one manifest")
        for member in members:
            name = member.filename.replace("\\", "/")
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise BindingsUnavailable("bindings archive contains an unsafe path")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise BindingsUnavailable("bindings archive may not contain symbolic links")
            allowed_manifest = path.name == "rithmic_bindings_manifest.json"
            if not member.is_dir() and not path.name.endswith("_pb2.py") and not allowed_manifest:
                raise BindingsUnavailable(
                    "bindings archive may contain only generated *_pb2.py files and the bindings manifest"
                )
            total += member.file_size
            if total > max_uncompressed_bytes:
                raise BindingsUnavailable("bindings archive is too large")

        for member in members:
            if member.is_dir():
                continue
            relative = Path(member.filename.replace("\\", "/"))
            target = (destination / relative).resolve()
            if not _is_relative_to(target, destination.resolve()):
                raise BindingsUnavailable("bindings archive escaped its extraction directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

    candidates = [destination, *[p for p in destination.rglob("*") if p.is_dir()]]
    for candidate in candidates:
        if any(candidate.glob("*_pb2.py")):
            return candidate
    raise BindingsUnavailable("bindings archive contains no importable generated modules")


def _required_binding_files() -> frozenset[str]:
    bindings = (*INBOUND_BINDINGS.values(), *OUTBOUND_BINDINGS.values())
    return frozenset(f"{module_name}.py" for module_name, _ in bindings)


def _validate_bindings_manifest(path: Path) -> None:
    manifest_path = path / "rithmic_bindings_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
        raise BindingsUnavailable("generated bindings require a bounded manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BindingsUnavailable("bindings manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "protocol_package",
        "template_version",
        "files",
    }:
        raise BindingsUnavailable("bindings manifest has an unexpected shape")
    if manifest["protocol_package"] != f"RProtocolAPI.{PROTOCOL_PACKAGE_VERSION}":
        raise BindingsUnavailable("bindings manifest protocol package does not match runtime")
    if manifest["template_version"] != PROTOCOL_TEMPLATE_VERSION:
        raise BindingsUnavailable("bindings manifest template version does not match runtime")
    checksums = manifest["files"]
    expected_files = _required_binding_files()
    if not isinstance(checksums, dict) or set(checksums) != expected_files:
        raise BindingsUnavailable("bindings manifest module set does not match runtime")
    actual_files = frozenset(item.name for item in path.glob("*_pb2.py") if item.is_file())
    if actual_files != expected_files:
        raise BindingsUnavailable("generated binding directory contains missing or extra modules")
    for filename in sorted(expected_files):
        expected = checksums[filename]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or expected != expected.lower()
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            raise BindingsUnavailable("bindings manifest contains an invalid checksum")
        if not hmac.compare_digest(_archive_digest(path / filename), expected):
            raise BindingsUnavailable(f"generated binding checksum mismatch: {filename}")


def prepare_bindings(config: ExternalBindingsConfig) -> PreparedBindings:
    configured = sum(
        item is not None
        for item in (config.generated_path, config.archive_path, config.archive_b64_file)
    )
    if configured > 1:
        raise BindingsUnavailable("configure one bindings directory or archive source")
    if config.generated_path is not None:
        path = _validate_external_path(config.generated_path, config.forbid_workspace_root)
        _validate_bindings_manifest(path)
        return PreparedBindings(path)
    if config.archive_path is None and config.archive_b64_file is None:
        raise BindingsUnavailable(
            "configure an external generated-bindings directory or archive"
        )
    expected = (config.archive_sha256 or "").strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise BindingsUnavailable("a valid RITHMIC_GENERATED_BINDINGS_SHA256 is required")
    temporary = tempfile.TemporaryDirectory(prefix="mooney-rithmic-bindings-")
    try:
        temporary_root = Path(temporary.name)
        if config.archive_path is not None:
            archive = _validate_external_file(
                config.archive_path, config.forbid_workspace_root, "bindings archive"
            )
        else:
            encoded_path = _validate_external_file(
                config.archive_b64_file,  # type: ignore[arg-type]
                config.forbid_workspace_root,
                "base64 bindings secret file",
            )
            if encoded_path.stat().st_size > 96 * 1024 * 1024:
                raise BindingsUnavailable("base64 bindings secret file is too large")
            try:
                decoded = base64.b64decode(encoded_path.read_bytes(), validate=True)
            except Exception as exc:
                raise BindingsUnavailable("bindings secret file is not strict base64") from exc
            archive = temporary_root / "bindings.zip"
            archive.write_bytes(decoded)
            decoded = b""
        if not hmac.compare_digest(_archive_digest(archive), expected):
            raise BindingsUnavailable("bindings archive checksum mismatch")
        path = _safe_extract_generated_archive(archive, temporary_root / "extracted")
        _validate_bindings_manifest(path)
    except Exception:
        temporary.cleanup()
        raise
    return PreparedBindings(path, temporary)


class ExternalBindingRegistry:
    """Loads generated protobuf modules from an explicitly external location."""

    def __init__(
        self,
        binding_path: Path,
        *,
        inbound: Mapping[int, tuple[str, str]] = INBOUND_BINDINGS,
        outbound: Mapping[int, tuple[str, str]] = OUTBOUND_BINDINGS,
    ) -> None:
        self.binding_path = binding_path.expanduser().resolve(strict=True)
        if not self.binding_path.is_dir():
            raise BindingsUnavailable("binding_path must be a directory")
        self._inbound = dict(inbound)
        self._outbound = dict(outbound)
        self._modules: dict[str, ModuleType] = {}

    def _load_module(self, module_name: str) -> ModuleType:
        if module_name in self._modules:
            return self._modules[module_name]
        module_file = self.binding_path / f"{module_name}.py"
        if not module_file.is_file():
            raise BindingsUnavailable(f"external generated module is missing: {module_name}")
        existing = sys.modules.get(module_name)
        if existing is not None:
            existing_file = getattr(existing, "__file__", None)
            if existing_file is None or not _is_relative_to(
                Path(existing_file).resolve(), self.binding_path
            ):
                raise BindingsUnavailable(
                    f"refusing already-loaded binding module outside configured path: {module_name}"
                )
        original = list(sys.path)
        try:
            sys.path.insert(0, str(self.binding_path))
            importlib.invalidate_caches()
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise BindingsUnavailable(f"could not import external module: {module_name}") from exc
        finally:
            sys.path[:] = original
        loaded_file = getattr(module, "__file__", None)
        if loaded_file is None or not _is_relative_to(Path(loaded_file).resolve(), self.binding_path):
            raise BindingsUnavailable(
                f"imported binding module escaped configured path: {module_name}"
            )
        self._modules[module_name] = module
        return module

    def message_class(self, template_id: int, *, outbound: bool) -> type[Any]:
        table = self._outbound if outbound else self._inbound
        try:
            module_name, class_name = table[int(template_id)]
        except KeyError as exc:
            raise UnsupportedTemplate(f"template {template_id} has no registered binding") from exc
        module = self._load_module(module_name)
        try:
            message_class = getattr(module, class_name)
        except AttributeError as exc:
            raise BindingsUnavailable(f"{module_name} does not export {class_name}") from exc
        return message_class

    def build(self, template_id: int, **fields: Any) -> Any:
        message = self.message_class(template_id, outbound=True)()
        setattr(message, "template_id", int(template_id))
        for name, value in fields.items():
            target = getattr(message, name, None)
            if isinstance(value, (list, tuple)) and hasattr(target, "extend"):
                target.extend(value)
            else:
                setattr(message, name, value)
        return message

    def decode(self, frame: bytes) -> Any:
        template_id = extract_template_id(frame)
        message = self.message_class(template_id, outbound=False)()
        parser = getattr(message, "ParseFromString", None)
        if not callable(parser):
            raise BindingsUnavailable("generated message does not implement ParseFromString")
        try:
            parser(frame)
        except Exception as exc:
            raise BindingsUnavailable(f"could not decode template {template_id}") from exc
        return message
