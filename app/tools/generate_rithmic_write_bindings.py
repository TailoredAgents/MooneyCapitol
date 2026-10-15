from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from app.tools.generate_rithmic_bindings import (
    _proto_directory,
    _require_external,
    _sha256,
)


WRITE_PROTO_FILES = (
    "request_new_order.proto",
    "response_new_order.proto",
    "request_modify_order.proto",
    "response_modify_order.proto",
    "request_cancel_order.proto",
    "response_cancel_order.proto",
    "request_oco_order.proto",
    "response_oco_order.proto",
    "request_bracket_order.proto",
    "response_bracket_order.proto",
    "request_exit_position.proto",
    "response_exit_position.proto",
)


def generate_write_bindings(sdk_path: Path, output_path: Path) -> dict[str, object]:
    """Generate the separate Test-write binding set outside the repository."""

    if importlib.util.find_spec("grpc_tools.protoc") is None:
        raise RuntimeError(
            "install requirements-rithmic-build.txt in a private/local build environment"
        )
    proto_dir = _proto_directory(sdk_path)
    output_dir = _require_external(output_path, "write bindings output")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"{Path(name).stem}_pb2.py" for name in WRITE_PROTO_FILES}
    existing = {item.name for item in output_dir.glob("*_pb2.py")}
    if existing - expected:
        raise ValueError(
            "write binding output contains unexpected modules; use a clean directory"
        )
    missing = [name for name in WRITE_PROTO_FILES if not (proto_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"official SDK is missing required write schemas: {', '.join(missing)}"
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={output_dir}",
            *(str(proto_dir / name) for name in WRITE_PROTO_FILES),
        ],
        check=True,
    )
    generated = sorted(output_dir / filename for filename in expected)
    if any(not path.is_file() for path in generated):
        raise RuntimeError("protobuf generation did not produce the complete write set")
    manifest: dict[str, object] = {
        "protocol_package": "RProtocolAPI.0.90.0.0",
        "template_version": "5.55",
        "files": {path.name: _sha256(path) for path in generated},
    }
    (output_dir / "rithmic_write_bindings_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate licensed R|Protocol Test-write bindings outside Git"
    )
    parser.add_argument("--sdk-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = generate_write_bindings(args.sdk_path, args.output)
    print(json.dumps({"generated_file_count": len(manifest["files"])}))


if __name__ == "__main__":
    main()
