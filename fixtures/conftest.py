"""Current-main 3fbb3bba private draft, unexecuted. No installer, launcher or resource-control claims."""
from __future__ import annotations
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent

def pytest_addoption(parser):
    parser.addoption("--run-serialized-assets", action="store_true", default=False)

def _read_head(repo):
    gitdir = repo / ".git"
    if gitdir.is_file():
        line = gitdir.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir: "):
            raise pytest.UsageError("Unrecognized Git directory pointer")
        gitdir = (repo / line[8:]).resolve()
    common = gitdir
    if (gitdir / "commondir").is_file():
        common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve()
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    for base in (gitdir, common):
        if (base / ref).is_file():
            return (base / ref).read_text().strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                sha, name = line.split(" ", 1)
                if name == ref:
                    return sha
    raise pytest.UsageError("Unable to bind source HEAD without a Git command")

@pytest.fixture(scope="session")
def runtime():
    """Import real production modules only after explicit source binding."""
    supplied = os.environ.get("VESUVIUS_REGRESSION_SOURCE_ROOT")
    if not supplied:
        raise pytest.UsageError("Set VESUVIUS_REGRESSION_SOURCE_ROOT to the reviewed checkout")
    repo = Path(supplied).resolve(strict=True)
    manifest = json.loads((HERE / "source-pins.json").read_text())
    if _read_head(repo) != manifest["source_revision"]:
        raise pytest.UsageError("Source revision differs; no automatic fetch/update allowed")
    for relative, expected in manifest["files"].items():
        text = (repo / relative).read_text(encoding="utf-8")
        body = text.replace("\r\n", "\n").encode("utf-8")
        actual = hashlib.sha1(b"blob " + str(len(body)).encode("ascii") + b"\0" + body).hexdigest()
        if actual != expected:
            raise pytest.UsageError("Current-main source mismatch: " + relative)
    if sys.version_info[:2] != (3, 14):
        raise pytest.UsageError("Reviewed Villa declares Python >=3.14,<3.15; do not substitute")
    if any(name == "vesuvius" or name.startswith("vesuvius.") for name in sys.modules):
        raise pytest.UsageError("Use a fresh private pytest session; vesuvius was preloaded")
    sys.path.insert(0, str(repo / "vesuvius" / "src"))
    importlib.invalidate_caches()
    import numpy as np
    import torch
    import zarr
    from vesuvius.ink_detection.config import InkDataConfig
    from vesuvius.ink_detection.data.dataset import InkDataset
    from vesuvius.ink_detection.data import augmentations
    from vesuvius.ink_detection.types import Patch, Segment
    import vesuvius.ink_detection.data.dataset as dataset_module
    assert Path(dataset_module.__file__).resolve() == (
        repo / "vesuvius/src/vesuvius/ink_detection/data/dataset.py"
    ).resolve()
    assert (
        augmentations.NATIVE_CROP_TRANSLATION_PROBABILITY,
        augmentations.NATIVE_CROP_TRANSLATION_MIN_VOXELS,
        augmentations.NATIVE_CROP_TRANSLATION_MAX_VOXELS,
        augmentations.NATIVE_CROP_TRANSLATION_KEEP_FRACTION,
        augmentations.NATIVE_CROP_TRANSLATION_MAX_AXES,
        augmentations.NATIVE_CROP_TRANSLATION_MAX_ATTEMPTS,
    ) == (0.30, 10, 40, 1.0 / 3.0, 2, 24)
    # Exact current upstream class AND its _sampled helper, not replacement filtering.
    fixture_path = repo / "vesuvius/tests/ink_detection/test_data_foundation.py"
    fixture_text = fixture_path.read_text()
    classes = [
        node for node in ast.parse(fixture_text).body
        if (isinstance(node, ast.ClassDef) and node.name == "_FakeTifxyz")
        or (isinstance(node, ast.FunctionDef) and node.name == "_sampled")
    ]
    assert len(classes) == 2
    fixture_class = next(node for node in classes if isinstance(node, ast.ClassDef))
    namespace = {"np": np}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(fixture_path), "exec"), namespace)
    return SimpleNamespace(
        repo=repo, np=np, torch=torch, zarr=zarr, InkDataConfig=InkDataConfig,
        InkDataset=InkDataset, augmentations=augmentations, Patch=Patch,
        Segment=Segment, dataset_module=dataset_module, FakeTifxyz=namespace["_FakeTifxyz"],
        fixture_source_sha256=hashlib.sha256(
            ast.get_source_segment(fixture_text, fixture_class).encode()
        ).hexdigest(),
    )

