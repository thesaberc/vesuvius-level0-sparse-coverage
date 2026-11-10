"""Stage and reproduce the published checks with a local offline runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PACKET = Path(__file__).resolve().parent
REVISION = "3fbb3bbadb84fba3648a0c5c7cdbf7d767f9a3c0"
PLANNER = "vesuvius/src/vesuvius/ink_detection/preprocessing/download_required_zarr_chunks.py"
GEOMETRY = "vesuvius/src/vesuvius/ink_detection/data/geometry.py"
REGRESSION = "vesuvius/tests/ink_detection/test_sparse_download_translation.py"
CASES = {
    "m1": "test_sparse_training_witness.py::test_cached_fixture_accepted_crop_has_missing_chunk_and_is_restored",
    "focused": REGRESSION + "::test_sparse_plan_covers_original_and_translated_native_crop",
    "serialized": "test_serialized_planner_loader.py::test_serialized_plan_matches_dense_for_translated_crop",
    "w035": "test_w035_manifest.py::test_w035_manifest",
}
EXCLUDED_ASSETS = {
    "vesuvius/src/vesuvius/examples/img/slice.png",
    "vesuvius/src/vesuvius/fiber_spline_generator/extend/gap_model.pt",
    *(f"vesuvius/src/vesuvius/neural_tracing/tracers/sheets/docs/figures/{name}.png"
      for name in ("crops_real","crops_xy","dataflow","layout_section","layout_top","tilt","trace_real")),
}

def check_digest(path, expected):
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch: {path}")

def blob(path):
    body = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()

def git(checkout, *args, **kwargs):
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_ALLOW_PROTOCOL="", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", "-c", "core.autocrlf=false", "-C", str(checkout), *args],
                          check=True, env=env, timeout=30, **kwargs)

def stage(args, tree, fixtures):
    pins = json.loads((PACKET/"fixtures/source-pins.json").read_bytes())
    assert pins["source_revision"] == REVISION
    head = git(args.checkout, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
    if head != REVISION:
        raise RuntimeError("Expected the pinned local Villa checkout")
    paths = ["vesuvius/src", *(p for p in pins["files"] if not p.startswith("vesuvius/src/"))]
    tracked = git(args.checkout, "ls-tree", "-r", "-z", REVISION, "--", *paths,
                  capture_output=True).stdout
    staged_bytes = 0
    for entry in tracked.rstrip(b"\0").split(b"\0"):
        identity, filename = entry.split(b"\t", 1)
        mode, kind, expected = identity.split()
        relative = Path(filename.decode())
        if relative.as_posix() in EXCLUDED_ASSETS:
            continue
        if kind != b"blob" or mode not in (b"100644", b"100755") or relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unexpected tracked input: {relative}")
        source = args.checkout/relative
        if source.resolve(strict=True) != source or not source.is_file():
            raise RuntimeError(f"Expected a regular local source file: {source}")
        size = source.stat().st_size
        if staged_bytes + size > 128*1024**2:
            raise RuntimeError("Source staging exceeds 128 MiB")
        body = source.read_bytes()
        if len(body) != size:
            raise RuntimeError(f"Source size changed: {source}")
        def matches(data):
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest().encode() == expected
        if not matches(body):
            body = body.replace(b"\r\n", b"\n")
            if not matches(body):
                raise RuntimeError(f"Working-tree source does not match pinned revision: {relative}")
        staged_bytes += size
        target = tree/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    for name, expected in pins["files"].items():
        if blob(tree/name) != expected:
            raise RuntimeError(f"Source blob mismatch: {name}")
    if args.case != "m1":
        for patch in (("01-level0-planner.diff", "02-focused-regression.diff")
                      if args.minimal else ("00-executed-m2.diff",)):
            git(tree, "apply", "--check", str(PACKET/"patches"/patch))
            git(tree, "apply", str(PACKET/"patches"/patch))
    if args.case in ("serialized", "w035"):
        patch = str(PACKET/"patches/03-bicubic-compatibility.diff")
        git(tree, "apply", "--check", patch)
        git(tree, "apply", patch)
        for name in (PLANNER, GEOMETRY):
            pins["files"][name] = blob(tree/name)
    fixtures.mkdir()
    for name in ("independent_oracle.py", "test_sparse_training_witness.py",
                 "test_serialized_planner_loader.py"):
        (fixtures/name).write_bytes((PACKET/"fixtures"/name).read_bytes())
    text = (PACKET/"fixtures/conftest.py").read_text(encoding="utf-8")
    original = 'if _read_head(repo) != manifest["source_revision"]:'
    assert text.count(original) == 1
    text = text.replace(original, 'if _read_head(Path(os.environ["VESUVIUS_PINNED_CHECKOUT"])) != manifest["source_revision"]:')
    (fixtures/"conftest.py").write_text(text, encoding="utf-8")
    (fixtures/"source-pins.json").write_text(json.dumps(pins, indent=2)+"\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    for name in ("checkout", "scratch"):
        parser.add_argument("--"+name, type=lambda path: Path(path).resolve(strict=True), required=True)
    parser.add_argument("--python", type=lambda path: Path(path).absolute())
    for name in ("codec-overlay", "tqdm-root", "auxiliary-root", "ct-root"):
        parser.add_argument("--"+name, type=lambda path: Path(path).resolve(strict=True))
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--minimal", action="store_true", help="Use the smaller, separately static-checked patch")
    args = parser.parse_args()
    if not args.scratch.is_dir() or any(args.scratch.iterdir()):
        parser.error("--scratch must be an existing, empty directory")
    if args.scratch.is_relative_to(args.checkout):
        parser.error("--scratch must be outside the Villa checkout")
    if not args.stage_only and (args.python is None or not args.python.is_file()):
        parser.error("--python must select an installed, compatible environment")
    if args.case == "w035" and not args.stage_only and not (args.auxiliary_root and args.ct_root):
        parser.error("w035 requires the verified auxiliary and CT local stores")
    if args.case != "w035" and (args.auxiliary_root or args.ct_root):
        parser.error("Scientific input roots are only accepted by the w035 replay")
    for line in (PACKET/"SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        check_digest(PACKET/name, digest)

    tree, fixtures = args.scratch/"villa", args.scratch/"fixtures"
    stage(args,tree,fixtures)
    env = dict(os.environ,VESUVIUS_PINNED_CHECKOUT=str(args.checkout),
               VESUVIUS_REGRESSION_SOURCE_ROOT=str(tree),
               VESUVIUS_M2_CANDIDATE_PLANNER=str(tree/PLANNER),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
    if args.case == "w035":
        w035 = PACKET/"w035"
        for name in ("test_w035_manifest.py","w035_selector.py","w035_bounded_assets.py","w035_ct_compare.py"):
            (fixtures/name).write_bytes((w035/name).read_bytes())
        original = (args.checkout/PLANNER).read_bytes().replace(b"\r\n", b"\n")
        assert original.count(b'interpolation="smooth"') == 1
        baseline = original.replace(b'interpolation="smooth"', b'interpolation="bicubic"')
        assert hashlib.sha256(baseline).hexdigest() == "2c3927bdefeb8c09c8704cf02feb6a5e05382f7d99bbf7f27369aab99d3a407d"
        (fixtures/"baseline_planner.py").write_bytes(baseline)
        env.update(W035_PHASE="compare",W035_PACKAGE=str(w035),W035_ASSETS=str(args.auxiliary_root),
                   W035_CT_ROOT=str(args.ct_root),W035_CT_MANIFEST=str(w035/"reference-manifest.json"),
                   W035_PREFLIGHT_RECEIPT=str(w035/"preflight-witness.json"),
                   W035_BASELINE_PLANNER=str(fixtures/"baseline_planner.py"))
    if args.stage_only:
        print(f"PASS: staged {args.case} with pinned source and no test imports")
        return 0
    paths = [str(path) for path in (args.tqdm_root,args.codec_overlay,tree/"vesuvius/src") if path is not None]
    entry = """import importlib.metadata as md, json, sys
paths, args = json.loads(sys.argv[1]), sys.argv[2:]
sys.path[:0] = paths
assert sys.version_info[:2] == (3,14)
assert [md.version(p) for p in ('pytest','numcodecs','zarr','numpy','torch','vc-reader')] == [
    '9.2.0','0.15.2','2.18.7','2.5.4','2.12.0+cpu','0.1.0']
import pytest
raise SystemExit(pytest.main(args))
"""
    test_root = tree if args.case == "focused" else fixtures
    options = ["--noconftest"] if args.case == "focused" else []
    command = [str(args.python),"-I","-B","-c",entry,json.dumps(paths),
               "-c","/dev/null",f"--rootdir={test_root}",f"--confcutdir={test_root}",*options,
               "--import-mode=prepend","-p","no:cacheprovider","-o","junit_family=xunit1",
               f"--basetemp={args.scratch/'pytest-tmp'}",f"--junitxml={args.scratch/'junit.xml'}",
               "-q",str(test_root/CASES[args.case])]
    return subprocess.run(command,cwd=args.scratch,env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
