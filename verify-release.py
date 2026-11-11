"""Check the exact public-candidate allowlist without reading data inputs."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ALLOW = {
    "LICENSE", "README.md", "REPRODUCE.md", "reproduce.py", "verify-release.py",
    "patches/00-executed-m2.diff", "patches/01-level0-planner.diff",
    "patches/02-focused-regression.diff", "patches/03-bicubic-compatibility.diff",
    "fixtures/conftest.py", "fixtures/independent_oracle.py", "fixtures/source-pins.json",
    "fixtures/test_serialized_planner_loader.py", "fixtures/test_sparse_training_witness.py",
    "evidence/results.json",
    "evidence/followup.json",
    "w035/followup/preflight-witness.json",
    "w035/followup/reference-manifest.json",
    "w035/followup/test_w035_cohort.py",
    "w035/followup/test_w035_manifest.py",
    "w035/followup/w035-followup-spec.json",
    "w035/followup/w035-stage1-assets-draft.json",
    "w035/followup/w035_bounded_assets.py",
    "w035/followup/w035_cohort_metrics.py",
    "w035/followup/w035_cohort_replay.py",
    "w035/followup/w035_ct_compare.py",
    "w035/followup/w035_followup_freeze.py",
    "w035/followup/w035_selector.py",
    "w035/alternative-native-release-metadata.json", "w035/preflight-witness.json",
    "w035/reference-manifest.json", "w035/test_w035_manifest.py", "w035/w035-frozen-witness.json",
    "w035/w035-stage1-assets-draft.json", "w035/w035_bounded_assets.py", "w035/w035_ct_compare.py",
    "w035/w035_selector.py",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((ROOT/"SHARE-MANIFEST.json").read_bytes())
    records = manifest["files"]
    assert len(records) == len(ALLOW) and {row["path"] for row in records} == ALLOW
    assert {f.relative_to(ROOT).as_posix() for f in ROOT.rglob("*") if f.is_file()} == (
        ALLOW | {"SHARE-MANIFEST.json","SHA256SUMS"})
    for row in records:
        path = ROOT/row["path"]
        assert path.resolve() == path and path.stat().st_size == row["bytes"]
        assert digest(path) == row["sha256"]
        body = path.read_text(encoding="utf-8")
        if path.name != "verify-release.py":
            assert not re.search(r"(?i)/home/|/mnt/c/|[A-Z]:\\Users|saberc|BRIAN-M3|\.codex|@gmail",body), path
        if path.suffix == ".py":
            ast.parse(body,filename=str(path))
        elif path.suffix == ".md":
            for link in re.findall(r"\]\(([^ )]+)\)",body):
                if not link.startswith("https://"):
                    assert (path.parent/link).is_file(), link
    checksums = {}
    for line in (ROOT/"SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected,name = line.split("  ",1)
        assert name not in checksums and name in ALLOW | {"SHARE-MANIFEST.json"}
        assert digest(ROOT/name) == expected
        checksums[name] = expected
    assert set(checksums) == ALLOW | {"SHARE-MANIFEST.json"}
    ct = json.loads((ROOT/"w035/reference-manifest.json").read_bytes())["objects"]
    results = json.loads((ROOT/"evidence/results.json").read_bytes())
    zero,moved = results["w035_targeted"]["cases"]
    assert zero["comparison"]["baseline"]["mismatches"] == 0
    assert moved["comparison"]["baseline"]["mismatches"] == 4096
    assert moved["comparison"]["candidate"]["mismatches"] == 0
    assert all(moved["comparison"]["tensor_equal"]["candidate"].values())
    assert moved["comparison"]["added_stored_bytes"] == 4*1024**2
    assert len(ct) == 6 and sum(row["bytes"] for row in ct.values()) == 12*1024**2
    cohort = results["w035_cohort"]
    assert cohort["counts"]["baseline_omission_fraction"] is None
    assert cohort["counts"]["fallbacks"] == 6 and cohort["ct_bodies"] == 0
    assert cohort["deduplicated_cost"]["added"]["logical_bytes"] == 16*1024**2
    follow = json.loads((ROOT/"evidence/followup.json").read_bytes())
    windows = follow["summary"]["windows"]
    assert len(windows) == 2
    for window, count in zip(windows, (4095, 4088), strict=True):
        control, shifted = window["cases"]
        assert control["comparison"]["baseline"]["mismatches"] == 0
        assert shifted["comparison"]["baseline"]["mismatches"] == count
        for case in (control, shifted):
            assert case["comparison"]["candidate"]["mismatches"] == 0
            assert all(case["comparison"]["tensor_equal"]["candidate"].values())
    assert follow["summary"]["deduplicated_cost"]["added"]["stored_bytes"] == 8*1024**2
    print("PASS: exact public allowlist, checksums, links, syntax, redaction and evidence totals")


if __name__ == "__main__":
    verify()
