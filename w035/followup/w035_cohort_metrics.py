"""Compute window and cohort metrics from fixed-cohort sampler receipts."""
from math import prod

SHAPE = (20974, 6621, 6621)
EDGE = 128
BODY = EDGE**3


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def logical_bytes(keys):
    total = 0
    for key in keys:
        zyx = tuple(map(int, key.split("/")))
        require(len(zyx) == 3 and all(0 <= i*EDGE < n for i,n in zip(zyx, SHAPE)),
                f"Unexpected CT key {key}")
        total += prod(min(EDGE,n-i*EDGE) for i,n in zip(zyx, SHAPE))
    return total


def cost(baseline, candidate, required, stored=None):
    require(baseline <= candidate, "Candidate dropped a baseline key")
    require(required <= candidate, "Patched planner omitted an actual read")
    added = candidate - baseline
    subsets = {"baseline": baseline, "candidate": candidate, "required": required, "added": added,
               "baseline_missed": required-baseline, "candidate_extra": candidate-required}
    result = {}
    for name, keys in subsets.items():
        result[name] = {"keys": sorted(keys), "chunks": len(keys), "logical_bytes": logical_bytes(keys),
                        "stored_bytes": None if stored is None else sum(stored[key] for key in keys)}
    for field in ("chunks", "logical_bytes", "stored_bytes"):
        n, b, r = (result[name][field] for name in ("candidate", "baseline", "required"))
        result[f"candidate_to_baseline_{field}"] = n/b if b else None
        result[f"candidate_to_required_{field}"] = n/r if r else None
    return result


def summarize(spec, cases, plans, stored=None):
    window_ids = [row["id"] for row in spec["windows"]]
    expected = [(row["id"],tuple(offset)) for row in spec["windows"] for offset in row["offsets_yx"]]
    require([(case["window_id"], tuple(case["proposed_yx"])) for case in cases] == expected,
            "Cases do not match the full frozen schedule")
    require(set(plans) == set(window_ids), "Planner windows differ from cohort")
    counts = {"accepted_controls": 0, "accepted_nonzero": 0, "fallbacks": 0, "rejections": 0,
              "accepted_nonzero_with_baseline_omission": 0, "accepted_nonzero_with_baseline_voxel_difference": 0}
    baseline_all, candidate_all, required_all = set(), set(), set()
    windows = []
    for window in window_ids:
        baseline, candidate = (set(plans[window][name]) for name in ("baseline", "candidate"))
        required = set()
        window_cases = []
        for case in (row for row in cases if row["window_id"] == window):
            state = case["state"]
            read = set(case["read_keys"])
            proposed, realized = tuple(case["proposed_yx"]), tuple(case["realized_yx"])
            require(realized in (proposed,(0,0)), "Unexpected realized offset")
            require(state in ("accepted_control", "accepted_nonzero", "fallback", "rejected"),
                    "Unexpected case status")
            required.update(read)
            missing = read - baseline
            comparison = case.get("comparison")
            if state == "accepted_nonzero":
                require(read and proposed == realized != (0,0), "Invalid accepted translation")
                counts["accepted_nonzero"] += 1
                counts["accepted_nonzero_with_baseline_omission"] += bool(missing)
                if comparison:
                    counts["accepted_nonzero_with_baseline_voxel_difference"] += bool(
                        comparison["baseline"]["mismatches"])
            elif state == "accepted_control":
                require(read and proposed == realized == (0,0), "Invalid accepted control")
                counts["accepted_controls"] += 1
            elif state == "fallback":
                require(proposed != (0,0) and realized == (0,0), "Invalid fallback")
                counts["fallbacks"] += 1
            else:
                counts["rejections"] += 1
            if stored is not None and state != "rejected":
                require(comparison is not None, "Missing accepted comparison")
                require(comparison["candidate"]["mismatches"] == 0 and all(
                    comparison["tensor_equal"]["candidate"].values()), "Patched sample differs")
                require(comparison["tensor_equal"]["baseline"]["inklabels"]
                        and comparison["tensor_equal"]["baseline"]["supervision_mask"], "Target tensors differ")
                require(not comparison["baseline"]["mismatches"] or missing,
                        "Baseline changed without a planner omission")
            window_cases.append({**case, "baseline_omissions": sorted(missing)})
        windows.append({"window_id": window, "cost": cost(baseline,candidate,required,stored),
                        "cases": window_cases})
        baseline_all |= baseline
        candidate_all |= candidate
        required_all |= required
    union = baseline_all | candidate_all | required_all
    require(len(union) <= 64 and logical_bytes(union) <= 128*1024**2, "CT union exceeds cap")
    if stored is not None:
        require(set(stored) == union, "Stored reference differs from admitted union")
    denominator = counts["accepted_nonzero"]
    counts["baseline_omission_fraction"] = (counts["accepted_nonzero_with_baseline_omission"] / denominator
                                             if denominator else None)
    if stored is None:
        counts["accepted_nonzero_with_baseline_voxel_difference"] = None
    return {"counts": counts, "windows": windows,
            "deduplicated_cost": cost(baseline_all,candidate_all,required_all,stored), "union_keys": sorted(union)}
