"""Fixed three-window cohort on the pinned w035 level-0 Villa paths."""
import json
import os
from pathlib import Path

from test_w035_manifest import check_knots, load_module, plan_window
from w035_bounded_assets import load_assets, tile_intersections
from w035_cohort_metrics import BODY, logical_bytes, summarize
from w035_cohort_replay import CohortVolume, names, replay
from w035_ct_compare import Budget, CompleteReference
from w035_selector import PATCH, SHAPE, chunk_keys
from w035_followup_freeze import freeze

CAP = 128*1024**2
METADATA_BYTES = 3257


def load(path):
    return json.loads(path.read_bytes())


def case_id(case):
    return case["window_id"], tuple(case["proposed_yx"])


def eligible(geometry, surface, window, mask, np):
    y0, y1, x0, x1 = window
    assert (y1-y0,x1-x0) == PATCH[1:] and tile_intersections(window) == 1
    grid, valid = geometry.read_tifxyz_on_flat_grid(surface, y0=y0, y1=y1, x0=x0, x1=x1)
    if not valid[64,64] or not np.isfinite(grid[valid]).all():
        return None
    if not (valid & (mask[0,slice(y0,y1),slice(x0,x1)] != 0)).any():
        return None
    original = geometry.compute_native_crop_bbox(grid,valid,PATCH)
    if not chunk_keys(original):
        return None
    check_knots(surface, window, geometry, np)
    return original


def test_w035_cohort(runtime, monkeypatch, tmp_path, record_property):
    from vc_reader import tifxyz
    from vesuvius.ink_detection.data import geometry

    approved = Path(os.environ["W035_PACKAGE"])
    phase = os.environ.get("W035_PHASE", "preflight")
    assert phase in ("preflight","compare")
    spec = load(approved/"w035-followup-spec.json")
    assert [len(t["windows"]) for t in spec["candidate_tiles"]] == [16,16]
    receipt = {"status":"INCOMPLETE", "phase":phase, "tiles":[], "cases":[], "plans":{}}
    if phase == "preflight":
        receipt["ct_chunk_bodies"] = 0
    output = tmp_path/"witness-receipt.json"
    try:
        mesh, arrays = load_assets(approved, Path(os.environ["W035_ASSETS"]), runtime.np, receipt["tiles"])
        surface = tifxyz.read(mesh)
        np = runtime.np
        assert surface.xyz.shape == (291,262,3) and surface.xyz.dtype == np.float32
        assert surface.xyz.nbytes == 914904
        assert all(abs(scale-0.05) <= np.spacing(np.float32(0.05)) for scale in surface.scale)
        assert geometry.native_tifxyz_pyramid_params(0)[:2] == (1,1.0)
        monkeypatch.setattr(tifxyz,"read",lambda path: surface if Path(path) == mesh else
                            (_ for _ in ()).throw(ValueError("Unexpected geometry path")))

        frozen = freeze(spec,geometry,surface,arrays["supervision"],np,receipt)
        receipt["schedule"] = frozen
        spec["windows"] = frozen
        if len(frozen) != 2:
            assert phase == "preflight"
            receipt["status"] = "INSUFFICIENT_FROZEN_CASES"
            record_property("key_manifest_status", json.dumps(receipt["status"]))
            return

        prior = reference = None
        previous = {}
        stored = None
        if phase == "compare":
            prior = load(Path(os.environ["W035_PREFLIGHT_RECEIPT"]))
            assert prior["status"] == "PASS" and prior["phase"] == "preflight" and prior["ct_chunk_bodies"] == 0
            assert prior["schedule"] == receipt["schedule"]
            previous = {case_id(case):case for case in prior["cases"]}
            assert len(previous) == 4
            manifest = load(Path(os.environ["W035_CT_MANIFEST"]))
            assert set(manifest["objects"]) == set(prior["union_keys"])
            objects = {tuple(map(int,key.split("/"))):row for key,row in manifest["objects"].items()}
            reference = CompleteReference(Path(os.environ["W035_CT_ROOT"]), SHAPE, objects, METADATA_BYTES, Budget())
            stored = {key:row["bytes"] for key,row in manifest["objects"].items()}
        ct = CohortVolume(reference)
        paths = {name:tmp_path/name for name in ("inklabels","supervision","ct")}
        config = runtime.InkDataConfig.from_mapping({
            "mode":"full_3d", "patch_size":list(PATCH), "patch_overlap":0.25,
            "patch_min_labeled_coverage":0.0, "image_normalization":"none", "augmentation_preset":"none",
            "full_3d":{"label_projection_half_thickness":0.0,"background_projection_half_thickness":0.0},
            "normal_pooling":{"support_grid_max_distance":64.0}, "dataloader_workers":1,
            "datasets":[{"segments_path":str(mesh.parent),"volume_path":str(paths["ct"]),"volume_scale":0}],
        })
        segment = runtime.Segment(data_config=config, source=config.datasets[0], dataset_idx=0,
            segment_relpath=mesh.name, segment_dir=mesh, segment_name=mesh.name, image_volume=paths["ct"],
            inklabels=paths["inklabels"], supervision_mask=paths["supervision"], validation_mask=None)

        def open_mask(path, resolution, *args, **kwargs):
            name = Path(path).name
            assert resolution == 0 and name in arrays and Path(path) == paths[name]
            return arrays[name]

        def open_sampler(path, resolution, *args, **kwargs):
            return ct if Path(path) == paths["ct"] and resolution == 0 else open_mask(path,resolution)

        patches = []
        for window in spec["windows"]:
            y0,y1,x0,x1 = window["flat_yx"]
            patches.append(runtime.Patch(segment=segment, bbox=(0,y0,x0,1,y1,x1)))
        dataset = runtime.InkDataset(config, do_augmentations=True, patches=patches)
        dataset._tifxyz_cache[str(mesh)] = surface
        monkeypatch.setattr(dataset,"_open",open_sampler)
        planners = {}
        if phase == "preflight":
            for arm,env in (("baseline","W035_BASELINE_PLANNER"),("candidate","VESUVIUS_M2_CANDIDATE_PLANNER")):
                planners[arm] = load_module(f"_w035_cohort_{arm}",os.environ[env])
        for window,patch in zip(spec["windows"],patches):
            identifier = window["id"]
            flat = tuple(window["flat_yx"])
            original = eligible(geometry,surface,flat,arrays["supervision"],np)
            arms = {name:set() for name in ("baseline","candidate")}
            assert original == tuple(window["native_zyx"]), "Frozen geometry changed"
            if original is not None:
                envelope = tuple(a+d for a,d in zip(original,(0,-40,-40,0,40,40)))
                for arm in arms:
                    expected_keys = chunk_keys(original if arm == "baseline" else envelope)
                    if prior is None:
                        with monkeypatch.context() as sub:
                            keys, box = plan_window(planners[arm],segment,flat[0],flat[2],open_mask,ct,tmp_path,sub)
                        assert box == original
                    else:
                        keys = {tuple(map(int,key.split("/"))) for key in prior["plans"][identifier][arm]}
                    assert keys == expected_keys
                    arms[arm] = keys
            receipt["plans"][identifier] = {arm:names(keys) for arm,keys in arms.items()}
            if prior is not None:
                assert receipt["plans"][identifier] == prior["plans"][identifier]
            for proposed in window["offsets_yx"]:
                expected = previous.get((identifier,tuple(proposed)))
                if original is None:
                    case = {"state":"rejected","rejection":"INELIGIBLE_WINDOW","proposed_yx":proposed,
                            "realized_yx":[0,0],"read_keys":[],"read_box_zyx":None}
                    if expected is not None:
                        assert expected == {"window_id":identifier,**case}
                else:
                    case = replay(runtime,dataset,patch,ct,original,tuple(proposed),
                                  arms["baseline"],arms["candidate"],monkeypatch,expected)
                receipt["cases"].append({"window_id":identifier,**case})
        summary = summarize(spec,receipt["cases"],receipt["plans"],stored)
        receipt.update(summary=summary,union_keys=summary["union_keys"])
        unique = len(receipt["union_keys"])
        receipt["logical_ct_bytes"] = logical_bytes(receipt["union_keys"])
        receipt["worst_ct_staging_bytes"] = unique*BODY + METADATA_BYTES
        ct_reads = sum(row["read_box_zyx"] is not None for row in receipt["cases"])
        receipt["worst_ct_read_bytes"] = unique*BODY + ct_reads*6*32*128**2
        assert unique <= 36 and max(receipt["worst_ct_staging_bytes"],receipt["worst_ct_read_bytes"]) <= CAP
        if reference is not None:
            receipt["ct_bytes_charged"] = reference.budget.used
            assert receipt["union_keys"] == prior["union_keys"]
        counts = summary["counts"]
        qualifies = counts["accepted_controls"] == counts["accepted_nonzero"] == 2
        for row in summary["windows"]:
            control,moved = row["cases"]
            frozen_case = next(w for w in frozen if w["id"] == row["window_id"])
            qualifies &= (not control["baseline_omissions"] and
                          moved["baseline_omissions"] == frozen_case["predicted_omissions"])
            if phase == "compare":
                assert qualifies
                assert control["comparison"]["baseline"]["exact_image"]
                assert moved["comparison"]["baseline"]["mismatches"] > 0
        receipt["status"] = "PASS" if qualifies else "INSUFFICIENT_SAMPLER_WITNESSES"
        record_property("key_manifest_status", json.dumps(receipt["status"]))
    finally:
        output.write_text(json.dumps(receipt,indent=2) + "\n")
