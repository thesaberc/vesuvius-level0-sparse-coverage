"""Pinned Villa planner and sampler replay for the native w035 release."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

from w035_selector import PATCH, SHAPE, Window, chunk_keys, shift
from w035_ct_compare import Budget, CompleteReference, compare, reads
from w035_bounded_assets import load_assets, tile_intersections

CAP = 128 * 1024**2


class CTIntercept(Exception):
    pass


class OffsetRng:
    def __init__(self, offset):
        y, x = offset
        values = [v for v in offset if v]
        self.axes = (1, 2) if y or not x else (2, 1)
        proposal = [0.25 if len(values) == 2 else 0.75]
        proposal.extend(0.25 if value < 0 else 0.75 for value in values)
        self.randoms = iter([0.0, *(proposal * 24)] if values else [0.99])
        self.magnitudes = iter([abs(value) for value in values] * 24)

    def random(self):
        return next(self.randoms)

    def shuffle(self, axes):
        axes[:] = self.axes

    def randint(self, low, high):
        assert (low, high) == (10, 41)
        return next(self.magnitudes)


class CT:
    shape = SHAPE
    dtype = "uint8"
    chunks = (128, 128, 128)

    def __init__(self):
        self.reference = self.included = self.expected = None

    def __getitem__(self, selection):
        assert all(axis.step is None for axis in selection)
        box = tuple(axis.start for axis in selection) + tuple(axis.stop for axis in selection)
        if self.reference is None:
            raise CTIntercept(box)
        expected = self.expected
        assert box == clipped(expected), "Unadmitted CT array lookup"
        return self.reference.crop(box, reads(box, SHAPE), self.included)


def clipped(box):
    return tuple(max(0, v) for v in box[:3]) + tuple(min(v, n) for v, n in zip(box[3:], SHAPE))


def names(keys):
    return ["/".join(map(str, key)) for key in sorted(keys)]


def load_json(path):
    return json.loads(path.read_bytes())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_knots(surface, window, geometry, np):
    y0, y1, x0, x1 = window
    positions, valid = geometry.read_tifxyz_on_flat_grid(surface, y0=y0, y1=y1, x0=x0, x1=x1)
    knots = []
    for y in range(-(-y0//20)*20, y1, 20):
        for x in range(-(-x0//20)*20, x1, 20):
            if surface.valid[y//20, x//20]:
                expect = surface.xyz[y//20, x//20, ::-1]
                assert valid[y-y0, x-x0] and np.isfinite(expect).all()
                assert (expect >= 0).all() and (expect < SHAPE).all()
                np.testing.assert_array_max_ulp(positions[y-y0, x-x0], expect, maxulp=1)
                knots.append((y, x))
    assert len(knots) >= 3 and np.linalg.matrix_rank(np.asarray(knots[1:])-knots[0]) == 2
    return knots


def plan_window(planner, segment, y0, x0, open_mask, ct, tmp_path, monkeypatch):
    class CTGroup(dict):
        def array_keys(self):
            return ("0",)

    monkeypatch.setattr(planner, "open_volume", open_mask)
    monkeypatch.setattr(planner, "open_volume_root", lambda *a, **kw: CTGroup({"0": ct}))
    monkeypatch.setattr(planner, "_segment_patch_candidates", lambda *a, **kw: ([(y0, x0)], 1))
    native_boxes = []

    def build_frozen(**_):
        rows, count = planner._build_segment_download_patches(segment, patch_filter="supervision")
        assert len(rows) == count == 1 and not rows[0]["is_validation"]
        native_boxes.append(tuple(rows[0]["world_bbox"]))
        return rows, count

    monkeypatch.setattr(planner, "_build_dataset_download_patches", build_frozen)
    plan = planner.build_dataset_chunk_plan(
        source_spec=planner.DatasetSourceSpec(segment.segment_name, str(segment.image_volume)),
        datasets_root=segment.segment_dir.parent, output_root=tmp_path, patch_size_zyx=PATCH,
        overlap_fraction=0.25, patch_filter="supervision", patch_finding_type="default",
        patch_min_labeled_coverage=0.0, patch_finding_tile_size=None,
        patch_finding_stride=None, patch_finding_filter_empty_tile=False, label_version=None)
    assert len(native_boxes) == 1
    return set(plan.chunk_ids_by_scale[0]), native_boxes[0]


def probe(runtime, dataset, patch, window, offset, expected, ct, monkeypatch, record):
    actual = runtime.augmentations.maybe_translate_crop_bbox
    io = runtime.dataset_module.read_bbox_with_padding

    def translate(box, grid, valid, supervision):
        assert tuple(box) == window.native_zyx
        if offset != (0, 0):
            np = runtime.np
            points = grid[valid & np.isfinite(grid).all(axis=-1) & (supervision != 0)].astype(np.int64)
            kept = ((points >= expected[:3]) & (points < expected[3:])).all(axis=-1)
            assert len(points) == window.supervised and int(kept.sum()) == window.retained
        result = actual(box, grid, valid, supervision, rng=OffsetRng(offset))
        assert result == expected, "Pinned sampler rejected or changed the frozen offset"
        record["realized_yx"] = offset
        return result

    def guard(volume, box, **kwargs):
        if volume is ct:
            assert tuple(box) == expected and "read_box_zyx" not in record
            record["read_box_zyx"] = tuple(box)
        return io(volume, box, **kwargs)

    with monkeypatch.context() as sub:
        sub.setattr(runtime.dataset_module, "maybe_translate_crop_bbox", translate)
        sub.setattr(runtime.dataset_module, "read_bbox_with_padding", guard)
        try:
            sample = dataset._native_sample(patch)
        except CTIntercept as exc:
            assert ct.reference is None and exc.args[0] == clipped(expected)
            record["would_read_keys"] = names(chunk_keys(expected))
            return None
    assert ct.reference is not None and sample is not None and "read_box_zyx" in record
    assert not bool(sample["is_unlabeled"]) and sample["supervision_mask"].any()
    record["would_read_keys"] = names(chunk_keys(expected))
    return sample


def compare_tensors(reference, baseline, candidate):
    import torch

    assert set(reference) == set(baseline) == set(candidate)
    assert {"image", "inklabels", "supervision_mask"} <= reference.keys()
    result = {}
    for name, arm in (("baseline", baseline), ("candidate", candidate)):
        result[name] = {key: torch.equal(reference[key], arm[key])
                        for key in ("image", "inklabels", "supervision_mask")}
        assert result[name]["inklabels"] and result[name]["supervision_mask"]
    assert result["candidate"]["image"]
    return result


def test_w035_manifest(runtime, monkeypatch, tmp_path, record_property):
    from vc_reader import tifxyz
    from vesuvius.ink_detection.data import geometry

    approved = Path(os.environ["W035_PACKAGE"])
    assets = Path(os.environ["W035_ASSETS"])
    phase = os.environ.get("W035_PHASE", "preflight")
    assert phase in ("preflight", "compare")
    receipt = {"status": "INCOMPLETE", "phase": phase, "tiles": [], "cases": []}
    if phase == "preflight":
        receipt["ct_chunk_bodies"] = 0
    output = tmp_path / "witness-receipt.json"
    try:
        mesh, arrays = load_assets(approved, assets, runtime.np, receipt["tiles"])
        surface = tifxyz.read(mesh)
        np = runtime.np
        assert surface.xyz.shape == (291, 262, 3) and surface.xyz.dtype == np.float32
        assert surface.xyz.nbytes == 914904
        assert all(abs(scale - 0.05) <= np.spacing(np.float32(0.05)) for scale in surface.scale)
        assert geometry.native_tifxyz_pyramid_params(0)[:2] == (1, 1.0)
        monkeypatch.setattr(tifxyz, "read", lambda path: surface if Path(path) == mesh else
                            (_ for _ in ()).throw(ValueError("Unexpected geometry read")))
        saved = load_json(approved / "w035-frozen-witness.json")
        entry = saved["window"]
        witness = Window(tuple(entry["flat_yx"]), tuple(entry["native_zyx"]), tuple(entry["offset_yx"]),
                         entry["supervised"], entry["retained"])
        assert witness.offset_yx == (-40, 0)
        assert witness.retained >= (witness.supervised + 2)//3
        y0, y1, x0, x1 = witness.flat_yx
        grid, valid = geometry.read_tifxyz_on_flat_grid(surface, y0=y0, y1=y1, x0=x0, x1=x1)
        assert valid[64,64] and np.isfinite(grid[valid]).all()
        assert geometry.compute_native_crop_bbox(grid, valid, PATCH) == witness.native_zyx
        assert tile_intersections(witness.flat_yx) == 1
        mask = arrays["supervision"][0, slice(y0,y1), slice(x0,x1)]
        points = grid[valid & (mask != 0)].astype(np.int64, copy=False)
        translated = shift(witness.native_zyx, witness.offset_yx)
        retained = int(((points >= translated[:3]) & (points < translated[3:])).all(axis=1).sum())
        assert len(points) == witness.supervised and retained == witness.retained
        assert chunk_keys(translated) - chunk_keys(witness.native_zyx)
        receipt["frozen"] = [asdict(witness)]
        receipt["source_receipt_sha256"] = saved["source_receipt_sha256"]
        distinct = {witness.flat_yx: witness}
        ct = CT()
        paths = {name: tmp_path / name for name in ("inklabels", "supervision", "ct")}
        config = runtime.InkDataConfig.from_mapping({
            "mode": "full_3d", "patch_size": list(PATCH), "patch_overlap": 0.25,
            "patch_min_labeled_coverage": 0.0, "image_normalization": "none",
            "augmentation_preset": "none",
            "full_3d": {"label_projection_half_thickness": 0.0,
                        "background_projection_half_thickness": 0.0},
            "normal_pooling": {"support_grid_max_distance": 64.0},
            "dataloader_workers": 1,
            "datasets": [{"segments_path": str(mesh.parent), "volume_path": str(paths["ct"]), "volume_scale": 0}],
        })
        segment = runtime.Segment(
            data_config=config, source=config.datasets[0], dataset_idx=0,
            segment_relpath=mesh.name, segment_dir=mesh, segment_name=mesh.name,
            image_volume=paths["ct"], inklabels=paths["inklabels"],
            supervision_mask=paths["supervision"], validation_mask=None)

        def open_mask(path, resolution, *args, **kwargs):
            name = Path(path).name
            assert resolution == 0 and name in arrays and Path(path) == paths[name]
            return arrays[name]

        def open_sampler(path, resolution, *args, **kwargs):
            return ct if Path(path) == paths["ct"] and resolution == 0 else open_mask(path, resolution)

        patches = {}
        for flat, window in distinct.items():
            check_knots(surface, flat, geometry, np)
            y0, y1, x0, x1 = flat
            patches[flat] = runtime.Patch(segment=segment, bbox=(0, y0, x0, 1, y1, x1))
        dataset = runtime.InkDataset(config, do_augmentations=True, patches=list(patches.values()))
        dataset._tifxyz_cache[str(mesh)] = surface
        monkeypatch.setattr(dataset, "_open", open_sampler)
        cases = [(witness, (0, 0)), (witness, witness.offset_yx)]
        mask_upper = len(receipt["tiles"]) + 4 * len(distinct)
        for window, offset in cases:
            bands = dataset._support_bands(segment, shift(window.native_zyx, offset), required=False)
            assert bands, "No support bands for frozen case"
            mask_upper += (1 + sum(tile_intersections(band[0]) for band in bands)) * 4
        receipt["supervision_tile_upper"] = mask_upper
        if mask_upper > 32:
            receipt["status"] = "SUPERVISION_BUDGET_STOP"
            raise AssertionError("Projected supervision read cap exceeded")
        plans = {}
        for flat, window in distinct.items():
            arms = {}
            for arm, env in (("baseline", "W035_BASELINE_PLANNER"), ("candidate", "VESUVIUS_M2_CANDIDATE_PLANNER")):
                planner = load_module(f"_w035_{arm}", os.environ[env])
                keys, box = plan_window(planner, segment, flat[0], flat[2], open_mask, ct, tmp_path, monkeypatch)
                assert box == window.native_zyx
                expected = box if arm == "baseline" else tuple(
                    a + d for a, d in zip(box, (0, -40, -40, 0, 40, 40)))
                assert keys == chunk_keys(expected)
                arms[arm] = keys
            plans[flat] = arms

        all_keys = set()
        for window, offset in cases:
            record = {"flat_yx": window.flat_yx, "proposal_yx": offset}
            receipt["cases"].append(record)
            arms = plans[window.flat_yx]
            record["plans"] = {name: names(keys) for name, keys in arms.items()}
            box = shift(window.native_zyx, offset)
            probe(runtime, dataset, patches[window.flat_yx], window, offset, box, ct, monkeypatch, record)
            actual = chunk_keys(box)
            assert actual <= arms["candidate"]
            record["baseline_omissions"] = names(actual - arms["baseline"])
            all_keys |= actual | arms["baseline"] | arms["candidate"]
        assert receipt["cases"][-1]["baseline_omissions"]
        receipt["union_keys"] = names(all_keys)
        receipt["logical_ct_bytes"] = sum(
            int(np.prod([min(128, n-k*128) for n,k in zip(SHAPE,key)])) for key in all_keys)
        receipt["worst_ct_staging_bytes"] = len(all_keys)*128**3 + 3257
        receipt["worst_ct_read_bytes"] = len(all_keys)*128**3 + len(cases)*6*32*128**2
        assert 0 < len(all_keys) <= 64 and receipt["logical_ct_bytes"] <= CAP
        assert max(receipt["worst_ct_staging_bytes"], receipt["worst_ct_read_bytes"]) <= CAP

        if phase == "compare":
            manifest = load_json(Path(os.environ["W035_CT_MANIFEST"]))
            prior = load_json(Path(os.environ["W035_PREFLIGHT_RECEIPT"]))
            frozen = json.loads(json.dumps(receipt["frozen"]))
            rows = json.loads(json.dumps(receipt["cases"]))
            assert prior["status"] == "PASS" and prior["phase"] == "preflight" and prior["ct_chunk_bodies"] == 0
            assert prior["frozen"] == frozen and prior["cases"] == rows and prior["union_keys"] == receipt["union_keys"]
            assert set(manifest["objects"]) == set(receipt["union_keys"])
            objects = {tuple(map(int, key.split("/"))): item for key,item in manifest["objects"].items()}
            reference = CompleteReference(Path(os.environ["W035_CT_ROOT"]), SHAPE, objects, 3257, Budget())
            ct.reference = reference
            for (window, offset), record in zip(cases, receipt["cases"]):
                arms = plans[window.flat_yx]
                box = shift(window.native_zyx, offset)
                result = compare(reference, box, arms["baseline"], arms["candidate"])
                assert result["candidate"]["exact_image"]
                samples = {}
                ct.expected = box
                for arm, keys in (("reference", chunk_keys(box)), *arms.items()):
                    ct.included = keys
                    replay = {}
                    samples[arm] = probe(runtime, dataset, patches[window.flat_yx], window,
                                         offset, box, ct, monkeypatch, replay)
                    assert replay["would_read_keys"] == record["would_read_keys"]
                result["tensor_equal"] = compare_tensors(samples["reference"], samples["baseline"], samples["candidate"])
                record["comparison"] = result
            receipt["ct_bytes_charged"] = reference.budget.used
            receipt["real_voxel_witness"] = receipt["cases"][-1]["comparison"]["baseline"]["mismatches"] > 0
        else:
            receipt["ct_chunk_bodies"] = 0
        receipt["status"] = "PASS"
        record_property("key_manifest_status", json.dumps("PASS"))
    finally:
        output.write_text(json.dumps(receipt, indent=2) + "\n")
