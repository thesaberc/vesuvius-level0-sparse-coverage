"""Current-main 3fbb3bba characterization draft: UNEXECUTED.
A passing first test would reproduce the frozen witness defect and its data restoration;
it would not certify a production correction or current upstream prevalence.
"""
from __future__ import annotations
import json
import shutil
import sys
import time
import pytest
from independent_oracle import (
    B0, B1, CHUNKS, EXPECTED_MISSING_VOXELS, P0, PATCH_SHAPE, RESTORATION,
    SHAPE, expected_flat_values, missing_flat_mask, required_chunks,
)

def _new_array(rt, path, shape, dtype, chunks):
    kwargs = {"mode": "w"}
    if int(rt.zarr.__version__.split(".")[0]) >= 3:
        kwargs["zarr_format"] = 2
    root = rt.zarr.open_group(str(path), **kwargs)
    if int(rt.zarr.__version__.split(".")[0]) >= 3:
        array = root.create_array("0", shape=shape, dtype=dtype, chunks=chunks, fill_value=0)
    else:
        array = root.create_dataset("0", shape=shape, dtype=dtype, chunks=chunks, fill_value=0)
    return array

def _positions(rt):
    np = rt.np
    result = np.empty((128, 128, 3), dtype=np.float32)
    result[..., 0] = 15
    result[..., 1] = 191 + np.arange(128, dtype=np.float64)[:, None] / 127
    result[..., 2] = 191 + np.arange(128, dtype=np.float64)[None, :] / 127
    assert tuple(result[0, 0]) == (15, 191, 191)
    assert tuple(result[-1, -1]) == (15, 192, 192)
    return result

def _chunk_slices(chunk_id):
    return tuple(slice(i * step, (i + 1) * step) for i, step in zip(chunk_id, CHUNKS))

def _assets(rt, base):
    np = rt.np
    paths = {name: base / (name + ".zarr") for name in ("dense", "sparse", "restored", "labels", "supervision")}
    arrays = {
        name: _new_array(rt, paths[name], SHAPE, "uint32", CHUNKS)
        for name in ("dense", "sparse", "restored")
    }
    dense_values = np.arange(1, 1 + SHAPE[0] * SHAPE[1] * SHAPE[2], dtype=np.uint32).reshape(SHAPE)
    arrays["dense"][:] = dense_values
    for name in ("sparse", "restored"):
        for chunk_id in P0:
            window = _chunk_slices(chunk_id)
            arrays[name][window] = dense_values[window]
    restored_window = _chunk_slices(RESTORATION)
    arrays["restored"][restored_window] = dense_values[restored_window]
    for name in ("labels", "supervision"):
        flat = _new_array(rt, paths[name], (32, 128, 128), "uint8", (1, 128, 128))
        flat[16, :, :] = 1
    # Force actual Zarr v2 metadata equality, including codec/filter/fill fields.
    metadata = [
        json.loads((paths[name] / "0" / ".zarray").read_text())
        for name in ("dense", "sparse", "restored")
    ]
    assert metadata[0] == metadata[1] == metadata[2]
    assert metadata[0]["shape"] == list(SHAPE)
    assert metadata[0]["chunks"] == list(CHUNKS)
    assert metadata[0]["fill_value"] == 0
    assert all(json.loads((paths[name] / ".zgroup").read_text()) == {"zarr_format": 2}
               for name in ("dense", "sparse", "restored"))
    for name in ("dense", "sparse", "restored"):
        group = rt.zarr.open_group(str(paths[name]), mode="r")
        assert dict(group.attrs) == {}
        assert dict(group["0"].attrs) == {}
    return paths

def _dataset(rt, base, paths, image_name, positions, serialized):
    segment_dir = base / ("segment-" + image_name)
    segment_dir.mkdir()
    config = rt.InkDataConfig.from_mapping({
        "mode": "full_3d",
        "patch_size": list(PATCH_SHAPE),
        "patch_overlap": 0.5,
        "patch_min_labeled_coverage": 0.0,
        "image_normalization": "none",
        "augmentation_preset": "none",
        "full_3d": {"label_projection_half_thickness": 0.0,
                    "background_projection_half_thickness": 0.0},
        "normal_pooling": {"support_grid_max_distance": 64.0},
        "dataloader_workers": 1,
        "datasets": [{"segments_path": str(base), "volume_path": str(paths[image_name]), "volume_scale": 0}],
    })
    segment = rt.Segment(
        data_config=config, source=config.datasets[0], dataset_idx=0,
        segment_relpath=segment_dir.name, segment_dir=segment_dir,
        segment_name=segment_dir.name, image_volume=paths[image_name],
        inklabels=paths["labels"], supervision_mask=paths["supervision"],
    )
    patch = rt.Patch(segment=segment, bbox=(0, 0, 0, 32, 128, 128))
    dataset = rt.InkDataset(config, do_augmentations=True, patches=[patch])
    if serialized:
        # Current vc_reader Surface/write API is present in reviewed upstream tests.
        # Native binary/codec/file round-trip compatibility is UNVERIFIED until run.
        from vc_reader import tifxyz
        surface = tifxyz.Surface(
            rt.np.ascontiguousarray(positions[..., ::-1], dtype=rt.np.float32),
            scale=(1.0, 1.0), uuid=segment_dir.name,
        )
        tifxyz.write(segment_dir, surface, overwrite=True)
        assert not dataset._tifxyz_cache
        assert all((segment_dir / axis).is_file() for axis in ("x.tif", "y.tif", "z.tif"))
    else:
        dataset._tifxyz_cache[str(segment_dir)] = rt.FakeTifxyz(positions)
    return dataset

class _RngScript:
    def __init__(self, translated):
        self.translated = translated
        self.values = iter((0.0, 0.75, 0.75) if translated else (0.99,))
        self.calls = []

    def random(self):
        self.calls.append("random")
        return next(self.values)  # Unexpected additional draws fail the test.

    def shuffle(self, axes):
        assert self.translated
        assert tuple(axes) == (1, 2)
        axes[:] = (1, 2)  # First selected axis is Y.
        self.calls.append("shuffle")

    def randint(self, low, high):
        assert self.translated and (low, high) == (10, 41)
        self.calls.append("randint")
        return 40

def _sample(rt, monkeypatch, dataset, translated):
    rng = _RngScript(translated)
    with monkeypatch.context() as scope:
        # Inject only the RNG argument; execute the real translation function.
        actual_translate = rt.augmentations.maybe_translate_crop_bbox
        def deterministic_translate(*args, **kwargs):
            assert "rng" not in kwargs
            return actual_translate(*args, rng=rng, **kwargs)
        scope.setattr(rt.dataset_module, "maybe_translate_crop_bbox", deterministic_translate)
        # Actual __getitem__ must return; a rejected one-patch dataset raises.
        sample = dataset[0]
    expected_calls = (
        ["random", "shuffle", "random", "randint", "random"]
        if translated else ["random"]
    )
    assert rng.calls == expected_calls
    assert isinstance(sample, dict)
    assert tuple(sample["image"].shape) == (1, *PATCH_SHAPE)
    assert tuple(sample["inklabels"].shape) == (1, *PATCH_SHAPE)
    assert tuple(sample["supervision_mask"].shape) == (1, *PATCH_SHAPE)
    assert not bool(sample["is_unlabeled"])
    assert "surface_mask" not in sample
    return sample

def _expected(rt, box):
    return rt.np.fromiter(expected_flat_values(box), dtype=rt.np.float32).reshape(PATCH_SHAPE)

def _assert_equal_labels(rt, samples):
    first = samples[0]
    assert int(rt.torch.count_nonzero(first["inklabels"])) == 4
    assert int(rt.torch.count_nonzero(first["supervision_mask"])) == 4
    for other in samples[1:]:
        assert rt.torch.equal(first["inklabels"], other["inklabels"])
        assert rt.torch.equal(first["supervision_mask"], other["supervision_mask"])

def _run_witness(rt, tmp_path, monkeypatch, serialized):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    np = rt.np
    positions = _positions(rt)
    paths = _assets(rt, tmp_path)
    datasets = {
        name: _dataset(rt, tmp_path, paths, name, positions, serialized)
        for name in ("dense", "sparse", "restored")
    }
    assert required_chunks(B0) == P0
    assert required_chunks(B1) == frozenset({(0, 1, 1), (0, 2, 1)})
    assert required_chunks(B1) - P0 == frozenset({RESTORATION})
    # Explicitly exclude legitimate padding; every read voxel is in bounds.
    assert all(0 <= B1[i] < B1[i + 3] <= SHAPE[i] for i in range(3))
    controls = [_sample(rt, monkeypatch, datasets[name], False)
                for name in ("dense", "sparse", "restored")]
    _assert_equal_labels(rt, controls)
    for sample in controls:
        np.testing.assert_array_equal(sample["image"][0].numpy(), _expected(rt, B0))
    samples = [_sample(rt, monkeypatch, datasets[name], True)
               for name in ("dense", "sparse", "restored")]
    _assert_equal_labels(rt, samples)
    dense, sparse, restored = (sample["image"][0].numpy() for sample in samples)
    expected_dense = _expected(rt, B1)
    np.testing.assert_array_equal(dense, expected_dense)
    expected_missing = np.fromiter(missing_flat_mask(B1, P0), dtype=np.bool_).reshape(PATCH_SHAPE)
    actual_missing = dense != sparse
    assert int(expected_missing.sum()) == EXPECTED_MISSING_VOXELS
    np.testing.assert_array_equal(actual_missing, expected_missing)
    # Every missing value was positive in source: these are not genuine zeros.
    assert bool(np.all(dense > 0))
    assert bool(np.all(sparse[expected_missing] == 0))
    np.testing.assert_array_equal(sparse[~expected_missing], dense[~expected_missing])
    np.testing.assert_array_equal(restored, dense)
    expected_projected = np.zeros(PATCH_SHAPE, dtype=np.float32)
    for y in (23, 24):       # 191/192 minus translated crop start 168.
        for x in (63, 64):   # 191/192 minus crop start 128.
            expected_projected[15, y, x] = 1
    for sample in samples:
        np.testing.assert_array_equal(sample["inklabels"][0].numpy(), expected_projected)
        np.testing.assert_array_equal(sample["supervision_mask"][0].numpy(), expected_projected)
    (tmp_path / "witness-receipt.json").write_text(json.dumps({
        "stage": "accepted-crop-sampling-only",
        "overall_pytest_result": "Authoritative pytest/supervisor result required; later assertions may fail",
        "scope": "serialized" if serialized else "cached-existing-fixture",
        "missing_chunk_ids": [[0, 2, 1]],
        "mismatching_image_voxels": int(actual_missing.sum()),
        "no_padding": True, "source_all_positive": True,
        "restored_image_exact": bool(np.array_equal(restored, dense)),
        "fixture_class_source_sha256": rt.fixture_source_sha256,
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "torch": rt.torch.__version__, "zarr": rt.zarr.__version__},
        "source_revision": "3fbb3bbadb84fba3648a0c5c7cdbf7d767f9a3c0",
        "sampling_and_oracle_wall_seconds": time.perf_counter() - wall_start,
        "sampling_and_oracle_process_cpu_seconds": time.process_time() - cpu_start,
        "logical_asset_bytes_before_receipt": sum(
            item.stat().st_size for item in tmp_path.rglob("*") if item.is_file()
        ),
        "peak_memory": "External approved process supervisor must record this",
        "python_fixture_is_not_resource_containment": True,
    }, indent=2))
    return rt, datasets, paths

def test_cached_fixture_accepted_crop_has_missing_chunk_and_is_restored(runtime, tmp_path, monkeypatch):
    _run_witness(runtime, tmp_path, monkeypatch, serialized=False)

def test_serialized_assets_accepted_crop_has_missing_chunk_and_is_restored(
    request, runtime, tmp_path, monkeypatch
):
    if not request.config.getoption("--run-serialized-assets"):
        pytest.skip("Separate serialized-assets execution scope required")
    rt, datasets, paths = _run_witness(runtime, tmp_path, monkeypatch, serialized=True)
    from vc_reader import tifxyz
    from vesuvius.ink_detection.preprocessing import download_required_zarr_chunks as downloader
    for dataset in datasets.values():
        assert len(dataset._tifxyz_cache) == 1
        assert all(isinstance(surface, tifxyz.Surface) for surface in dataset._tifxyz_cache.values())
    # Actual segment planner, with an independent voxel oracle for expected IDs.
    segment = datasets["dense"].patches[0].segment
    patches, candidate_count = downloader._build_segment_download_patches(
        segment, patch_filter="supervision"
    )
    assert candidate_count == len(patches) == 4
    assert {tuple(row["world_bbox"]) for row in patches} == {B0}
    assert {tuple(row["flat_bbox"][i] for i in (1, 2)) for row in patches} == {
        (0, 0), (0, 64), (64, 0), (64, 64)
    }
    expected_ids = frozenset().union(
        *(required_chunks(tuple(row["world_bbox"])) for row in patches)
    )
    production_ids = frozenset(downloader.collect_unique_chunk_ids(
        patches, chunk_shape_zyx=CHUNKS, array_shape_zyx=SHAPE
    ))
    assert production_ids == expected_ids == P0
    (tmp_path / "serialized-planner-receipt.json").write_text(json.dumps({
        "stage": "serialized-segment-planner-assertions-completed",
        "overall_pytest_result": "Authoritative pytest/supervisor result required",
        "candidate_count": candidate_count,
        "requested_level_chunk_ids": [list(value) for value in sorted(production_ids)],
        "full_cli_or_all_pyramid_copy_tested": False,
    }, indent=2))


@pytest.fixture(autouse=True)
def _remove_owned_synthetic_assets(tmp_path):
    """Remove only this test directory's explicitly named synthetic assets.

    Retain JSON receipts for the supervisor. This is cleanup, not a disk quota.
    """
    yield
    for name in ("dense.zarr", "sparse.zarr", "restored.zarr", "labels.zarr",
                 "supervision.zarr", "segment-dense", "segment-sparse", "segment-restored"):
        target = tmp_path / name
        assert target.parent.resolve() == tmp_path.resolve()
        if target.is_symlink():
            raise AssertionError("Unexpected synthetic asset symlink; preserve for review")
        if target.exists():
            shutil.rmtree(target)
