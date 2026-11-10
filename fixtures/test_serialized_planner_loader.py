"""Level-0 M2 integration using real, synthetic TIFXYZ files."""
import importlib.util
import json
import os
from pathlib import Path
import sys

from independent_oracle import B0, B1, CHUNKS, PATCH_SHAPE, SHAPE, required_chunks
from test_sparse_training_witness import (
    _assert_equal_labels,
    _chunk_slices,
    _dataset,
    _expected,
    _new_array,
    _positions,
    _sample,
)


def test_serialized_plan_matches_dense_for_translated_crop(runtime, tmp_path, monkeypatch, record_property):
    from vc_reader import tifxyz

    rt = runtime
    name = "_reviewed_m2_serialized_planner"
    path = Path(os.environ["VESUVIUS_M2_CANDIDATE_PLANNER"]).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    planner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, planner)
    spec.loader.exec_module(planner)

    np = rt.np
    paths = {key: tmp_path / f"{key}.zarr" for key in ("dense", "planned", "labels", "supervision")}
    # Add a second Z slab so the same nine planned chunks leave actual holes.
    # Keeping Y and X fixed preserves the frozen M1 oracle's value strides.
    volume_shape = (2 * SHAPE[0], SHAPE[1], SHAPE[2])
    dense = _new_array(rt, paths["dense"], volume_shape, "uint32", CHUNKS)
    sparse = _new_array(rt, paths["planned"], volume_shape, "uint32", CHUNKS)
    dense[:] = np.arange(1, 1 + np.prod(volume_shape), dtype=np.uint32).reshape(volume_shape)
    for key in ("labels", "supervision"):
        surface = _new_array(rt, paths[key], (32, 128, 128), "uint8", (1, 128, 128))
        surface[16, :, :] = 1

    positions = _positions(rt)
    dense_dataset = _dataset(rt, tmp_path, paths, "dense", positions, serialized=True)
    sparse_dataset = _dataset(rt, tmp_path, paths, "planned", positions, serialized=True)
    assert not dense_dataset._tifxyz_cache and not sparse_dataset._tifxyz_cache
    segment = dense_dataset.patches[0].segment

    # Supply the single frozen flat window and constructed segment. The actual
    # segment planner opens the files, resamples XYZ and computes the native box.
    monkeypatch.setattr(planner, "_segment_patch_candidates", lambda *a, **kw: ([(0, 0)], 1))

    def plan_fixture_segment(**kwargs):
        patches, count = planner._build_segment_download_patches(
            segment, patch_filter=kwargs["patch_filter"],
        )
        assert count == len(patches) == 1
        assert tuple(patches[0]["world_bbox"]) == B0
        return patches, count

    monkeypatch.setattr(planner, "_build_dataset_download_patches", plan_fixture_segment)
    plan = planner.build_dataset_chunk_plan(
        datasets_root=tmp_path.parent, output_root=tmp_path,
        source_spec=planner.DatasetSourceSpec(tmp_path.name, str(paths["dense"])),
        patch_size_zyx=PATCH_SHAPE, overlap_fraction=0.25, patch_filter="supervision",
        patch_finding_type="default", patch_min_labeled_coverage=0.0,
        patch_finding_tile_size=None, patch_finding_stride=None,
        patch_finding_filter_empty_tile=False, label_version=None,
    )
    planned = plan.chunk_ids_by_scale[0]
    record_property("planned_chunk_ids", json.dumps(planned))
    assert required_chunks(B0) | required_chunks(B1) <= set(planned)
    assert planned == tuple((0, y, x) for y in range(3) for x in range(3))
    assert not dense_dataset._tifxyz_cache and not sparse_dataset._tifxyz_cache

    for chunk_id in planned:
        region = _chunk_slices(chunk_id)
        sparse[region] = dense[region]
    dense_metadata = json.loads((paths["dense"] / "0/.zarray").read_text())
    assert dense_metadata == json.loads(
        (paths["planned"] / "0/.zarray").read_text()
    )
    unrelated = (1, 1, 1)
    assert unrelated not in planned
    relative_key = dense_metadata.get("dimension_separator", ".").join(map(str, unrelated))
    assert (paths["dense"] / "0" / relative_key).is_file()
    assert not (paths["planned"] / "0" / relative_key).exists()
    assert bool(np.all(dense[_chunk_slices(unrelated)] > 0))
    assert not bool(np.any(sparse[_chunk_slices(unrelated)]))
    record_property("unrelated_dense_only_chunk", json.dumps(unrelated))
    assert not dense_dataset._tifxyz_cache and not sparse_dataset._tifxyz_cache

    for translated, bbox in ((False, B0), (True, B1)):
        dense_sample = _sample(rt, monkeypatch, dense_dataset, translated)
        sparse_sample = _sample(rt, monkeypatch, sparse_dataset, translated)
        _assert_equal_labels(rt, (dense_sample, sparse_sample))
        expected_mask = np.zeros(PATCH_SHAPE, dtype=np.float32)
        for native_y in (191, 192):
            for native_x in (191, 192):
                expected_mask[15 - bbox[0], native_y - bbox[1], native_x - bbox[2]] = 1
        for sample in (dense_sample, sparse_sample):
            np.testing.assert_array_equal(sample["inklabels"][0].numpy(), expected_mask)
            np.testing.assert_array_equal(sample["supervision_mask"][0].numpy(), expected_mask)
        dense_image = dense_sample["image"][0].numpy()
        sparse_image = sparse_sample["image"][0].numpy()
        np.testing.assert_array_equal(dense_image, _expected(rt, bbox))
        mismatches = int(np.count_nonzero(dense_image != sparse_image))
        record_property("translated_mismatches" if translated else "original_mismatches", mismatches)
        np.testing.assert_array_equal(sparse_image, dense_image)

    for dataset in (dense_dataset, sparse_dataset):
        assert len(dataset._tifxyz_cache) == 1
        assert isinstance(next(iter(dataset._tifxyz_cache.values())), tifxyz.Surface)
