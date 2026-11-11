"""Replay one cohort proposal using a shared sampler pass for both CT arms."""
import numpy as np

from w035_selector import PATCH, SHAPE, chunk_keys
from test_w035_manifest import OffsetRng
from w035_ct_compare import compare, reads


class NoSupport(Exception):
    pass


class CohortVolume:
    shape = SHAPE
    dtype = "uint8"
    chunks = (128, 128, 128)

    def __init__(self, reference=None):
        self.reference = reference
        self.expected = None

    def __getitem__(self, selection):
        assert len(selection) == 3 and all(s.step is None for s in selection)
        box = tuple(s.start for s in selection) + tuple(s.stop for s in selection)
        assert self.expected == box, "Unbound CT read"
        if self.reference is None:
            return np.zeros(tuple(b-a for a,b in zip(box[:3], box[3:])), dtype=np.uint8)
        regions = reads(box, self.shape)
        return self.reference.crop(box, regions, set(regions))


def offset_of(original, box):
    assert box[0] == original[0] and box[3] == original[3]
    assert tuple(b-a for a,b in zip(box[:3],box[3:])) == PATCH
    return box[1]-original[1], box[2]-original[2]


def names(keys):
    return sorted("/".join(map(str,key)) for key in keys)


def replay(runtime, dataset, patch, ct, original, proposal, baseline, candidate,
           monkeypatch, expected=None):
    """In comparison mode, share accepted support and run the actual tensor code for each image."""
    dy, dx = proposal
    assert all(v == 0 or 10 <= abs(v) <= 40 for v in (dy,dx))
    assert (ct.reference is None) == (expected is None)
    assert dataset.geometric_augmentations is dataset.photometric_augmentations is dataset.augmentations is None
    assert not dataset.emit_image_for_label
    assert dataset.config.normalization.mode == "none"
    sample_record = {"proposed_yx": list(proposal), "realized_yx": [0,0],
                     "read_keys": [], "read_box_zyx": None, "state": "rejected"}
    native_translate = runtime.augmentations.maybe_translate_crop_bbox
    native_read = runtime.dataset_module.read_bbox_with_padding
    native_support = dataset._support_bands
    native_tensor = dataset._tensor_sample

    def translate(box, grid, valid, supervision):
        assert tuple(box) == original
        result = native_translate(box, grid, valid, supervision, rng=OffsetRng(proposal))
        moved = offset_of(original, result)
        assert moved in (proposal, (0,0))
        sample_record["realized_yx"] = list(moved)
        return result

    def support(segment, box, *, required):
        assert required
        bands = native_support(segment, box, required=False)
        if not bands:
            raise NoSupport
        return bands

    def read(volume, box, **kwargs):
        if volume is not ct:
            return native_read(volume, box, **kwargs)
        assert sample_record["read_box_zyx"] is None and kwargs.get("fill_value") == 0
        assert offset_of(original, box) == tuple(sample_record["realized_yx"])
        key_set = chunk_keys(box)
        assert key_set and key_set <= candidate, "Required CT chunk omitted by the candidate"
        sample_record.update(read_box_zyx=list(box), read_keys=names(key_set))
        if expected is not None:
            for field in ("read_box_zyx", "read_keys", "realized_yx"):
                assert sample_record[field] == expected[field], f"Frozen sampler {field} changed"
        ct.expected = tuple(max(0,a) for a in box[:3]) + tuple(min(b,n) for b,n in zip(box[3:], SHAPE))
        try:
            return native_read(volume, box, **kwargs)
        finally:
            ct.expected = None

    def tensor(current_patch, image, valid_slices, labels, supervision, surface_mask):
        assert current_patch is patch and surface_mask is None
        assert sample_record["read_box_zyx"] is not None
        sample = native_tensor(current_patch, image, valid_slices, labels, supervision, surface_mask)
        if ct.reference is None:
            return sample
        box = tuple(sample_record["read_box_zyx"])
        regions = reads(box, SHAPE)
        raw = compare(ct.reference, box, baseline, candidate)
        tensors = {}
        for arm, keys in (("baseline",baseline), ("candidate",candidate)):
            sparse = ct.reference.crop(box, regions, keys)
            alternative = native_tensor(current_patch, sparse, valid_slices, labels, supervision, surface_mask)
            assert set(alternative) == set(sample)
            tensors[arm] = {name: runtime.torch.equal(sample[name], alternative[name])
                            for name in ("image", "inklabels", "supervision_mask")}
            assert tensors[arm]["image"] == raw[arm]["exact_image"]
        raw["tensor_equal"] = tensors
        sample_record["comparison"] = raw
        return sample

    with monkeypatch.context() as sub:
        sub.setattr(runtime.dataset_module, "maybe_translate_crop_bbox", translate)
        sub.setattr(runtime.dataset_module, "read_bbox_with_padding", read)
        sub.setattr(dataset, "_support_bands", support)
        sub.setattr(dataset, "_tensor_sample", tensor)
        try:
            sample = dataset._native_sample(patch)
        except NoSupport:
            sample = None
            sample_record["rejection"] = "NO_SUPPORT_BANDS"
    if sample is None:
        sample_record.setdefault("rejection", "NATIVE_SAMPLE_RETURNED_NONE")
    else:
        assert sample_record["read_keys"] and not bool(sample["is_unlabeled"])
        realized = tuple(sample_record["realized_yx"])
        sample_record["state"] = ("accepted_control" if proposal == (0,0) else
                                  "fallback" if realized == (0,0) else "accepted_nonzero")
    if expected is not None:
        for field in ("state", "realized_yx", "read_keys", "read_box_zyx"):
            assert sample_record[field] == expected[field], f"Frozen sampler {field} changed"
    return sample_record
