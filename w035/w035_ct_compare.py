"""Offline raw Zarr-v2 crop comparison using one complete, admitted local reference."""
from dataclasses import dataclass
from hashlib import file_digest
from itertools import product
from math import prod
from pathlib import Path
import stat

import numpy as np

CHUNK = 128
BODY_BYTES = CHUNK**3
BYTE_CAP = 128 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key_name(key):
    return "/".join(map(str, key))


def reads(box, shape):
    lo = tuple(max(0, v) for v in box[:3])
    hi = tuple(min(v, n) for v, n in zip(box[3:], shape))
    if any(a >= b for a, b in zip(lo, hi)):
        return {}
    spans = (range(a//CHUNK, (b-1)//CHUNK+1) for a, b in zip(lo, hi))
    regions = {}
    for key in product(*spans):
        origin = tuple(i*CHUNK for i in key)
        a = tuple(max(i, j) for i, j in zip(lo, origin))
        b = tuple(min(i, j+CHUNK) for i, j in zip(hi, origin))
        local = tuple(slice(i-j, k-j) for i, j, k in zip(a, origin, b))
        output = tuple(slice(i-j, k-j) for i, j, k in zip(a, box[:3], b))
        regions[key] = local, output
    return regions


@dataclass
class Budget:
    used: int = 0

    def charge(self, count):
        require(self.used + count <= BYTE_CAP, "Cumulative CT byte cap exceeded")
        self.used += count


class CompleteReference:
    def __init__(self, root, shape, objects, metadata_bytes, budget):
        self.shape = tuple(shape)
        require(len(self.shape) == 3 and all(type(n) is int and n > 0 for n in self.shape),
                "Invalid source shape")
        require(type(metadata_bytes) is int and 0 < metadata_bytes <= BYTE_CAP, "Invalid metadata bytes")
        require(len(objects) <= 64, "Too many admitted CT keys")
        require(sum(row["bytes"] for row in objects.values()) + metadata_bytes <= BYTE_CAP,
                "Staged CT bytes exceed limit")
        self.paths, self.stored, self.logical = {}, {}, {}
        self.budget = budget
        root = Path(root).resolve(strict=True)
        for key, row in sorted(objects.items()):
            require(len(key) == 3 and all(type(i) is int and 0 <= i*CHUNK < n
                                          for i, n in zip(key, self.shape)), "Invalid admitted chunk key")
            self.logical[key] = prod(min(CHUNK, n-i*CHUNK) for i, n in zip(key, self.shape))
            if row["disposition"] == "INTENTIONAL_FILL":
                require(row["bytes"] == 0 and bool(row.get("evidence")), "Unproven fill")
                self.paths[key] = None
                self.stored[key] = 0
                continue
            require(row["disposition"] == "PRESENT" and row["bytes"] == BODY_BYTES, "Unadmitted CT body")
            path = root / key_name(key)
            require(path.resolve(strict=True) == path, "Symlink in staged CT object path")
            with path.open("rb") as stream:
                size = path.stat().st_size
                require(stat.S_ISREG(path.stat().st_mode) and size == BODY_BYTES, "Unexpected CT object")
                budget.charge(size)
                require(file_digest(stream, "sha256").hexdigest() == row["sha256"], "CT object hash mismatch")
            self.paths[key] = path
            self.stored[key] = BODY_BYTES
        require(sum(self.logical.values()) <= BYTE_CAP, "Logical CT union exceeds cap")

    def crop(self, box, regions, included):
        require(set(regions) <= self.paths.keys(), "Incomplete local reference")
        output = np.zeros(tuple(b-a for a, b in zip(box[:3], box[3:])), dtype=np.uint8)
        for key, (local, dest) in regions.items():
            if key not in included or self.paths[key] is None:
                continue
            self.budget.charge(prod(part.stop-part.start for part in local))
            mapped = np.memmap(self.paths[key], dtype=np.uint8, mode="r", shape=(CHUNK,)*3)
            output[dest] = mapped[local]
            del mapped
        return output


def compare(reference, box, baseline, candidate):
    require(len(box) == 6 and all(type(v) is int for v in box)
            and tuple(b-a for a,b in zip(box[:3], box[3:])) == (32,128,128), "Invalid read box")
    regions = reads(box, reference.shape)
    required = set(regions)
    require(required and baseline <= reference.paths.keys() and candidate <= reference.paths.keys(),
            "Unadmitted comparison keys")
    dense = reference.crop(box, regions, required)
    original = reference.crop(box, regions, baseline)
    patched = reference.crop(box, regions, candidate)
    results = {}
    for name, arm, sparse in (("baseline", baseline, original), ("candidate", candidate, patched)):
        difference = dense != sparse
        by_key = {key_name(key): int(np.count_nonzero(difference[dest]))
                  for key, (_, dest) in regions.items() if difference[dest].any()}
        count = int(np.count_nonzero(difference))
        require(count == sum(by_key.values()), "Unexpected padding mismatch")
        results[name] = {
            "omitted_keys": [key_name(key) for key in sorted(required-arm)],
            "mismatches": count, "mismatches_by_key": by_key, "exact_image": count == 0,
            "planned_chunks": len(arm),
            "planned_logical_bytes": sum(reference.logical[key] for key in arm),
            "planned_stored_bytes": sum(reference.stored[key] for key in arm),
        }
    added = candidate - baseline
    results.update(added_keys=[key_name(key) for key in sorted(added)],
                   added_logical_bytes=sum(reference.logical[key] for key in added),
                   added_stored_bytes=sum(reference.stored[key] for key in added),
                   required_chunks=len(required),
                   required_logical_bytes=sum(reference.logical[key] for key in required),
                   padding_voxels=dense.size-sum(prod(s.stop-s.start for s in dest)
                                                for _, dest in regions.values()),
                   ct_bytes_charged=reference.budget.used)
    return results
