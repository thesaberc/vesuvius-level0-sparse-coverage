"""Independent integer-voxel oracle. DRAFT: NOT EXECUTED.
No NumPy, production geometry or production chunk-planning imports.
"""
from itertools import product

SHAPE = (32, 384, 384)
CHUNKS = (32, 128, 128)
PATCH_SHAPE = (32, 128, 128)
B0 = (0, 128, 128, 32, 256, 256)
B1 = (0, 168, 128, 32, 296, 256)
P0 = frozenset({(0, 1, 1)})
RESTORATION = (0, 2, 1)
EXPECTED_MISSING_VOXELS = 163840

def clipped_voxels(box, shape=SHAPE):
    """Enumerate integer members, rather than production interval/chunk ranges."""
    starts = tuple(max(0, box[i]) for i in range(3))
    stops = tuple(min(shape[i], box[i + 3]) for i in range(3))
    yield from product(*(range(a, b) for a, b in zip(starts, stops)))

def chunk_for_voxel(voxel, chunks=CHUNKS):
    return tuple(value // step for value, step in zip(voxel, chunks))

def required_chunks(box, shape=SHAPE, chunks=CHUNKS):
    return frozenset(chunk_for_voxel(v, chunks) for v in clipped_voxels(box, shape))

def source_value(z, y, x):
    return 1 + (z * SHAPE[1] + y) * SHAPE[2] + x

def expected_flat_values(box):
    """No padding for this frozen witness; order matches C-order crop arrays."""
    assert all(0 <= box[i] < box[i + 3] <= SHAPE[i] for i in range(3))
    return (source_value(*voxel) for voxel in clipped_voxels(box))

def missing_flat_mask(box, planned):
    return (chunk_for_voxel(voxel) not in planned for voxel in clipped_voxels(box))

def translated_box(box, delta):
    return tuple(box[i] + delta[i] for i in range(3)) + tuple(
        box[i + 3] + delta[i] for i in range(3)
    )

def proposal_offsets():
    """Preserved 3,969-offset design; not called by the narrow regression."""
    support = (0, *range(-40, -9), *range(10, 41))
    return ((0, dy, dx) for dy, dx in product(support, repeat=2))

def retention_eligible(box, integer_points):
    """Integer points must already reflect validation, validity and truncation."""
    if not integer_points:
        return False  # Production falls back to B0, which must be included separately.
    required = max(1, (len(integer_points) + 2) // 3)
    retained = sum(
        all(box[i] <= point[i] < box[i + 3] for i in range(3))
        for point in integer_points
    )
    return retained >= required

