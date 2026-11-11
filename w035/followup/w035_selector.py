"""Freeze two w035 windows at most before either planner is called."""
from collections import Counter
from dataclasses import dataclass
import itertools

import numpy as np

CANVAS = (5820, 5240)
SHAPE = (20974, 6621, 6621)
PATCH = (32, 128, 128)
TILE = 512
FILL = (2**64 - 1,) * 2
DELTAS = (0, *range(-40, -9), *range(10, 41))
OFFSETS = tuple(sorted(
    (pair for pair in itertools.product(DELTAS, repeat=2) if pair != (0, 0)),
    key=lambda pair: (max(map(abs, pair)), sum(map(abs, pair)), *pair),
))


@dataclass(frozen=True)
class Window:
    flat_yx: tuple
    native_zyx: tuple
    offset_yx: tuple
    supervised: int
    retained: int


def shift(box, offset):
    dy, dx = offset
    return tuple(value + delta for value, delta in zip(box, (0, dy, dx, 0, dy, dx)))


def chunk_keys(box):
    bounds = tuple((max(0, lo), min(hi, size)) for lo, hi, size in zip(box[:3], box[3:], SHAPE))
    if any(lo >= hi for lo, hi in bounds):
        return set()
    return set(itertools.product(*(range(lo // 128, (hi - 1) // 128 + 1) for lo, hi in bounds)))


def first_adding(box, points, *, exclude=frozenset()):
    original = chunk_keys(box)
    if not len(points) or not original:
        return None, "NO_SUPERVISED_IN_VOLUME_CROP"
    ykeys = {d: chunk_keys(shift(box, (d, 0))) for d in DELTAS}
    xkeys = {d: chunk_keys(shift(box, (0, d))) for d in DELTAS}
    adds_y = {d for d in DELTAS if ykeys[d] - original}
    adds_x = {d for d in DELTAS if xkeys[d] - original}
    candidates = [(dy, dx) for dy, dx in OFFSETS
                  if ykeys[dy] and xkeys[dx] and (dy in adds_y or dx in adds_x)]
    if not candidates:
        return None, "NO_CHUNK_CROSSING"
    z0, y0, x0, z1, y1, x1 = box
    anchor = (y0 - 40, x0 - 40)
    valid = ((points >= (z0, *anchor)) & (points < (z1, y1+40, x1+40))).all(axis=1)
    ys, xs = (points[valid, 1:] - anchor).T
    prefix = np.zeros((209, 209), dtype=np.int64)
    np.add.at(prefix, (ys + 1, xs + 1), 1)
    prefix = prefix.cumsum(0).cumsum(1)
    ay, ax = (np.asarray(candidates) + 40).T
    by, bx = ay + 128, ax + 128
    count = prefix[by,bx] - prefix[ay,bx] - prefix[by,ax] + prefix[ay,ax]
    for index in np.flatnonzero(count >= max(1, int(np.ceil(len(points)*(1.0/3.0))))):
        offset = candidates[index]
        added = chunk_keys(shift(box, offset)) - original
        if not added or added & exclude:
            continue
        return (offset, int(count[index])), "QUALIFIED"
    return None, "RETENTION_REJECTED"


def select(surface, supervision, geometry, receipt):
    """Freeze the first zero control and first adding/retained move, including nulls."""
    valid = (surface.valid & np.isfinite(surface.xyz).all(axis=-1)
             & (surface.xyz >= 0).all(axis=-1) & (surface.xyz < SHAPE[::-1]).all(axis=-1))
    ys, xs = np.nonzero(valid)
    by_tile = {}
    for y, x in zip(ys * 20, xs * 20):
        if (64 <= y % TILE <= 448 and 64 <= x % TILE <= 448
                and y + 64 <= CANVAS[0] and x + 64 <= CANVAS[1]):
            tile = (int(y // TILE), int(x // TILE))
            if supervision.entries[tile[0] * 11 + tile[1]] != FILL:
                by_tile.setdefault(tile, []).append((int(y), int(x)))
    tiles = sorted(sorted(by_tile, key=lambda tile: (-len(by_tile[tile]), tile))[:4])
    receipt["search"] = []
    control = None
    witness = None
    for ty, tx in tiles:
        tile = supervision.tile(ty, tx)
        centers = [(y, x) for y, x in by_tile[ty, tx] if tile[y % TILE, x % TILE]]
        counts = Counter()
        row = {"tile_yx": (ty, tx), "supervised_centers": len(centers), "windows": [], "outcomes": counts}
        receipt["search"].append(row)
        for index in np.linspace(0, len(centers)-1, min(16, len(centers)), dtype=int):
            y, x = centers[index]
            window = (y-64, y+64, x-64, x+64)
            y0, y1, x0, x1 = window
            grid, valid_points = geometry.read_tifxyz_on_flat_grid(
                surface, y0=y0, y1=y1, x0=x0, x1=x1)
            row["windows"].append(window)
            if not valid_points[64,64] or not np.isfinite(grid[valid_points]).all():
                counts["INVALID_GEOMETRY"] += 1
                continue
            dy, dx = y % TILE, x % TILE
            mask = tile[dy-64:dy+64, dx-64:dx+64]
            points = grid[valid_points & (mask != 0)].astype(np.int64, copy=False)
            box = geometry.compute_native_crop_bbox(grid, valid_points, PATCH)
            retained = int(((points >= box[:3]) & (points < box[3:])).all(axis=1).sum())
            if not retained or not chunk_keys(box):
                counts["NO_ZERO_CONTROL"] += 1
                continue
            if control is None:
                control = Window(window, box, (0, 0), len(points), retained)
            result, reason = first_adding(box, points)
            counts[reason] += 1
            if witness is None and result is not None:
                offset, count = result
                witness = Window(window, box, offset, len(points), count)
    return control, witness
