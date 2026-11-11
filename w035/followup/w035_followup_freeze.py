"""Screen saved, immutable windows without consulting either planner."""
from w035_selector import PATCH, SHAPE, chunk_keys, first_adding, shift


def freeze(spec, geometry, surface, supervision, np, receipt):
    excluded = {tuple(map(int,key.split("/"))) for key in spec["excluded_omissions"]}
    frozen = []
    receipt["screen"] = []
    for group in spec["candidate_tiles"]:
        ty,tx = group["tile_yx"]
        tile = supervision.tile(ty,tx)
        chosen = None
        for flat in group["windows"]:
            y0,y1,x0,x1 = flat
            assert (y1-y0,x1-x0) == PATCH[1:]
            assert y0//512 == (y1-1)//512 == ty and x0//512 == (x1-1)//512 == tx
            row = {"flat_yx":flat,"tile_yx":[ty,tx],"status":"NOT_SCREENED_AFTER_FREEZE"}
            receipt["screen"].append(row)
            if chosen is not None:
                continue
            grid,valid = geometry.read_tifxyz_on_flat_grid(surface,y0=y0,y1=y1,x0=x0,x1=x1)
            if not valid[64,64] or not np.isfinite(grid[valid]).all():
                row["status"] = "INVALID_GEOMETRY"
                continue
            mask = tile[y0-512*ty:y1-512*ty,x0-512*tx:x1-512*tx]
            points = grid[valid & (mask != 0)].astype(np.int64,copy=False)
            box = geometry.compute_native_crop_bbox(grid,valid,PATCH)
            if not len(points) or not ((points>=box[:3]) & (points<box[3:])).all(axis=1).any():
                row["status"] = "NO_ZERO_SUPERVISION"
                continue
            result,reason = first_adding(box,points,exclude=excluded)
            row["status"] = reason
            if result is None:
                continue
            offset,retained = result
            added = chunk_keys(shift(box,offset))-chunk_keys(box)
            assert added and not added & excluded
            excluded |= added
            chosen = {"id":f"W{len(frozen)+1}","flat_yx":flat,"source_tile_yx":[ty,tx],
                      "native_zyx":list(box),"offsets_yx":[[0,0],list(offset)],
                      "supervised":len(points),"retained":retained,
                      "predicted_omissions":sorted("/".join(map(str,key)) for key in added)}
            row["status"] = "FROZEN"
            frozen.append(chosen)
    return frozen
