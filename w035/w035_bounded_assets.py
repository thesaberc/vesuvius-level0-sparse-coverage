"""Bound the exact w035 Zarr shard reads and verify pinned input identities."""
import hashlib
import itertools
import json
import struct

from w035_selector import CANVAS, SHAPE

TILE_BYTES = 512**2
INDEX_BYTES = 12 * 11 * 16 + 4
FILL = 2**64 - 1

def crc32c(data):
    crc = 0xffffffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    return crc ^ 0xffffffff

class Shard:
    shape = (1, *CANVAS)
    dtype = "uint8"

    def __init__(self, path, name, np, decoder, charges):
        self.path, self.name, self.np, self.decoder = path, name, np, decoder
        self.charges = charges
        end = path.stat().st_size - INDEX_BYTES
        assert end >= 0
        with path.open("rb") as file:
            file.seek(end)
            index = file.read(INDEX_BYTES)
        assert len(index) == INDEX_BYTES
        assert crc32c(index[:-4]) == struct.unpack("<I", index[-4:])[0]
        self.entries = list(struct.iter_unpack("<QQ", index[:-4]))
        extents = []
        for offset, length in self.entries:
            if (offset, length) != (FILL, FILL):
                assert offset != FILL and 16 <= length <= end and offset <= end - length
                extents.append((offset, offset + length))
        extents.sort()
        assert all(a[1] <= b[0] for a, b in zip(extents, extents[1:]))

    def tile(self, y, x):
        assert 0 <= y < 12 and 0 <= x < 11
        group = "labels" if self.name == "inklabels" else "masks"
        assert sum(entry[0] == group for entry in self.charges) < 32, "Mask byte cap"
        self.charges.append((group, self.name, y, x))
        offset, length = self.entries[y * 11 + x]
        out = self.np.zeros((512, 512), dtype="uint8")
        if (offset, length) != (FILL, FILL):
            with self.path.open("rb") as file:
                file.seek(offset)
                data = file.read(length)
            assert len(data) == length
            version, _, _, typesize, nbytes, blocksize, cbytes = struct.unpack("<4B3I", data[:16])
            assert version == 2 and typesize == 1 and cbytes == length
            assert nbytes == TILE_BYTES and 0 < blocksize <= nbytes
            self.decoder.decode(data, out=out)
        return out

    def __getitem__(self, selection):
        z, ys, xs = selection
        assert z == 0 or z == slice(0, 1)
        y0, y1, x0, x1 = ys.start, ys.stop, xs.start, xs.stop
        assert ys.step is None and xs.step is None
        assert 0 <= y0 <= y1 <= CANVAS[0] and 0 <= x0 <= x1 <= CANVAS[1]
        out = self.np.zeros((y1-y0, x1-x0), dtype="uint8")
        for ty, tx in itertools.product(range(y0//512, (y1+511)//512),
                                        range(x0//512, (x1+511)//512)):
            oy, ox = ty*512, tx*512
            a, b, c, d = max(y0, oy), min(y1, oy+512), max(x0, ox), min(x1, ox+512)
            if a < b and c < d:
                out[a-y0:b-y0, c-x0:d-x0] = self.tile(ty, tx)[a-oy:b-oy, c-ox:d-ox]
        return out if z == 0 else out[None, ...]


def tile_intersections(box):
    y0, y1, x0, x1 = box
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(y1, CANVAS[0]), min(x1, CANVAS[1])
    if y0 >= y1 or x0 >= x1:
        return 0
    return ((y1-1)//512-y0//512+1) * ((x1-1)//512-x0//512+1)


def load_assets(package, assets, np, charges):
    import numcodecs

    scope = json.loads((package / "w035-stage1-assets-draft.json").read_bytes())
    release = (package / "alternative-native-release-metadata.json").read_bytes()
    assert hashlib.sha256(release).hexdigest() == scope["release_metadata_sha256"]
    saved = json.loads(release)[0]
    inputs = scope["asset_get"]
    assert len(inputs) == len({row["key"] for row in inputs}) == 12
    assert sum(row["bytes"] for row in inputs) == scope["expected_response_body_bytes"] == 950174
    assert scope["ct_chunk_bodies"] == scope["ct_head_requests"] == 0
    for row in inputs:
        path = assets / row["key"]
        assert path.stat().st_size == row["bytes"] and path.is_file()
        body = path.read_bytes()
        assert hashlib.md5(body, usedforsecurity=False).hexdigest() == row["etag"].strip('"')
        if "sha256" in row:
            assert hashlib.sha256(body).hexdigest() == row["sha256"]
    attrs, ct = (entry["json"] for entry in saved["volume"])
    multi, = attrs["multiscales"]
    assert [a["name"] for a in multi["axes"]] == ["z", "y", "x"]
    assert multi["datasets"][0] == {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1,1,1]}]}
    assert ct == {"shape": list(SHAPE), "chunks": [128,128,128], "dtype": "|u1", "fill_value": 0,
                  "order": "C", "filters": None, "dimension_separator": "/", "compressor": None, "zarr_format": 2}
    assert saved["decoded_geometry_bytes"] == 914904 <= scope["decoded_geometry_cap"]
    assert set(saved["masks"]) == {"inklabels", "supervision"}
    arrays = {}
    for name, mask in saved["masks"].items():
        metadata = mask["array"]["json"]
        assert metadata["shape"] == list(CANVAS) and metadata["data_type"] == "uint8"
        assert metadata["dimension_names"] == ["y", "x"] and metadata["fill_value"] == 0
        assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [6144,5632]
        codec, = metadata["codecs"]
        assert codec["name"] == "sharding_indexed" and codec["configuration"]["chunk_shape"] == [512,512]
        assert codec["configuration"]["index_location"] == "end"
        assert codec["configuration"]["index_codecs"] == [
            {"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]
        labels_attrs = mask["group"]["json"]["attributes"]
        assert labels_attrs["canvas_size"] == list(CANVAS) and labels_attrs["source_plane"] == 14
        assert labels_attrs["ome"]["multiscales"][0]["datasets"][0] == {
            "path": "0", "coordinateTransformations": [{"type": "scale", "scale": [9.362,9.362]}]}
        key, = mask["shards"]
        arrays[name] = Shard(assets / key, name, np, numcodecs.Blosc(cname="zstd", shuffle=2), charges)
    return (assets / saved["geometry_meta"]["key"]).parent, arrays
