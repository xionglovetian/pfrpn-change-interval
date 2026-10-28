"""Minimal ESRI shapefile reader for the figure script (PolyLine / Polygon records only).

No geopandas / shapely / pyshp needed: the shapefile binary layout for these geometry types is
straightforward and the Natural Earth 110 m coastline is a single PolyLine layer.
"""
import struct


def read_shp_parts(path):
    """Return a list of polylines; each polyline is (n_points, xs, ys) as numpy-free lists."""
    with open(path, "rb") as f:
        data = f.read()
    # 100-byte header: file length is at offset 24 (big-endian 16-bit words)
    file_len = struct.unpack(">i", data[24:28])[0] * 2
    parts = []
    off = 100
    while off < file_len and off + 8 <= len(data):
        rec_num, rec_len = struct.unpack(">ii", data[off:off + 8])       # both big-endian
        body = data[off + 8: off + 8 + rec_len * 2]
        off += 8 + rec_len * 2
        if len(body) < 4:
            continue
        shape_type = struct.unpack("<i", body[0:4])[0]
        if shape_type == 0:                       # null shape
            continue
        if shape_type not in (3, 5, 13, 15, 23, 25):   # PolyLine / Polygon families
            continue
        n_parts, n_points = struct.unpack("<ii", body[36:44])
        parts_start = 44 + 4 * n_parts
        pts = body[parts_start: parts_start + 16 * n_points]
        xs = [0.0] * n_points
        ys = [0.0] * n_points
        for i in range(n_points):
            x, y = struct.unpack("<dd", pts[16 * i: 16 * i + 16])
            xs[i] = x
            ys[i] = y
        starts = list(struct.unpack("<" + "i" * n_parts, body[44:44 + 4 * n_parts]))
        starts.append(n_points)
        for k in range(n_parts):
            a, b = starts[k], starts[k + 1]
            parts.append((xs[a:b], ys[a:b]))
    return parts
