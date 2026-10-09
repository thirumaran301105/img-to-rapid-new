"""Image -> ordered pen strokes in millimetres on the paper.

Modes
  lines    centre-lines of dark strokes (line art, text, sketches, logos): skeletonise and trace, no double lines
  outline  boundaries of dark regions (solid silhouettes)
  edges    thinned Canny edges (photos)
  shade    tonal hatching plus edges (photos, closest to the picture)

Everything is smoothed, simplified in millimetres (not pixels), ordered and joined so the pen lifts as little as possible.
No stroke is dropped for being short unless it is smaller than the pen itself.
"""
import cv2
import numpy as np

from curve_fit import fit_stroke, ops_to_polyline

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except Exception:  # dependency-free fallback
    _sk_skeletonize = None

MAX_OPS = 20000
MODES = ("auto", "lines", "outline", "edges", "shade")


# ---------------------------------------------------------------- loading and masks
def load_gray(data, max_side=1600):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not read the image. Use PNG, JPG or BMP.")
    if img.ndim == 3 and img.shape[2] == 4:  # flatten transparency onto white
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = max_side / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    elif s > 1.05:  # small images: upscale so skeletons and contours are smooth
        gray = cv2.resize(gray, None, fx=min(s, 2.5), fy=min(s, 2.5), interpolation=cv2.INTER_CUBIC)
    return gray


def ink_mask(gray, detail, k, pen_mm):
    """Boolean mask of 'ink'. Flattens uneven lighting first, then thresholds."""
    g = gray if np.median(gray) >= 110 else 255 - gray          # ink is the dark colour
    bg = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 25)
    norm = cv2.divide(g, bg, scale=255)                          # divide out the paper/lighting
    norm = cv2.GaussianBlur(norm, (0, 0), 0.8)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(250, max(5, t + (detail - 0.5) * 50))                # more detail keeps lighter strokes
    mask = (norm < t).astype(np.uint8)
    min_area = max(4, (0.5 * pen_mm / k) ** 2)                   # specks smaller than the pen
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------- skeleton tracing
def _zhang_suen(img):
    I = img.astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(I, 1)
            p2, p3, p4, p5 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:]
            p6, p7, p8, p9 = P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]
            nb = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            c = ((p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)) if step == 0 else ((p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0))
            m = (I == 1) & (nb >= 2) & (nb <= 6) & (A == 1) & c
            if m.any():
                I[m] = 0
                changed = True
    return I.astype(bool)


def skeletonize(mask):
    return _sk_skeletonize(mask) if _sk_skeletonize else _zhang_suen(mask)


_N8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def trace_skeleton(sk, spur_px, dt=None):
    """Skeleton pixels -> polylines (x, y). Short dead-end spurs are removed."""
    h, w = sk.shape
    P = np.pad(sk, 1)
    nbs = [P[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _N8]
    trans = sum((~a) & b for a, b in zip(nbs, nbs[1:] + nbs[:1]))   # 0->1 crossings around the pixel
    cnt = sum(a.astype(np.uint8) for a in nbs)
    end = sk & (cnt == 1)
    junc = sk & (trans >= 3)
    pix = set(zip(*np.nonzero(sk)))
    nodes = set(zip(*np.nonzero(end | junc)))
    ends = set(zip(*np.nonzero(end)))

    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in _N8 if (p[0] + dy, p[1] + dx) in pix]

    visited, edges = set(), []

    def walk(start, nxt):
        path, prev, cur = [start, nxt], start, nxt
        while cur not in nodes:
            visited.add(cur)
            near_prev = set(nbrs(prev)) | {prev}
            cand = [q for q in nbrs(cur) if q != prev and q not in visited and q not in path[-3:]]
            far = [q for q in cand if q not in near_prev] or cand
            if not far:
                break
            prev, cur = cur, far[0]
            path.append(cur)
        return path

    seen_pairs = set()
    for n in nodes:
        for m in nbrs(n):
            if m in nodes:
                key = frozenset((n, m))
                if key not in seen_pairs:
                    seen_pairs.add(key); edges.append([n, m])
            elif m not in visited:
                edges.append(walk(n, m))
    for p in pix:                                        # pure loops with no end or junction
        if p not in visited and p not in nodes:
            ns = nbrs(p)
            if ns:
                visited.add(p)
                path = walk(p, ns[0]); path.append(p) if len(path) > 2 else None
                edges.append(path)

    out = []
    for e in edges:
        if len(e) < 2:
            continue
        one_free_end = (e[0] in ends) != (e[-1] in ends)
        if one_free_end:
            lim = spur_px
            if dt is not None:                            # a spur at a junction of a thick stroke is about its half-width
                jy, jx = e[-1] if e[0] in ends else e[0]
                lim = max(lim, 1.0 * dt[jy, jx])
            if len(e) < lim:
                continue                                  # spur
        out.append(np.array([(x, y) for y, x in e], float))
    return out


# ---------------------------------------------------------------- extractors (return strokes in pixels)
def ex_lines(gray, k, detail, pen_mm):
    mask = ink_mask(gray, detail, k, pen_mm)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return trace_skeleton(skeletonize(mask), spur_px=pen_mm * 3 * (1.1 - detail) / k, dt=dt)


def ex_outline(gray, k, detail, pen_mm):
    cs = cv2.findContours(ink_mask(gray, detail, k, pen_mm).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    return [np.vstack([c.reshape(-1, 2), c.reshape(-1, 2)[:1]]).astype(float) for c in cs if len(c) >= 4]


def ex_edges(gray, k, detail, pen_mm):
    g = cv2.bilateralFilter(gray, 7, 40, 7)
    v, s = float(np.median(g)), 0.55 - 0.4 * detail
    e = cv2.Canny(g, int(max(0, (1 - s) * v * .7)), int(min(255, (1 + s) * v)))
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    return trace_skeleton(skeletonize(e), spur_px=pen_mm * 4 / k)


def _hatch(mask, angle, step_px, gap_px, min_px):
    h, w = mask.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    nw = int(h * abs(M[0, 1]) + w * abs(M[0, 0])); nh = int(h * abs(M[0, 0]) + w * abs(M[0, 1]))
    M[0, 2] += nw / 2 - c[0]; M[1, 2] += nh / 2 - c[1]
    rot = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (nw, nh), flags=cv2.INTER_NEAREST) > 0
    Minv = cv2.invertAffineTransform(M)
    segs = []
    for r, y in enumerate(np.arange(step_px / 2, nh, step_px)):
        d = np.diff(np.concatenate(([0], rot[int(y)].view(np.int8), [0])))
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        runs = []
        for a, b in zip(st, en):                          # bridge tiny gaps so the pen lifts less
            if runs and a - runs[-1][1] <= gap_px:
                runs[-1][1] = b
            else:
                runs.append([a, b])
        for a, b in runs:
            if b - a >= min_px:
                pts = np.array([[a, y], [b, y]], float)
                pts = (pts if r % 2 == 0 else pts[::-1]) @ Minv[:, :2].T + Minv[:, 2]
                segs.append(pts)
    return segs


def ex_shade(gray, k, detail, pen_mm):
    tone = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.GaussianBlur(gray, (0, 0), max(1.0, 0.8 / k)))
    spacing = max(pen_mm * 1.6, 0.8) * (1.6 - detail) / k
    strokes = []
    for thr, ang in ((190, 45), (140, 135), (90, 0), (50, 90)):
        strokes += _hatch(tone < thr, ang, spacing, gap_px=1.5 * pen_mm / k, min_px=1.5 * pen_mm / k)
    return strokes + ex_edges(gray, k, min(1.0, detail + 0.1), pen_mm)


def _ell(d):
    d = max(1, int(round(d)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def merge_through_junctions(strokes, radius_px, min_turn_cos=-0.6, look_px=6.0, rounds=8, align=0.0):
    """Skeleton tracing cuts every stroke at each junction. Re-join pieces that meet at a junction along the smoothest
    continuation (nearly straight through), the way a pen would carry on, so lines come out long instead of dashed."""
    from scipy.spatial import cKDTree
    cur = [np.asarray(s, float) for s in strokes if len(s) >= 2]

    def outward(pts):                                       # unit direction leaving the LAST point
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        acc, j = 0.0, len(pts) - 1
        while j > 0 and acc < look_px:
            j -= 1; acc += seg[j]
        v = pts[-1] - pts[j]
        n = float(np.hypot(*v))
        return v / n if n > 1e-9 else v

    for _ in range(rounds):
        if len(cur) < 2:
            break
        ends = [(i, e) for i in range(len(cur)) for e in (0, 1)]
        P = np.array([cur[i][0] if e == 0 else cur[i][-1] for i, e in ends])
        D = np.array([outward(cur[i][::-1] if e == 0 else cur[i]) for i, e in ends])
        cand = []
        for a, b in cKDTree(P).query_pairs(radius_px):
            if ends[a][0] == ends[b][0]:
                continue
            c = float(D[a] @ D[b])
            if c > min_turn_cos:
                continue
            if align:                                        # across a real gap: both ends must point at each other
                v = P[b] - P[a]; n = float(np.hypot(*v))
                if n > 2.5 and (D[a] @ v / n < align or D[b] @ -v / n < align):
                    continue
            cand.append((c, a, b))
        if not cand:
            break
        cand.sort()
        used, merged, out = set(), 0, []
        for c, a, b in cand:
            (ia, ea), (ib, eb) = ends[a], ends[b]
            if ia in used or ib in used:
                continue
            A = cur[ia] if ea == 1 else cur[ia][::-1]
            B = cur[ib] if eb == 0 else cur[ib][::-1]
            out.append(np.vstack([A, B])); used.add(ia); used.add(ib); merged += 1
        cur = out + [s for i, s in enumerate(cur) if i not in used]
        if not merged:
            break
    return cur


def ink_mask_soft(gray, detail, k, pen_mm):
    """Like ink_mask, but also keeps faint thin lines (light pencil, grey shirt lines) that are long enough to be real.
    Dark strokes are not thickened; paper texture and specks are rejected by size."""
    g = gray if np.median(gray) >= 110 else 255 - gray
    bg = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 25)
    pen_px = pen_mm / k
    norm = cv2.GaussianBlur(cv2.divide(g, bg, scale=255), (0, 0), max(0.8, 0.1 * pen_px))   # light smoothing; strands in dense hair must stay separate
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(250, max(5, t + (detail - 0.5) * 50))
    strong = norm < t
    weak = norm < min(247, t + (255 - t) * (0.25 + 0.5 * detail))
    add = weak & ~(cv2.dilate(strong.astype(np.uint8), _ell(2 + 0.3 * pen_px)) > 0)       # skip the soft halo around dark strokes
    n, lab, st, _ = cv2.connectedComponentsWithStats(add.astype(np.uint8), connectivity=8)
    big = (st[:, cv2.CC_STAT_AREA] >= 1.2 * pen_px ** 2) & (np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) >= 2.5 * pen_px)
    big[0] = False
    bridge = (cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_CLOSE, _ell(0.9 * pen_px)) > 0) & weak   # heal gaps inside one stroke
    mask = strong | bridge | big[lab]
    if mask.sum() > 2.5 * max(1, strong.sum()):                          # a noisy photo: do not trust the faint layer
        mask = strong
    min_area = max(4, (0.5 * pen_px) ** 2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


def drop_shadow_bits(strokes, pen_px, max_len_pen=3.0, near_pen=1.0):
    """Remove short fragments that lie within about one pen width of a longer stroke: the pen would cover them anyway."""
    from scipy.spatial import cKDTree
    def ln(s): return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())
    L = [ln(s) for s in strokes]
    big = [s for s, l in zip(strokes, L) if l >= max_len_pen * pen_px]
    if not big:
        return strokes
    pts = []
    for s in big:
        seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
        n = np.maximum(1, np.ceil(seg / max(1.0, 0.25 * pen_px)).astype(int))
        for a, b, m in zip(s[:-1], s[1:], n):
            pts.append(a + (b - a) * (np.arange(m)[:, None] / m))
        pts.append(s[-1:])
    tree = cKDTree(np.vstack(pts))
    out = []
    for s, l in zip(strokes, L):
        if l < max_len_pen * pen_px:
            d, _ = tree.query(s)
            if float(d.max()) <= near_pen * pen_px:
                continue
        out.append(s)
    return out


def _fill_small_holes(mask, max_area):
    """Fill background pockets smaller than the pen can draw. Pinholes in a thin stroke otherwise make the skeleton loop around them."""
    inv = (~mask).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = st[:, cv2.CC_STAT_AREA] <= max_area
    small[0] = False
    h, w = mask.shape
    for i in set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1]):   # never fill the page background
        small[i] = False
    return mask | small[lab]


def ex_auto(gray, k, detail, pen_mm):
    """Pen-aware line extraction for sketches, line art and portraits.
    Works at about 1/6 pen width per pixel, then splits the ink in two:
    strokes thinner than the pen become centre-lines, areas wider than the pen are drawn as a contour plus hatching."""
    f = float(np.clip(k / (pen_mm / 6.0), 1.0, 3.0))                  # supersample so strokes are smooth to sub-pen detail
    g = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1.05 else gray
    kk = k * gray.shape[1] / g.shape[1]                                # mm per working pixel
    pen_px = pen_mm / kk
    mask = ink_mask_soft(g, detail, kk, pen_mm)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _ell(3)) > 0   # heal one-pixel breaks in thin strokes
    mask = _fill_small_holes(mask, 0.12 * pen_px ** 2)

    # areas the pen cannot trace as a line: wider than ~1.4 pens
    dt0 = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = (dt0 >= cv2.dilate(dt0, np.ones((3, 3), np.uint8))) & (dt0 >= 1.5)
    w_typ = 2.0 * float(np.median(dt0[ridge])) if ridge.any() else pen_px      # how wide this image's own strokes are
    thick = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _ell(max(1.8 * pen_px, 2.6 * w_typ))) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(thick.astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= 2.0 * pen_px ** 2
    keep[0] = False
    thick = keep[lab]
    thin = mask & ~(cv2.dilate(thick.astype(np.uint8), _ell(2)) > 0)

    strokes = []
    dt = cv2.distanceTransform(thin.astype(np.uint8), cv2.DIST_L2, 3)
    thin_strokes = trace_skeleton(skeletonize(thin), spur_px=pen_mm * 2.5 * (1.1 - detail) / kk, dt=dt)
    thin_strokes = merge_through_junctions(thin_strokes, radius_px=2.2, look_px=max(5.0, 0.8 * pen_px))
    thin_strokes = merge_through_junctions(thin_strokes, radius_px=2.6 * pen_px, min_turn_cos=-0.9, look_px=max(8.0, 2.0 * pen_px), rounds=3, align=0.88)
    strokes += drop_shadow_bits(thin_strokes, pen_px)

    if thick.any():                                                    # fill wide areas: edge line + hatch inside
        edge_in = cv2.erode(thick.astype(np.uint8), _ell(pen_px)) > 0
        cs = cv2.findContours(edge_in.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
        strokes += [np.vstack([c.reshape(-1, 2), c.reshape(-1, 2)[:1]]).astype(float) for c in cs if len(c) >= 6]
        fill = cv2.erode(thick.astype(np.uint8), _ell(1.9 * pen_px)) > 0
        strokes += _hatch(fill, 45, 0.85 * pen_px, gap_px=1.5 * pen_px, min_px=0.6 * pen_px)
    return [s / (g.shape[1] / gray.shape[1]) for s in strokes]         # back to the caller's pixel grid


EXTRACTORS = {"auto": ex_auto, "lines": ex_lines, "outline": ex_outline, "edges": ex_edges, "shade": ex_shade}


# ---------------------------------------------------------------- clean-up in pixels / millimetres
def is_closed(s):
    return len(s) >= 4 and np.allclose(s[0], s[-1])


def smooth(s, win=5):
    if len(s) < win + 2:
        return s
    closed, h = is_closed(s), win // 2
    body = s[:-1] if closed else s
    pad = np.vstack([body[-h:], body, body[:h]]) if closed else np.vstack([np.repeat(body[:1], h, 0), body, np.repeat(body[-1:], h, 0)])
    ker = np.ones(win) / win
    out = np.column_stack([np.convolve(pad[:, i], ker, mode="valid") for i in range(2)])
    if closed:
        return np.vstack([out, out[:1]])
    out[0], out[-1] = s[0], s[-1]
    return out


def simplify(s, tol):
    if len(s) < 3:
        return s
    closed = is_closed(s)
    body = s[:-1] if closed else s
    ap = cv2.approxPolyDP(body.astype(np.float32).reshape(-1, 1, 2), tol, closed).reshape(-1, 2).astype(float)
    return np.vstack([ap, ap[:1]]) if closed else ap


def length(s):
    return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())


def order_strokes(strokes):
    """Greedy nearest-neighbour. Open strokes may start at either end, closed loops at any vertex."""
    n = len(strokes)
    if n == 0:
        return []
    closed = [is_closed(s) for s in strokes]
    pts, sid, kk = [], [], []
    for i, s in enumerate(strokes):
        cand = s[:-1] if closed[i] else s[[0, -1]]
        pts.append(cand); sid.append(np.full(len(cand), i)); kk.append(np.arange(len(cand)))
    P, S, K = np.vstack(pts), np.concatenate(sid), np.concatenate(kk)
    alive, pos, out = np.ones(n, bool), np.zeros(2), []
    for it in range(n):
        if it % 150 == 0:
            m = alive[S]; P, S, K = P[m], S[m], K[m]
        d = np.where(alive[S], np.hypot(P[:, 0] - pos[0], P[:, 1] - pos[1]), np.inf)
        j = int(d.argmin()); i, k = int(S[j]), int(K[j]); s = strokes[i]
        if closed[i]:
            body = s[:-1]; s = np.vstack([np.roll(body, -k, axis=0), body[k]])
        elif k == 1:
            s = s[::-1]
        out.append(s); alive[i] = False; pos = s[-1]
    return out


def join_close(strokes, gap):
    """Chain strokes whose end and next start are within `gap` mm: no pen lift needed."""
    chains = []
    for s in strokes:
        if chains and np.linalg.norm(s[0] - chains[-1][-1][-1]) <= gap:
            chains[-1].append(s)
        else:
            chains.append([s])
    return [np.vstack(c) if len(c) > 1 else c[0] for c in chains]


# ---------------------------------------------------------------- pipeline
def build_paths(data, area_w, area_h, mode="lines", detail=0.5, pen_mm=0.6, margin=5.0, tol_mm=None):
    """Returns (polylines for preview, ops for RAPID, image box, info). All in mm on the page."""
    if mode not in MODES:
        raise ValueError("Unknown line style.")
    detail = min(max(float(detail), 0.0), 1.0)
    pen_mm = min(max(float(pen_mm), 0.2), 3.0)
    gray = load_gray(data)
    h, w = gray.shape
    k = min((area_w - 2 * margin) / w, (area_h - 2 * margin) / h)      # mm per pixel
    ox = margin + (area_w - 2 * margin - w * k) / 2
    oy = margin + (area_h - 2 * margin - h * k) / 2

    dense = []
    for s in EXTRACTORS[mode](gray, k, detail, pen_mm):
        s = smooth(s, 3 if len(s) > 12 else 1)
        s = simplify(np.column_stack([ox + s[:, 0] * k, oy + (h - s[:, 1]) * k]), 0.03)   # light decimation only
        if len(s) >= 2 and length(s) >= 1.5 * pen_mm:
            dense.append(s)
    if not dense:
        raise ValueError("No lines found. Try another line style, a higher detail setting, or a higher-contrast image.")

    chains = join_close(order_strokes(dense), gap=pen_mm * 2)
    chains = [np.column_stack([np.clip(c[:, 0], 0, area_w), np.clip(c[:, 1], 0, area_h)]) for c in chains]

    tol = float(tol_mm) if tol_mm else max(0.05, 0.25 * pen_mm)        # how far a line or arc may stray from the traced path
    for _ in range(8):
        ops = [fit_stroke(c, tol) for c in chains]
        n_ops = sum(len(o["segs"]) for o in ops)
        if n_ops <= MAX_OPS:
            break
        tol *= 1.4
    else:
        raise ValueError("Too detailed for one program. Lower the detail or use a larger pen width.")
    polylines = [ops_to_polyline(o) for o in ops]
    return polylines, ops, {"x": ox, "y": oy, "w": w * k, "h": h * k}, {"tol_mm": round(tol, 3), "ops": n_ops}


def path_stats(strokes, robot):
    draw = sum(length(s) for s in strokes)
    travel, pos = 0.0, np.zeros(2)
    for s in strokes:
        travel += float(np.linalg.norm(s[0] - pos)); pos = s[-1]
    secs = draw / robot.draw_speed + travel / robot.travel_speed + len(strokes) * 0.8
    return {"strokes": len(strokes), "points": int(sum(len(s) for s in strokes)),
            "draw_mm": round(draw), "travel_mm": round(travel), "est_seconds": round(secs)}
