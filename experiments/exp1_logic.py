"""Exp 1: what the original rules and the improved classifier do on static poses."""
import json, numpy as np
from common import *

SW, SH = 1920, 1080
def original_actions(lm, w, h):
    P = lambda i: (SW / w * int(lm[i].x * w), SH / h * int(lm[i].y * h))
    (ix, iy), (tx, ty), (mx, my), (rx, ry) = P(8), P(4), P(12), P(16)
    imx, rmx = P(5)[0], P(13)[0]
    a = []
    if abs(ty - iy) < 20: a.append("click")
    if abs(ty - my) < 20: a.append("right_click")
    if abs(ix - mx) < 2: a.append("scroll_up")
    if abs(mx - rx) < 20: a.append("scroll_down")
    if abs(imx - tx) < 2: a.append("zoom_in")
    if abs(rmx - tx) < 2: a.append("zoom_out")
    if abs(ty - ry) < 20: a.append("screenshot")
    return a

INTENT = {"ok": "pinch_index", "one": "move", "peace": "scroll", "two_up": "scroll", "three": "zoom"}
ORIG_INTENT = {"ok": {"click"}, "one": set(), "peace": {"scroll_up"}, "two_up": {"scroll_up"}, "three": set()}

def improved_label(lm, w, h, fold=True):
    vm.FOLD_RATIO = 1.25 if fold else 0.0
    pinches = {k: vm.Pinch() for k in ("pinch_index", "pinch_middle", "pinch_ring")}
    return vm.classify(lm, w, h, pinches)

T = tiles(); det = detector()
variants = [(s, dx, dy) for s in (200, 260, 320) for dx, dy in ((0, 0), (-120, 30), (110, -20))]
rows = []
for name, tile in T.items():
    for s, dx, dy in variants:
        f = webcam_canvas(tile, s, dx=dx, dy=dy)
        lm = landmarks(det, f)
        if lm is None:
            rows.append(dict(g=name, s=s, det=False)); continue
        rows.append(dict(g=name, s=s, det=True, orig=original_actions(lm, 640, 480),
                         nofold=improved_label(lm, 640, 480, fold=False),
                         imp=improved_label(lm, 640, 480, fold=True)))
json.dump(rows, open("exp1_rows.json", "w"), indent=1)

det_rows = [r for r in rows if r["det"]]
neutral = [r for r in det_rows if r["g"] not in INTENT]
intended = [r for r in det_rows if r["g"] in INTENT]
DISCRETE_IMP = {"pinch_index", "pinch_middle", "pinch_ring", "scroll", "zoom"}
o_fp = sum(len(r["orig"]) > 0 for r in neutral)
i_fp = sum(r["imp"] in DISCRETE_IMP for r in neutral)
n_fp = sum(r["nofold"] in DISCRETE_IMP for r in neutral)
n_ok = sum(r["nofold"] == INTENT[r["g"]] for r in intended)
print(f"no-fold: neutral spurious {n_fp}/{len(neutral)}  intended right {n_ok}/{len(intended)}")
o_multi = np.mean([len(r["orig"]) for r in det_rows])
o_ge2 = sum(len(r["orig"]) >= 2 for r in det_rows)
i_ok = sum(r["imp"] == INTENT[r["g"]] for r in intended)
o_ok = sum(set(r["orig"]) == ORIG_INTENT[r["g"]] for r in intended)
print(f"frames {len(rows)} detected {len(det_rows)}  neutral {len(neutral)} intended {len(intended)}")
print(f"neutral frames with spurious action: original {o_fp}/{len(neutral)}  improved {i_fp}/{len(neutral)}")
print(f"original mean actions per frame {o_multi:.2f}; frames with >=2 simultaneous actions {o_ge2}/{len(det_rows)}")
print(f"intended poses exactly right: original {o_ok}/{len(intended)} improved {i_ok}/{len(intended)}")
from collections import Counter
print("orig action freq", Counter(a for r in det_rows for a in r["orig"]))
print("\nper gesture (size 260 centred):")
for r in rows:
    if r["s"] == 260 and r.get("det") is not None:
        pass
for name in T:
    rs = [r for r in det_rows if r["g"] == name]
    oc = Counter(a for r in rs for a in r["orig"]); ic = Counter(r["imp"] for r in rs)
    nc = Counter(r["nofold"] for r in rs)
    print(f"{name:12s} n={len(rs)} orig={dict(oc)} nofold={dict(nc)} imp={dict(ic)}")
