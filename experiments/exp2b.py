import json, math, numpy as np, cv2
from common import *
T = tiles(); det = detector()
variants = [(s, dx, dy) for s in (200, 260, 320) for dx, dy in ((0, 0), (-120, 30), (110, -20))]
frames = [webcam_canvas(t, s, dx=dx, dy=dy) for t in T.values() for s, dx, dy in variants]

def gamma_fix(f, target=120.0):
    m = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()
    if m >= 90: return f
    g = math.log(target / 255) / math.log(max(m, 1.0) / 255)
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(f, lut)

def gain_fix(f, target=120.0):
    m = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()
    if m >= 90: return f
    return cv2.convertScaleAbs(f, alpha=target / max(m, 1.0))

def clahe_global(f):  # CLAHE with a single tile = global, clip-limited histogram equalisation
    if cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() >= 90: return f
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))
    l, a, b = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((c.apply(l), a, b)), cv2.COLOR_LAB2BGR)

methods = {"none": lambda f: f, "clahe8x8": vm.enhance_if_dark, "gamma": gamma_fix, "gain": gain_fix, "clahe1x1": clahe_global}
res = []
for k in [1.0, 0.5, 0.3, 0.2, 0.12, 0.08]:
    rng = np.random.default_rng(int(k * 1000))
    dark = [np.clip(f.astype(np.float32) * k + rng.normal(0, 2.0, f.shape), 0, 255).astype(np.uint8) for f in frames]
    row = {"k": k, "luma": float(np.mean([cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).mean() for d in dark]))}
    for name, fn in methods.items():
        row[name] = 100 * sum(landmarks(det, fn(d)) is not None for d in dark) / len(dark)
    print({a: round(b, 1) for a, b in row.items()})
    res.append(row)
json.dump(res, open("exp2b.json", "w"), indent=1)
