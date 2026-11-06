import json, math, time, numpy as np, cv2
from common import *
T = tiles()
variants = [(s, dx, dy) for s in (200, 260, 320) for dx, dy in ((0, 0), (-120, 30), (110, -20))]
frames = [webcam_canvas(t, s, dx=dx, dy=dy) for t in T.values() for s, dx, dy in variants]
OUT = {}

# ---------- Exp 2: low light ----------
_cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def clahe_if_dark(f):
    if cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() >= 90: return f
    l, a, b = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((_cl.apply(l), a, b)), cv2.COLOR_LAB2BGR)
det = detector()
rows = []
for k in [1.0, 0.5, 0.3, 0.2, 0.12, 0.08]:
    rng = np.random.default_rng(int(k * 1000))
    dark = [np.clip(f.astype(np.float32) * k + rng.normal(0, 2.0, f.shape), 0, 255).astype(np.uint8) for f in frames]
    r = {"k": k, "luma": float(np.mean([cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).mean() for d in dark]))}
    for name, fn in {"none": lambda f: f, "clahe": clahe_if_dark, "gamma": vm.enhance_if_dark}.items():
        r[name] = 100 * sum(landmarks(det, fn(d)) is not None for d in dark) / len(dark)
    rows.append(r); print({a: round(b, 1) for a, b in r.items()})
OUT["lowlight"] = rows
# example images for figure
ex = frames[9 * 9 + 0]  # 'one' tile? index by name below instead
names = list(T.keys())
ex = frames[names.index("one") * 9 + 1]
d = np.clip(ex.astype(np.float32) * 0.12 + np.random.default_rng(1).normal(0, 2.0, ex.shape), 0, 255).astype(np.uint8)
cv2.imwrite("fig_ll_orig.png", ex); cv2.imwrite("fig_ll_dark.png", d)
cv2.imwrite("fig_ll_clahe.png", clahe_if_dark(d)); cv2.imwrite("fig_ll_gamma.png", vm.enhance_if_dark(d))

# ---------- Exp 3: jitter (static hand, sensor noise, tracking mode) ----------
def run_sequence(seq):
    dv = detector(static=False); pts = []
    for f in seq:
        lm = landmarks(dv, f)
        pts.append(None if lm is None else (vm.to_screen(lm[5].x, lm[5].y), vm.to_screen(lm[8].x, lm[8].y)))
    dv.close(); return pts
jit = {}
for name in ["one", "palm", "ok"]:
    base = frames[names.index(name) * 9 + 1]
    rng = np.random.default_rng(7)
    seq = [np.clip(base.astype(np.float32) + rng.normal(0, 3.0, base.shape), 0, 255).astype(np.uint8) for _ in range(300)]
    pts = [p for p in run_sequence(seq) if p is not None][20:]
    mcp = np.array([p[0] for p in pts]); tip = np.array([p[1] for p in pts])
    ema = mcp.copy()
    for i in range(1, len(ema)): ema[i] = ema[i - 1] + vm.SMOOTHING * (mcp[i] - ema[i - 1])
    rms = lambda a: float(np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))
    step = lambda a: float(np.sqrt((np.diff(a, axis=0) ** 2).sum(1).mean()))
    jit[name] = dict(n=len(pts), tip_rms=rms(tip), mcp_rms=rms(mcp), ema_rms=rms(ema),
                     tip_step=step(tip), mcp_step=step(mcp), ema_step=step(ema))
    if name == "one":
        np.save("jitter_trace.npy", np.stack([tip, mcp, ema]))
    print(name, {a: round(b, 2) for a, b in jit[name].items()})
OUT["jitter"] = jit
a = vm.SMOOTHING
OUT["ema"] = dict(alpha=a, mean_delay_frames=(1 - a) / a, t90_frames=math.log(0.1) / math.log(1 - a))

# ---------- Exp 4: per-frame processing time ----------
dv = detector(static=False)
base = frames[names.index("one") * 9 + 1]; darkf = (base * 0.3).astype(np.uint8)
def timeit(fn, n=200):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts[20:])), float(np.std(ts[20:]))
pin = {k: vm.Pinch() for k in ("pinch_index", "pinch_middle", "pinch_ring")}
lm = landmarks(dv, base)
tm = {
    "flip_convert": timeit(lambda: cv2.cvtColor(cv2.flip(base, 1), cv2.COLOR_BGR2RGB)),
    "enhance_bright": timeit(lambda: vm.enhance_if_dark(base)),
    "enhance_dark": timeit(lambda: vm.enhance_if_dark(darkf)),
    "mediapipe": timeit(lambda: dv.process(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))),
    "classify": timeit(lambda: vm.classify(lm, 640, 480, pin), 2000),
}
for k, v in tm.items(): print(f"{k:15s} {v[0]:.3f} +- {v[1]:.3f} ms")
OUT["timing"] = tm
json.dump(OUT, open("results.json", "w"), indent=1)
