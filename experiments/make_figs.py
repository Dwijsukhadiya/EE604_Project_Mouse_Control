import json, math, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
FIG = "../fig/"
T = tiles(); names = list(T); det = detector()
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),
        (14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

# ---- Fold-gating figure: real landmarks on two HaGRID poses ----
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75))
for ax, (name, tip, mcp, title) in zip(axes, [("ok", 8, 5, "ok: real pinch"), ("one", 12, 9, "one: thumb on folded finger")]):
    f = webcam_canvas(T[name], 320); lm = landmarks(det, f)
    P = np.array([(p.x * 640, p.y * 480) for p in lm])
    x0, y0 = (640 - 320) // 2, (480 - 320) // 2
    ax.imshow(cv2.cvtColor(f[y0:y0+320, x0:x0+320], cv2.COLOR_BGR2RGB)); P -= (x0, y0)
    for a, b in CONN: ax.plot(*P[[a, b]].T, color="white", lw=0.8, alpha=0.8)
    ax.scatter(*P.T, s=4, color="#1f77b4", zorder=3)
    ax.plot(*P[[0, tip]].T, color="#d62728", lw=1.6, ls="-", label="tip to wrist")
    ax.plot(*P[[0, mcp]].T, color="#ff7f0e", lw=1.6, ls="--", label="knuckle to wrist")
    for i, c in ((0, "k"), (tip, "#d62728"), (mcp, "#ff7f0e")): ax.scatter(*P[i], s=18, color=c, zorder=4)
    d = lambda a, b: math.hypot(*(P[a] - P[b]))
    ax.set_title(f"{title}\nratio = {d(tip,0)/d(mcp,0):.2f}", fontsize=7, pad=2)
    ax.axis("off")
axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, -0.2), ncol=2, fontsize=6, frameon=False)
plt.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.1, wspace=0.05)
plt.savefig(FIG + "fold_gating.pdf"); plt.close()

# ---- Low-light example crops ----
ex = webcam_canvas(T["one"], 260)
d = np.clip(ex.astype(np.float32) * 0.12 + np.random.default_rng(1).normal(0, 2.0, ex.shape), 0, 255).astype(np.uint8)
cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l, a, b = cv2.split(cv2.cvtColor(d, cv2.COLOR_BGR2LAB))
clahe = cv2.cvtColor(cv2.merge((cl.apply(l), a, b)), cv2.COLOR_LAB2BGR)
x0, y0 = (640 - 260) // 2 - 20, (480 - 260) // 2 - 20
for nm, im in [("ll_orig", ex), ("ll_dark", d), ("ll_clahe", clahe), ("ll_gamma", vm.enhance_if_dark(d))]:
    cv2.imwrite(FIG + nm + ".png", im[y0:y0+300, x0:x0+300])

# ---- Jitter trace (ok pose, tracking mode, per-frame sensor noise) ----
base = webcam_canvas(T["ok"], 260, dx=-120, dy=30)
rng = np.random.default_rng(7); dv = detector(static=False); mcp = []
for _ in range(300):
    f = np.clip(base.astype(np.float32) + rng.normal(0, 3.0, base.shape), 0, 255).astype(np.uint8)
    lm = landmarks(dv, f)
    if lm is not None: mcp.append(vm.to_screen(lm[5].x, lm[5].y))
mcp = np.array(mcp[20:]); ema = mcp.copy()
for i in range(1, len(ema)): ema[i] = ema[i-1] + vm.SMOOTHING * (mcp[i] - ema[i-1])
t = np.arange(len(mcp))
fig, ax = plt.subplots(figsize=(3.3, 1.5))
ax.plot(t, mcp[:, 0] - mcp[:, 0].mean(), color="0.6", lw=0.7, label="raw landmark")
ax.plot(t, ema[:, 0] - mcp[:, 0].mean(), color="#1f77b4", lw=1.1, label=f"EMA, $\\alpha$ = {vm.SMOOTHING}")
ax.set_xlabel("Frame"); ax.set_ylabel("Cursor x offset (px)")
ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper right"); ax.grid(alpha=0.3)
plt.tight_layout(pad=0.2); plt.savefig(FIG + "jitter.pdf"); plt.close()
print("figs done")
