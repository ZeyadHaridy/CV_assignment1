"""Part 3.1 - energy map and removed seams on broadway_tower.png (30% of the width)."""
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from utils import load_image, save_image
from paths import IMAGES, RESULTS
from seam import carve, energy_fd, draw_seams, seam_search_space

OUT = RESULTS; os.makedirs(OUT, exist_ok=True)
img = load_image(f"{IMAGES}/part3_seams/broadway_tower.png").astype(np.float64)
H, W = img.shape[:2]; n = int(round(0.30 * W))
print(f"broadway_tower {img.shape}; removing {n} seams = {100*n/W:.1f}% of the width -> {W-n} columns")
lo, hi = seam_search_space(H, W)
print(f"possible vertical seams: between 10^{lo:.0f} and 10^{hi:.0f};  DP work per seam: H*W*3 = {3*H*W:,} comparisons")

t = time.perf_counter(); res = carve(img, n, energy_fd); t_fd = time.perf_counter() - t
print(f"finite-difference energy: {t_fd:.1f} s for {n} seams ({1000*t_fd/n:.0f} ms per seam incl. energy recompute); result {res['image'].shape}")
assert res["image"].shape == (H, W - n, 3)
print(f"mean energy per seam pixel: first seam {res['costs'][0]:.4f}, last seam {res['costs'][-1]:.4f}, "
      f"image mean energy {energy_fd(img).mean():.4f}")

e0 = energy_fd(img)
fig, ax = plt.subplots(1, 2, figsize=(16, 5.6))
ax[0].imshow(np.clip(e0 / np.percentile(e0, 99.5), 0, 1), cmap="gray"); ax[0].set_title("energy (finite-difference gradient magnitude), shown / 99.5th percentile", fontsize=11)
ax[1].imshow(draw_seams(img, res["seams"])); ax[1].set_title(f"{n} removed seams (30% of the width) drawn on the original", fontsize=11)
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_3_1_energy_and_seams.png", dpi=80); plt.close(fig)
save_image(f"{OUT}/tower_carved_30pct.png", np.clip(res["image"], 0, 1))

# ---- does the padding mode of the energy matter?  (zero padding = Part 1 rule, reflect = Part 2 rule)
print("\nPADDING OF THE ENERGY:  zero (as in 1.2)  vs  reflect")
for mode in ("zeros", "reflect"):
    e = energy_fd(img, pad_mode=mode)
    r = carve(img, n, partial(energy_fd, pad_mode=mode))
    s = r["seams"]
    print(f"  {mode:8s}: energy on the 1-px frame = {np.r_[e[0], e[-1], e[:, 0], e[:, -1]].mean():.3f} (interior mean {e[2:-2, 2:-2].mean():.3f}); "
          f"seam pixels in the 5 leftmost / rightmost columns: {int(((s < 5) | (s >= W - 5)).sum())} of {s.size}; "
          f"mean seam energy {r['costs'].mean():.4f}")
    if mode == "zeros":
        res_zero = r
fig, ax = plt.subplots(1, 2, figsize=(16, 5.6))
ax[0].imshow(draw_seams(img, res_zero["seams"])); ax[0].set_title("energy with ZERO padding", fontsize=11)
ax[1].imshow(draw_seams(img, res["seams"])); ax[1].set_title("energy with REFLECT padding (used)", fontsize=11)
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_3_1_padding_comparison.png", dpi=70); plt.close(fig)
