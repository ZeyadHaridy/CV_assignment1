"""Part 2.3 - build both pyramids on facade.png, rebuild, measure errors, make the figures."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_image, save_image
from paths import IMAGES, RESULTS
from pyramids import (reduce, expand, gaussian_pyramid, laplacian_pyramid, reconstruct,
                      lecture_style_pyramid)
from freq import blur_then_downsample

OUT = RESULTS; os.makedirs(OUT, exist_ok=True)
LEVELS = 6
img = load_image(f"{IMAGES}/part1_filters/facade.png")                 # color float32 in [0,1], 670 x 736 x 3
img = img.astype(np.float64)
print(f"image {img.shape} (sides are not powers of 2)")

gp = gaussian_pyramid(img, LEVELS)
lp = laplacian_pyramid(gp)
print("level  Gaussian shape     pixels   Laplacian shape")
for i, (g, l) in enumerate(zip(gp, lp)):
    print(f"  {i}    {str(g.shape[:2]):16s} {g.shape[0]*g.shape[1]:8d}   {l.shape[:2]}")
total = sum(g.shape[0] * g.shape[1] for g in gp); base = img.shape[0] * img.shape[1]
print(f"total pixels in the Gaussian pyramid / image = {total}/{base} = {total/base:.5f}  (4/3 = {4/3:.5f}); "
      f"Laplacian pyramid has the same sizes -> same total")


def mosaic(levels, to_display):
    """Level 0 on the left; levels 1, 2, 3, ... stacked in a column to its right (like the handout)."""
    H0, W0 = levels[0].shape[:2]; W1 = levels[1].shape[1]
    canvas = np.ones((H0, W0 + W1 + 4, 3))
    canvas[:, :W0] = to_display(levels[0], 0)
    y = 0
    for i in range(1, len(levels)):
        h, w = levels[i].shape[:2]
        canvas[y:y + h, W0 + 4:W0 + 4 + w] = to_display(levels[i], i)
        y += h + 4
    return canvas


# Display only: Gaussian levels as they are; Laplacian levels 0 .. n-2 as 0.5 + gain * L (zero = mid-gray),
# one global gain so levels stay comparable; the last Laplacian entry is a Gaussian level and is shown unchanged.
p = np.percentile(np.concatenate([np.abs(l).ravel() for l in lp[:-1]]), 99.5)
gain = 0.5 / p
print(f"Laplacian display: 0.5 + {gain:.2f} * L  (99.5th percentile of |L| over levels 0..{LEVELS-2} is {p:.4f}), clipped to [0, 1]")
save_image(f"{OUT}/fig_2_3_gaussian_pyramid.png", mosaic(gp, lambda g, i: np.clip(g, 0, 1)))
save_image(f"{OUT}/fig_2_3_laplacian_pyramid.png",
           mosaic(lp, lambda l, i: np.clip(l, 0, 1) if i == LEVELS - 1 else np.clip(0.5 + gain * l, 0, 1)))

# ---------------------------------------------------------------- reconstruction and its error
rec = reconstruct(lp)
err = rec - img
print(f"\nREBUILD (ours): shape {rec.shape}; max|error| = {np.abs(err).max():.3e}; RMSE = {np.sqrt(np.mean(err**2)):.3e}; "
      f"exactly zero in {100*np.mean(err == 0):.1f}% of the values")
# the same, from a float32 / 8-bit style input (what a loaded file really is)
img8 = np.round(img * 255) / 255
rec8 = reconstruct(laplacian_pyramid(gaussian_pyramid(img8, LEVELS)))
print(f"        8-bit-quantized input: max|error| = {np.abs(rec8 - img8).max():.3e}")

# 'level minus its own blur' details used with the same rebuild recipe
lec = lecture_style_pyramid(gp)
rec_l = gp[-1]
for lv in reversed(range(LEVELS - 1)):
    rec_l = lec[lv] + expand(rec_l, gp[lv].shape)
err_l = rec_l - img
print(f"REBUILD ('level minus its blur', Lecture-3 style as I understand it): max|error| = {np.abs(err_l).max():.4f}; "
      f"RMSE = {np.sqrt(np.mean(err_l**2)):.4f}")

fig, ax = plt.subplots(1, 4, figsize=(20, 5.6))
sc = np.percentile(np.abs(err_l), 99.5)
for a, im, t in [(ax[0], img, "original"), (ax[1], np.clip(rec, 0, 1), f"rebuilt from our Laplacian pyramid\nmax|error| = {np.abs(err).max():.1e}"),
                 (ax[2], np.clip(np.abs(err).max(axis=2) / sc, 0, 1), f"|error|, ours   (white = {sc:.3f})"),
                 (ax[3], np.clip(np.abs(err_l).max(axis=2) / sc, 0, 1), f"|error|, 'level minus its blur'\n(same scale, white = {sc:.3f})")]:
    a.imshow(im if im.ndim == 3 else im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=1); a.set_title(t, fontsize=11); a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_2_3_reconstruction.png", dpi=80); plt.close(fig)
save_image(f"{OUT}/facade_rebuilt.png", np.clip(rec, 0, 1))

# ---------------------------------------------------------------- how much aliasing does reduce() leave?  (link to 2.2)
print("\nALIASING LEFT BY reduce():  one reduce keeps every 2nd pixel, new Nyquist = 0.25 cycles/px; kernel response H(f) = cos^4(pi f)")
x = np.arange(1024)
for f in (0.30, 0.35, 0.40):
    sine = np.tile(np.sin(2 * np.pi * f * x), (64, 1))
    out = reduce(sine)[16, :]                                       # 512 samples; false frequency = 1 - 2f cycles/new px
    fa = 1 - 2 * f
    n = np.arange(len(out)); win = slice(6, 6 + 250)               # 250 samples = a whole number of periods for fa = 0.4, 0.3, 0.2
    meas = 2 * abs(np.mean(out[win] * np.exp(-2j * np.pi * fa * n[win])))
    print(f"  sine f={f:.2f}: false pattern amplitude {meas:.3f}   (cos^4(pi f) = {np.cos(np.pi*f)**4:.3f})")

rings = load_image(f"{IMAGES}/part2_frequencies/rings.png", gray=True).astype(np.float64)
h = 128; yy, xx = np.mgrid[0:h, 0:h]
region = (np.abs(xx - (h - 1) / 2) >= 45) | (np.abs(yy - (h - 1) / 2) >= 45)       # same flat region as in 2.2
strength = lambda a: float(np.sqrt(np.mean((a[region] - 0.5) ** 2)) / 0.3536)
r2 = reduce(reduce(rings))
print(f"rings.png reduced twice with our reduce (-> {r2.shape}): false-pattern strength = {strength(r2):.3f}")
for s in (2.0, 2.25, 2.5):
    print(f"   Gaussian sigma={s} then every 4th pixel (from 2.2)            : {strength(blur_then_downsample(rings, 4, s)):.3f}")
print(f"   every 4th pixel, no blur                                    : {strength(rings[::4, ::4]):.3f}")
save_image(f"{OUT}/rings_reduce_twice.png", np.clip(r2, 0, 1))
