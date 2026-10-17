"""Part 1.2 and 1.3: produce all figures and the numbers quoted in the report."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_image
from paths import IMAGES, RESULTS
from conv import separable_factors
from filters import default_size
from edges import (finite_difference_gradient, dog_gradient, dog_filters, threshold_edges,
                   signed_for_display, magnitude_for_display)

IMG = f"{IMAGES}/part1_filters/facade.png"
OUT = RESULTS
SIGMA, T12, T13 = 2, 0.25, 0.08                     # chosen parameters (see report text)
os.makedirs(OUT, exist_ok=True)

img = load_image(IMG, gray=True)                     # float32 in [0,1], shape (670, 736)
print(f"image: part1_filters/facade.png  shape={img.shape}")


def panel(ax, im, title, binary=False):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest" if binary else "antialiased")
    ax.set_title(title, fontsize=11)
    if binary:      # colored outline: a white frame line must not vanish into the white page background
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("tab:red"); sp.set_linewidth(1.5)
    else:
        ax.axis("off")


def four_maps(gx, gy, mag, edges, name, labels):
    interior = (slice(12, -12), slice(12, -12))      # display scale ignores the frame artifact
    s_d = np.percentile(np.abs(np.concatenate([gx[interior].ravel(), gy[interior].ravel()])), 99.5)
    s_m = np.percentile(mag[interior], 99.5)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.9))
    panel(ax[0], signed_for_display(gx, s_d), labels[0])
    panel(ax[1], signed_for_display(gy, s_d), labels[1])
    panel(ax[2], magnitude_for_display(mag, s_m), labels[2])
    panel(ax[3], edges.astype(float), labels[3], binary=True)
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}", dpi=110); plt.close(fig)
    return s_d, s_m


# ------------------------------------------------------------------ 1.2
gx, gy, mag = finite_difference_gradient(img)                       # zero padding (Part 1 rule)
edges12 = threshold_edges(mag, T12)
s_d, s_m = four_maps(gx, gy, mag, edges12, "fig_1_2_maps.png",
                     ["dI/dx  (gray = 0)", "dI/dy  (gray = 0)", "gradient magnitude", f"edges, t = {T12}"])
inner = (slice(2, -2), slice(2, -2))
print("\n[1.2] finite differences, zero padding")
print(f"  gx range [{gx.min():.3f}, {gx.max():.3f}]  gy range [{gy.min():.3f}, {gy.max():.3f}]  mag max {mag.max():.3f}")
print(f"  display scale: derivatives +-{s_d:.3f} (99.5th pct of interior |g|), magnitude 0..{s_m:.3f}; arrays not modified")
print(f"  threshold {T12}: {100*edges12.mean():.1f}% of all pixels, {100*edges12[inner].mean():.1f}% of interior pixels are edges")
robust = np.median(np.abs(gx[inner] - np.median(gx[inner]))) * 1.4826
print(f"  robust noise level of gx (MAD): {robust:.3f}  -> threshold is {T12/robust:.1f} x that level")

# frame artifact: zero vs reflect
gxr, gyr, magr = finite_difference_gradient(img, pad_mode="reflect")
edges12r = threshold_edges(magr, T12)
border = np.zeros(img.shape, bool); border[0] = border[-1] = True; border[:, 0] = border[:, -1] = True
print(f"  frame: mean magnitude on the 1-px border = {mag[border].mean():.3f} (zero pad) vs {magr[border].mean():.3f} (reflect); "
      f"interior {mag[inner].mean():.3f}")
print(f"         border pixels flagged as edges: {100*edges12[border].mean():.1f}% (zero pad) vs {100*edges12r[border].mean():.1f}% (reflect)")
print(f"         top row: corr(magnitude, image brightness) = {np.corrcoef(mag[0, 1:-1], img[0, 1:-1])[0,1]:.2f}")
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
c = (slice(0, 110), slice(0, 110))
panel(ax[0], img[c], "image, top-left corner (110x110)")
panel(ax[1], edges12[c].astype(float), "edges, zero padding", binary=True)
panel(ax[2], edges12r[c].astype(float), "edges, reflect padding", binary=True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_1_2_frame_corner.png", dpi=110); plt.close(fig)

# threshold sweep (justification)
ths12, ths13 = [0.15, 0.25, 0.35, 0.45], [0.05, 0.08, 0.11, 0.15]
gx3, gy3, mag3 = dog_gradient(img, SIGMA)                            # computed here for the sweep too
inner_sweep = (slice(12, -12), slice(12, -12))
print("\nTHRESHOLD SWEEP (share of pixels flagged as edges: whole image / interior)")
for t in ths12:
    e = threshold_edges(mag, t); print(f"  1.2  t={t:.2f}: {100 * e.mean():5.1f}% / {100 * e[inner_sweep].mean():5.1f}%")
for t in ths13:
    e = threshold_edges(mag3, t); print(f"  1.3  t={t:.2f}: {100 * e.mean():5.1f}% / {100 * e[inner_sweep].mean():5.1f}%")
fig, ax = plt.subplots(2, 4, figsize=(16, 9.2))
for j, t in enumerate(ths12):
    panel(ax[0, j], threshold_edges(mag, t).astype(float), f"1.2   t = {t}", True)
for j, t in enumerate(ths13):
    panel(ax[1, j], threshold_edges(mag3, t).astype(float), f"1.3 (sigma={SIGMA})   t = {t}", True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_1_threshold_sweep.png", dpi=90); plt.close(fig)

# ------------------------------------------------------------------ 1.3
n = default_size(SIGMA)
dog_x, dog_y = dog_filters(SIGMA)
fx, fy = separable_factors(dog_x), separable_factors(dog_y)
print(f"\n[1.3] DoG, sigma = {SIGMA}, filter size {n}x{n}")
print(f"  DoG_x: min {dog_x.min():.4f} max {dog_x.max():.4f} sum {dog_x.sum():.1e}; rank-1 (separable): {fx is not None}")
print(f"  DoG_y == DoG_x^T: {np.allclose(dog_y, dog_x.T)}")
s = np.abs(dog_x).max()
fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
for a, f, t in [(ax[0], dog_x, f"DoG in x, sigma = {SIGMA}"), (ax[1], dog_y, f"DoG in y, sigma = {SIGMA}")]:
    a.imshow(signed_for_display(f, s), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_1_3_dog_filters.png", dpi=110); plt.close(fig)

edges13 = threshold_edges(mag3, T13)
s_d3, s_m3 = four_maps(gx3, gy3, mag3, edges13, "fig_1_3_maps.png",
                       [f"DoG dI/dx  (sigma={SIGMA})", f"DoG dI/dy  (sigma={SIGMA})", "gradient magnitude", f"edges, t = {T13}"])
inner3 = (slice(10, -10), slice(10, -10))
print(f"  gx range [{gx3.min():.3f}, {gx3.max():.3f}]  mag max {mag3.max():.3f}")
print(f"  display scale: derivatives +-{s_d3:.3f}, magnitude 0..{s_m3:.3f}")
print(f"  threshold {T13}: {100*edges13.mean():.1f}% of all pixels, {100*edges13[inner3].mean():.1f}% of interior pixels are edges")
print(f"  1.2 threshold {T12} keeps {100*edges12[inner3].mean():.1f}% of the same interior region (similar density -> fair comparison)")
robust3 = np.median(np.abs(gx3[inner3] - np.median(gx3[inner3]))) * 1.4826
print(f"  robust noise level of DoG gx (MAD): {robust3:.3f}  (finite difference: {robust:.3f})")

# how much does the DoG shrink noise?  (one real number from the actual photo)
print(f"  magnitude 99th percentile (interior): 1.2 = {np.percentile(mag[inner3],99):.3f}, DoG = {np.percentile(mag3[inner3],99):.3f}")

# frame width for each operator, from a constant image (any response is purely the padding artifact)
const = np.full((80, 80), 0.6)
_, _, c12 = finite_difference_gradient(const)
_, _, c13 = dog_gradient(const, SIGMA)
w12 = int((c12[40] > 1e-6).sum() // 2); w13 = int((c13[40] > 1e-6).sum() // 2)
print(f"  frame band width on a constant image: 1.2 -> {w12} px per side, DoG -> {w13} px per side (the {n}x{n} filter reaches {n // 2} px)")

# side by side crop
r, cc = slice(250, 450), slice(230, 430)
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
panel(ax[0, 0], img[r, cc], "image crop (200x200)")
panel(ax[0, 1], magnitude_for_display(mag, s_m)[r, cc], "magnitude 1.2")
panel(ax[0, 2], magnitude_for_display(mag3, s_m3)[r, cc], f"magnitude DoG sigma={SIGMA}")
panel(ax[1, 0], img[r, cc] * 0 + 1, "")
ax[1, 0].axis("off")
panel(ax[1, 1], edges12[r, cc].astype(float), f"edges 1.2 (t={T12})", True)
panel(ax[1, 2], edges13[r, cc].astype(float), f"edges DoG (t={T13})", True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_1_2_vs_1_3_crop.png", dpi=90); plt.close(fig)

np.save(f"{OUT}/mag_1_2.npy", mag); np.save(f"{OUT}/mag_1_3.npy", mag3)
print("\nfigures written to results/")
