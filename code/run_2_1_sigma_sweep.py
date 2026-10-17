"""Part 2.1 - how sigma_low / sigma_high were chosen for the man + cat hybrid, plus a check of the Fourier-image 'cross'.

Everything quoted in the report about the choice of sigma comes from this script's log.
Far-away viewing is simulated with OUR reduce() (3 levels = 1/8 size); no library resizing.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_image
from paths import IMAGES, RESULTS
from align_images import align_images
from freq import hybrid_image, to_gray, gaussian_blur, gaussian_cutoff_frequency, log_fft_magnitude
from pyramids import reduce

D = IMAGES + "/part2_frequencies/"
man, cat = load_image(D + "hybrid_man.jpg"), load_image(D + "hybrid_cat.jpg")
cat_al = align_images(man, cat, [(299, 341), (443, 332)], [(605, 290), (760, 372)])
gm, gc = to_gray(man), to_gray(cat_al)


def far(x, levels=3):
    """What the hybrid looks like from far away: shrunk 2**levels times with our reduce()."""
    x = np.clip(x, 0, 1)
    for _ in range(levels):
        x = reduce(x)
    return x


# ------------------------------------------------------------------ contact sheets (look at them!)
lows, highs = [3, 6, 9], [2, 4, 6]
fig, ax = plt.subplots(3, 3, figsize=(12, 12)); fig2, ax2 = plt.subplots(3, 3, figsize=(12, 12))
for i, sl in enumerate(lows):
    for j, sh in enumerate(highs):
        h = np.clip(hybrid_image(gm, gc, sl, sh), 0, 1)
        ax[i, j].imshow(h[190:570, 180:560], cmap="gray", vmin=0, vmax=1); ax[i, j].axis("off")
        ax[i, j].set_title(f"NEAR (full-resolution crop)  sigma_low={sl}  sigma_high={sh}", fontsize=9)
        f = far(h); canvas = np.full((200, 200), 0.8); fh, fw = f.shape
        canvas[(200 - fh) // 2:(200 - fh) // 2 + fh, (200 - fw) // 2:(200 - fw) // 2 + fw] = f
        ax2[i, j].imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax2[i, j].axis("off")
        ax2[i, j].set_title(f"FAR (1/8 size)  sigma_low={sl}  sigma_high={sh}", fontsize=9)
fig.tight_layout(); fig.savefig(f"{RESULTS}/fig_2_1_sigma_sweep_near.png", dpi=60)
fig2.tight_layout(); fig2.savefig(f"{RESULTS}/fig_2_1_sigma_sweep_far.png", dpi=60)
print("sigma -> half-amplitude cutoff of the Gaussian (cycles/pixel, period in pixels)")
for s in (2, 3, 4, 6, 9):
    print(f"  sigma {s}: {gaussian_cutoff_frequency(s):.4f} cyc/px (period {1 / gaussian_cutoff_frequency(s):.1f} px)")

# ------------------------------------------------------------------ numbers
fine = lambda x: x - gaussian_blur(x, 2.0)                       # fine-detail band, used ONLY for measuring
corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
fm, fc = fine(gm), fine(gc)
print("\nnear = correlation of the hybrid's fine-detail band with the CAT's (want high) and the MAN's (want low)")
print("far  = RMSE between the hybrid and the MAN / the CAT after shrinking to 1/8 with our reduce (want low / high)")
print(f"{'s_low':>5} {'s_high':>6} | {'near~cat':>8} {'near~man':>8} | {'far RMSE man':>12} {'far RMSE cat':>12}")
for sl in (4, 6, 8):
    for sh in (2, 3, 4, 5):
        h = hybrid_image(gm, gc, sl, sh); fh = fine(h)
        print(f"{sl:>5} {sh:>6} | {corr(fh, fc):8.3f} {corr(fh, fm):8.3f} | "
              f"{np.sqrt(np.mean((far(h) - far(gm)) ** 2)):12.4f} {np.sqrt(np.mean((far(h) - far(gc)) ** 2)):12.4f}")

# ------------------------------------------------------------------ is the bright cross in the low-pass FFT a border artifact?
lp = gaussian_blur(gm, 6)
H, W = lp.shape; cy, cx = H // 2, W // 2
win = np.outer(np.hanning(H), np.hanning(W))


def cross_excess(img):
    """Mean log-magnitude ON the two central axes (beyond 20 px from DC) minus the mean on parallel lines 15 px away."""
    L = log_fft_magnitude(img)
    on = np.r_[L[cy, cx + 20:], L[cy, :cx - 20], L[cy + 20:, cx], L[:cy - 20, cx]].mean()
    off = np.r_[L[cy + 15, cx + 20:], L[cy + 15, :cx - 20], L[cy + 20:, cx + 15], L[:cy - 20, cx + 15]].mean()
    return on - off


print("\nFOURIER CROSS of the low-pass man (sigma=6): log-magnitude on the central axes minus just beside them")
print(f"  plain image                : {cross_excess(lp):+.2f}")
print(f"  image x Hann window (no border jump): {cross_excess(lp * win):+.2f}")
