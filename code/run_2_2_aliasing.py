"""Part 2.2 - aliasing experiment on rings.png (shrink to 1/4 width and height)."""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_image, save_image
from paths import IMAGES, RESULTS
from freq import downsample, blur_then_downsample, log_fft_magnitude

OUT = RESULTS; os.makedirs(OUT, exist_ok=True)
FACTOR = 4
rings = load_image(f"{IMAGES}/part2_frequencies/rings.png", gray=True).astype(np.float64)   # 512x512

# --- ideal alias-free reference (a CHECK only): keep |fx|,|fy| < new Nyquist = 1/(2*4), then keep every 4th pixel
F = np.fft.fft2(rings)
fy = np.fft.fftfreq(rings.shape[0])[:, None]; fx = np.fft.fftfreq(rings.shape[1])[None, :]
ideal = downsample(np.real(np.fft.ifft2(F * ((np.abs(fy) < 1 / (2 * FACTOR)) & (np.abs(fx) < 1 / (2 * FACTOR))))), FACTOR)

# --- region where the correct 1/4-size image must be FLAT.  The zone plate's local frequency is ~0.00085*r, so the
#     x-frequency passes the new Nyquist (1/8) at |x| ~ 147 px from the center, and likewise for y.  An ideal result keeps only
#     the central square |x|,|y| < 147 and is flat everywhere else.  I use |x'| >= 45 or |y'| >= 45 in shrunk pixels
#     (original 180 px, f >= ~0.15) so the region sits safely outside that square.
h = rings.shape[0] // FACTOR
yy, xx = np.mgrid[0:h, 0:h]
region = (np.abs(xx - (h - 1) / 2) >= 180 / FACTOR) | (np.abs(yy - (h - 1) / 2) >= 180 / FACTOR)


def false_pattern_strength(img):
    """RMS deviation from the mean in `region`, as a fraction of a full-amplitude ring pattern (0.354)."""
    return float(np.sqrt(np.mean((img[region] - 0.5) ** 2)) / 0.3536)


naive = downsample(rings, FACTOR)
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
print(f"original {rings.shape} -> shrunk {naive.shape}; region for the false-circle measure: {region.sum()} px, "
      f"where the ideal result has strength {false_pattern_strength(ideal):.4f}")
print("EXPERIMENT: strength of the false pattern (1.00 = full-amplitude rings, 0 = flat gray) and RMSE to the ideal result")
print(f"{'sigma':>6} {'strength':>9} {'RMSE vs ideal':>14}")
print(f"{'none':>6} {false_pattern_strength(naive):9.3f} {rmse(naive, ideal):14.4f}")
sigmas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0]
results = {}
for s in sigmas:
    results[s] = blur_then_downsample(rings, FACTOR, s)
    print(f"{s:>6} {false_pattern_strength(results[s]):9.3f} {rmse(results[s], ideal):14.4f}")

# --- contact sheet: every tile shown at 2x with nearest-neighbour (no extra resampling in the viewer)
tiles = [("no blur", naive)] + [(f"sigma={s}", results[s]) for s in sigmas] + [("ideal (check)", ideal)]
cols = 5; rows_ = int(np.ceil(len(tiles) / cols))
fig, ax = plt.subplots(rows_, cols, figsize=(cols * 3.2, rows_ * 3.5))
for a in ax.ravel(): a.axis("off")
for a, (t, im) in zip(ax.ravel(), tiles):
    a.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a.set_title(f"{t}   [{false_pattern_strength(im):.2f}]", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_2_2_sigma_sweep.png", dpi=100); plt.close(fig)

if len(sys.argv) > 1:                                                       # final figure for the chosen sigma
    S = float(sys.argv[1]); chosen = blur_then_downsample(rings, FACTOR, S)
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    items = [("original 512x512 (full size)", rings), ("every 4th pixel (no blur)", naive),
             (f"Gaussian sigma={S}, then every 4th pixel", chosen)]
    for j, (t, im) in enumerate(items):
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest" if im.shape[0] < 512 else "antialiased")
        ax[0, j].set_title(t); ax[0, j].axis("off")
    for j, (t, im) in enumerate([("original", rings), ("every 4th pixel", naive), (f"sigma={S}, then every 4th", chosen)]):
        L = log_fft_magnitude(im)
        ax[1, j].imshow(L, cmap="gray"); ax[1, j].set_title(f"log |FFT|, {t}"); ax[1, j].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_2_2_side_by_side.png", dpi=80); plt.close(fig)
    save_image(f"{OUT}/rings_naive_4x.png", naive); save_image(f"{OUT}/rings_blur_s{S}_4x.png", chosen)
    print(f"saved side-by-side for sigma={S}")
