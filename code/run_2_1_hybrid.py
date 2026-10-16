"""Part 2.1 - hybrid images.  `process_pair` is reusable for any pair of images:

    process_pair("name", path_low, path_high, pts_low, pts_high, sigma_low, sigma_high)

im_low supplies the LOW frequencies (Gaussian blur), im_high the HIGH frequencies (image - blur).
pts_* are two matching points (x, y) on each image, e.g. the eye centers. im_high is warped onto im_low.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_image, save_image
from paths import IMAGES, RESULTS
from align_images import align_images
from pyramids import reduce
from freq import hybrid_image, low_pass, high_pass, to_gray, log_fft_magnitude, gaussian_cutoff_frequency
from edges import signed_for_display

OUT = RESULTS
MISALIGN = (80, 60)          # (dx, dy) in pixels of the deliberate misalignment used only for the comparison figure
D = IMAGES + "/part2_frequencies/"


def _show(ax, im, title, vmin=0, vmax=1):
    ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="antialiased")
    ax.set_title(title, fontsize=10); ax.axis("off")


def process_pair(name, path_low, path_high, pts_low, pts_high, sigma_low, sigma_high, out=OUT):
    os.makedirs(out, exist_ok=True)
    im_low = load_image(path_low); im_high_orig = load_image(path_high)
    im_high = align_images(im_low, im_high_orig, pts_low, pts_high)       # second image warped onto the first
    g_low, g_high = to_gray(im_low), to_gray(im_high)

    lp = low_pass(g_low, sigma_low)                                       # our convolution, reflect padding
    hp = high_pass(g_high, sigma_high)
    hyb = lp + hp                                                         # float64, may leave [0, 1]
    print(f"[{name}] sigma_low={sigma_low} (half-amplitude cutoff {gaussian_cutoff_frequency(sigma_low):.4f} cyc/px), "
          f"sigma_high={sigma_high} (cutoff {gaussian_cutoff_frequency(sigma_high):.4f} cyc/px)")
    print(f"[{name}] hybrid range [{hyb.min():.3f}, {hyb.max():.3f}]; "
          f"{100*np.mean((hyb < 0) | (hyb > 1)):.2f}% of pixels clipped for display/saving")
    save_image(f"{out}/hybrid_{name}_gray.png", hyb)
    save_image(f"{out}/hybrid_{name}_color.png", hybrid_image(im_low, im_high, sigma_low, sigma_high))

    # ---- process figure: 6 images + log-magnitude FFT of each (FFT on grayscale, as the assignment says)
    s = np.percentile(np.abs(hp), 99.5)
    panels = [("first image (low-pass source)", to_gray(im_low), g_low),
              ("second image, original", to_gray(im_high_orig), to_gray(im_high_orig)),
              ("second image, aligned", g_high, g_high),
              (f"first, low-pass  (sigma={sigma_low})", np.clip(lp, 0, 1), lp),
              (f"second, high-pass  (sigma={sigma_high})", signed_for_display(hp, s), hp),
              ("hybrid", np.clip(hyb, 0, 1), hyb)]
    ffts = [log_fft_magnitude(raw) for _, _, raw in panels]
    vmin, vmax = np.percentile(np.concatenate([f.ravel() for f in ffts]), 0.5), max(f.max() for f in ffts)
    fig, ax = plt.subplots(2, 6, figsize=(24, 9.5))
    for j, ((title, disp, _), F) in enumerate(zip(panels, ffts)):
        _show(ax[0, j], disp, title)
        _show(ax[1, j], F, "log |FFT|", vmin, vmax)
    plt.tight_layout(); plt.savefig(f"{out}/fig_2_1_{name}_process.png", dpi=80); plt.close(fig)

    # ---- distance figure: the hybrid at 1, 1/2, 1/4, 1/8 size, made with OUR reduce() from 2.3 (blur + keep every other
    #      row/column), only to SIMULATE what the eye sees from farther away
    shown = [np.clip(hyb, 0, 1)]
    for _ in range(3):
        shown.append(reduce(shown[-1]))
    H = shown[0].shape[0]; gap = 20
    canvas = np.full((H, sum(i.shape[1] for i in shown) + gap * 3), 0.85)
    x = 0
    for im in shown:
        canvas[:im.shape[0], x:x + im.shape[1]] = im; x += im.shape[1] + gap
    save_image(f"{out}/fig_2_1_{name}_distance.png", canvas)

    # ---- alignment figure: aligned vs deliberately MISALIGNED (same align_images call, but the second image's points are
    #      displaced by MISALIGN pixels), same sigmas
    bad_pts = [(x + MISALIGN[0], y + MISALIGN[1]) for (x, y) in pts_high]
    unaligned = to_gray(align_images(im_low, im_high_orig, pts_low, bad_pts))
    hyb_bad = low_pass(g_low, sigma_low) + high_pass(unaligned, sigma_high)
    fig, ax = plt.subplots(1, 4, figsize=(18, 6.5))
    _show(ax[0], 0.5 * g_low + 0.5 * g_high, "50/50 overlay, aligned")
    _show(ax[1], 0.5 * g_low + 0.5 * unaligned, f"50/50 overlay, misaligned by {MISALIGN} px")
    _show(ax[2], np.clip(hyb, 0, 1), "hybrid, aligned")
    _show(ax[3], np.clip(hyb_bad, 0, 1), "hybrid, misaligned")
    plt.tight_layout(); plt.savefig(f"{out}/fig_2_1_{name}_alignment.png", dpi=80); plt.close(fig)
    return hyb


if __name__ == "__main__":
    process_pair("man_cat", D + "hybrid_man.jpg", D + "hybrid_cat.jpg",
                 [(299, 341), (443, 332)], [(605, 290), (760, 372)], sigma_low=6, sigma_high=4)
