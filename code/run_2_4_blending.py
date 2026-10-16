"""Part 2.4 - multiresolution blending: oraple (required) + hand/eye (practice, does not count).

`blend_pair` is reusable:  blend_pair(name, im1, im2, mask, levels)   (im2 already aligned to im1; mask = 1 where im1 shows)
"""
import os
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_image, save_image
from paths import IMAGES, RESULTS
from align_images import align_images
from blend import multires_blend, hard_cut, effective_alpha

OUT = RESULTS; os.makedirs(OUT, exist_ok=True)
D = IMAGES + "/part2_frequencies/"
ORAPLE_LEVELS, EYE_LEVELS = 6, 5


def _panel(a, im, title, fs=12):
    """imshow with a thin gray outline, so a WHITE region (e.g. a white mask half) is visible on the white page."""
    a.imshow(np.clip(im, 0, 1), cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
    a.set_title(title, fontsize=fs); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor("0.45"); sp.set_linewidth(1.0)


def blend_pair(name, im1, im2, mask, levels, titles=("image 1", "image 2"), out=OUT):
    """Figure with the two inputs, the mask, the hard cut and the multiresolution blend; saves the blend as a PNG."""
    hard = hard_cut(im1, im2, mask)
    blend = multires_blend(im1, im2, mask, levels)
    clipped = 100 * np.mean((blend < 0) | (blend > 1))
    print(f"[{name}] {levels} levels; blend range [{blend.min():.3f}, {blend.max():.3f}], {clipped:.2f}% of values clipped for display/saving")
    h, w = im1.shape[:2]
    fig, ax = plt.subplots(1, 5, figsize=(5 * 4.4, 4.4 * h / w + 0.7))
    for a, im, t in zip(ax, [im1, im2, mask, hard, blend],
                        [titles[0], titles[1], "mask", "hard cut", f"multiresolution blend ({levels} levels)"]):
        _panel(a, im, t)
    plt.tight_layout(); plt.savefig(f"{out}/fig_2_4_{name}.png", dpi=80); plt.close(fig)
    save_image(f"{out}/blend_{name}.png", np.clip(blend, 0, 1))
    return blend


def depth_sweep(name, im1, im2, mask, level_list, crop, out=OUT):
    ys, xs = crop
    cols = 4; rows_ = int(np.ceil(len(level_list) / cols))
    fig, ax = plt.subplots(rows_, cols, figsize=(cols * 4.2, rows_ * 4.2 * (ys.stop - ys.start) / (xs.stop - xs.start) + 0.5))
    for a in ax.ravel(): a.axis("off")
    for a, lv in zip(ax.ravel(), level_list):
        a.imshow(np.clip(multires_blend(im1, im2, mask, lv), 0, 1)[ys, xs]); a.set_title(f"{lv} level(s)", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{out}/fig_2_4_{name}_depth.png", dpi=80); plt.close(fig)


def level_by_level(name, im1, im2, mask, levels, out=OUT):
    """For each level: M, M*L1, (1-M)*L2 and their sum.  Laplacian-type levels shown as 0.5 + gain*x (zero = mid-gray)."""
    result, p = multires_blend(im1, im2, mask, levels, return_parts=True)
    gain = 0.5 / np.percentile(np.concatenate([np.abs(c).ravel() for c in p["combined"][:-1]]), 99.5)
    fig, ax = plt.subplots(levels, 4, figsize=(16, 3.9 * levels))
    heads = ["mask level  M", "M * L1", "(1 - M) * L2", "sum"]
    for i in range(levels):
        last = (i == levels - 1)
        show = [p["GM"][i], p["A"][i], p["B"][i], p["combined"][i]]
        for j, x in enumerate(show):
            d = x if (j == 0 or last) else 0.5 + gain * x
            _panel(ax[i, j], d, f"level {i} {x.shape[:2]}  {heads[j]}" + ("  (smallest Gaussian level, unchanged)" if last and j else ""), fs=9)
    plt.tight_layout(); plt.savefig(f"{out}/fig_2_4_{name}_levels.png", dpi=60); plt.close(fig)
    print(f"[{name}] level-by-level figure: Laplacian levels shown as 0.5 + {gain:.2f} * x, clipped to [0, 1]")
    return result


if __name__ == "__main__":
    # ================= oraple: apple on the left (mask = 1 on the left half), orange on the right =================
    apple, orange = load_image(D + "apple.jpg").astype(np.float64), load_image(D + "orange.jpg").astype(np.float64)
    mask = np.zeros(apple.shape[:2]); mask[:, : apple.shape[1] // 2] = 1.0
    print(f"[oraple] images {apple.shape}; mask: left half = 1 (white), right half = 0")
    print("[oraple] seam width (pixels where image 1 weighs 10%..90%) vs levels:",
          {lv: int(((effective_alpha(mask, lv)[256] > .1) & (effective_alpha(mask, lv)[256] < .9)).sum()) for lv in range(1, 9)})
    blend_pair("oraple", apple, orange, mask, ORAPLE_LEVELS, ("apple", "orange"))
    depth_sweep("oraple", apple, orange, mask, list(range(1, 9)), (slice(0, 512), slice(96, 416)))
    level_by_level("oraple", apple, orange, mask, ORAPLE_LEVELS)

    # ================= hand + eye (practice pair; irregular elliptical mask) =================
    hand, eye = load_image(D + "hand.jpg").astype(np.float64), load_image(D + "eye.jpg")
    eye_al = align_images(hand, eye, [(170, 400), (390, 410)], [(165, 255), (470, 265)]).astype(np.float64)
    angle = float(np.degrees(np.arctan2(410 - 400, 390 - 170)))                    # follow the line between the two palm points
    emask = np.zeros(hand.shape[:2], dtype=np.uint8)
    cv2.ellipse(emask, (265, 395), (115, 52), angle, 0, 360, 1, -1)                 # filled ellipse, SHARP edge
    emask = emask.astype(np.float64)
    print(f"[hand/eye] mask: cv2.ellipse center (265,395), semi-axes (115,52), angle {angle:.1f} deg; {int(emask.sum())} px = {100*emask.mean():.1f}% of the image")
    print("[hand/eye] eye 'wash-out' vs levels: effective weight of the eye at the ellipse center / mean over the mask / seam width at the center row")
    for lv in range(1, 9):
        al = effective_alpha(emask, lv)
        print(f"   {lv} levels: center {al[395, 265]:.3f}   mean in mask {al[emask > 0].mean():.3f}   width {int(((al[395] > .1) & (al[395] < .9)).sum())} px")
    # image 1 = eye (mask = 1 inside the ellipse), image 2 = hand
    blend_pair("handeye", eye_al, hand, emask, EYE_LEVELS, ("eye (aligned)", "hand"))
    depth_sweep("handeye", eye_al, hand, emask, list(range(1, 9)), (slice(250, 560), slice(60, 520)))
