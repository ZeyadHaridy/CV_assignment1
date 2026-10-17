"""Part 3.2 - resizing vs center crop vs seam carving (two energy functions), 30% narrower.

ALLOWED library use here (comparison only): cv2.resize for the 'normal resizing' column.  The center crop is plain slicing.
The seam-carving columns use only our own code from seam.py.
"""
import os, sys, time
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from utils import load_image, save_image
from paths import IMAGES, RESULTS
from seam import carve, energy_fd, energy_dog, draw_seams

OUT = RESULTS; os.makedirs(OUT, exist_ok=True)
D = IMAGES + "/part3_seams/"
SIGMA = 2.0
fd, dog = energy_fd, partial(energy_dog, sigma=SIGMA)


def compare(name, img, frac=0.30, out=OUT):
    """Run all three methods (+ both energies) on `img`; save the two figures; return numbers for the report."""
    H, W = img.shape[:2]
    n = int(round(frac * W)); tw = W - n
    t0 = time.perf_counter()
    r_fd = carve(img, n, fd); r_dog = carve(img, n, dog)
    secs = time.perf_counter() - t0
    resized = cv2.resize(img, (tw, H), interpolation=cv2.INTER_AREA)               # normal resizing (allowed, comparison only)
    left = (W - tw) // 2; cropped = img[:, left:left + tw]                          # center crop
    for im in (resized, cropped, r_fd["image"], r_dog["image"]):
        assert im.shape == (H, tw, 3), (name, im.shape)                              # exact target width for every method

    ws = [W, tw, tw, tw, tw]
    fig, ax = plt.subplots(1, 5, figsize=(20, 20 * H / sum(ws) + 0.7), gridspec_kw=dict(width_ratios=ws))
    for a, im, t in zip(ax, [img, resized, cropped, r_fd["image"], r_dog["image"]],
                        [f"original ({W}x{H})", f"resized ({tw}x{H})", f"center crop ({tw}x{H})",
                         "seam carved, finite differences", f"seam carved, DoG (sigma={SIGMA:g})"]):
        a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/fig_3_2_{name}_compare.png", dpi=70); plt.close(fig)

    e_fd, e_dog = fd(img), dog(img)
    fig, ax = plt.subplots(2, 2, figsize=(16, 16 * H / W * 1.0 + 0.8))
    for a, im, t in [(ax[0, 0], np.clip(e_fd / np.percentile(e_fd, 99.5), 0, 1), "energy: finite differences"),
                     (ax[0, 1], np.clip(e_dog / np.percentile(e_dog, 99.5), 0, 1), f"energy: DoG (sigma={SIGMA:g})"),
                     (ax[1, 0], draw_seams(img, r_fd["seams"]), f"{n} seams chosen with finite differences"),
                     (ax[1, 1], draw_seams(img, r_dog["seams"], color=(0.0, 0.2, 1.0)), f"{n} seams chosen with DoG")]:
        a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t, fontsize=11); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/fig_3_2_{name}_seams.png", dpi=70); plt.close(fig)
    for tag, im in [("resized", resized), ("cropped", cropped), ("carved_fd", r_fd["image"]), ("carved_dog", r_dog["image"])]:
        save_image(f"{out}/{name}_{tag}.png", np.clip(im, 0, 1))

    mask_fd = np.zeros((H, W), bool); mask_fd[np.arange(H), r_fd["seams"]] = True
    mask_dog = np.zeros((H, W), bool); mask_dog[np.arange(H), r_dog["seams"]] = True
    same = np.logical_and(mask_fd, mask_dog).sum() / np.logical_or(mask_fd, mask_dog).sum()
    stats = dict(name=name, W=W, H=H, n=n, tw=tw, secs=secs,
                 cost_fd=float(r_fd["costs"].mean()) / float(e_fd.mean()), cost_dog=float(r_dog["costs"].mean()) / float(e_dog.mean()),
                 last_fd=float(r_fd["costs"][-1]) / float(e_fd.mean()), last_dog=float(r_dog["costs"][-1]) / float(e_dog.mean()),
                 jaccard=float(same))
    print(f"[{name}] {W}x{H} -> width {tw} ({n} seams, {secs:.0f} s for both energies) | mean seam energy / image mean energy: "
          f"FD {stats['cost_fd']:.2f} (last seam {stats['last_fd']:.2f}), DoG {stats['cost_dog']:.2f} (last {stats['last_dog']:.2f}) | "
          f"FD/DoG removed-pixel overlap (Jaccard) {stats['jaccard']:.2f}")
    return stats, dict(fd=r_fd, dog=r_dog)


if __name__ == "__main__":
    names = sys.argv[1:] or ["rocket", "home", "starry_night", "broadway_tower"]
    for nm in names:
        ext = ".png" if nm == "broadway_tower" else ".jpg"
        compare(nm, load_image(D + nm + ext).astype(np.float64))
