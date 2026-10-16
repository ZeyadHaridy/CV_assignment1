"""Bonus - remove an object with seams, then grow the photo back to its original width by inserting seams.

The photo is broadway_tower.png and the object is the walker (the README's practice example: it does NOT count as 'a photo of
your own' -- for the report, run remove/enlarge on a photo of your own with the same functions).
"""
import os, time
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_image, save_image
from paths import IMAGES, RESULTS
from seam import carve, remove_object, enlarge_by_seams, enlarge_naive, draw_seams

OUT = RESULTS; os.makedirs(OUT, exist_ok=True)


def object_removal_pipeline(name, img, mask, out=OUT):
    H, W = img.shape[:2]
    t0 = time.perf_counter(); rem = remove_object(img, mask); t_rem = time.perf_counter() - t0
    k, reduced = rem["k"], rem["image"]
    assert reduced.shape == (H, W - k) + img.shape[2:]
    t0 = time.perf_counter(); grown = enlarge_by_seams(reduced, k); t_grow = time.perf_counter() - t0
    restored = grown["image"]
    assert restored.shape == img.shape, (restored.shape, img.shape)                  # back at exactly the original width
    naive = enlarge_naive(reduced, k)
    widest = int(mask.sum(axis=1).max())
    print(f"[{name}] {W}x{H}; mask {int(mask.sum())} px, widest row {widest} px -> k = {k} seams removed ({t_rem:.1f} s), "
          f"reduced to width {W-k}; grown back by {k} seams ({t_grow:.1f} s) -> width {restored.shape[1]}")
    cl = lambda a: np.clip(a, 0, 1)
    # ---- figure A: the three required pictures
    over = img.copy(); over[mask > 0] = 0.55 * over[mask > 0] + 0.45 * np.array([1.0, 0.0, 0.0])
    gm = restored.copy(); gm[grown["inserted"]] = (0.0, 0.9, 0.1)
    fig, ax = plt.subplots(1, 3, figsize=(21, 21 * H / (W + (W - k) + W) + 0.8), gridspec_kw=dict(width_ratios=[W, W - k, W]))
    for a, im, t in [(ax[0], over, "photo with the mask over the object (red)"),
                     (ax[1], cl(reduced), f"object removed with {k} seams (width {W-k})"),
                     (ax[2], cl(gm), f"width restored to {W} with {k} inserted seams (green)")]:
        a.imshow(cl(im)); a.set_title(t, fontsize=11); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/fig_bonus_{name}.png", dpi=70); plt.close(fig)
    # ---- figure B: the seams that removed the object, and a zoom on the object's location
    win = (slice(330, 420), slice(10, 140))
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.5), gridspec_kw=dict(width_ratios=[W / H * 1.0, 1.4, 1.4, 1.4]))
    ax[0].imshow(draw_seams(img, rem["seams"])); ax[0].set_title(f"the {k} removal seams on the original", fontsize=10)
    for a, im, t in [(ax[1], img, "original (same window)"), (ax[2], reduced, "object removed (same window)"), (ax[3], restored, "restored width (same window)")]:
        a.imshow(cl(im)[win], interpolation="nearest"); a.set_title(t, fontsize=10)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/fig_bonus_{name}_zoom.png", dpi=70); plt.close(fig)
    # ---- figure C: k seams from a copy  vs  the cheapest seam inserted k times
    cnt = lambda src: np.stack([np.bincount(src[i], minlength=W - k) for i in range(H)])
    c_ours, c_naive = cnt(grown["src"]), cnt(naive["src"])
    print(f"[{name}] how often a pixel of the reduced photo is written: k seams from a copy -> at most {c_ours.max()}x "
          f"({int((c_ours > 1).sum())} pixels written twice, {int((c_ours > 2).sum())} more than twice); "
          f"cheapest seam inserted {k} times -> up to {c_naive.max()}x ({int((c_naive > 1).sum())} duplicated, {int((c_naive > 2).sum())} more than twice)")
    print(f"[{name}] distinct columns of the reduced photo that get stretched (mean over rows): "
          f"k-from-copy {np.mean((c_ours > 1).sum(axis=1)):.1f}, repeated-cheapest {np.mean((c_naive > 1).sum(axis=1)):.1f}  (of k = {k})")
    fig, ax = plt.subplots(2, 1, figsize=(15, 15 * 2 * H / W + 1.0))
    ax[0].imshow(cl(restored)); ax[0].set_title(f"{k} seams found once on a copy, then inserted together", fontsize=11)
    ax[1].imshow(cl(naive["image"])); ax[1].set_title(f"the single cheapest seam inserted {k} times in a row", fontsize=11)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/fig_bonus_{name}_naive_vs_copy.png", dpi=70); plt.close(fig)
    for tag, im in [("mask", np.repeat(mask[..., None], 3, axis=2)), ("overlay", over), ("removed", cl(reduced)),
                    ("restored", cl(restored)), ("restored_marked", cl(gm)), ("restored_naive", cl(naive["image"]))]:
        save_image(f"{out}/bonus_{name}_{tag}.png", im)
    return dict(k=k, restored=restored, reduced=reduced, naive=naive["image"], grown=grown, rem=rem)


def enlargement_comparison(name, img, frac=0.30, out=OUT):
    """Supplementary experiment (NOT object removal): carve `img` by `frac` of its width with the finite-difference energy,
    then grow it back by the same number of columns two ways -- k seams found once on a copy, or the cheapest seam inserted k times."""
    H, W = img.shape[:2]; k = int(round(frac * W)); Wr = W - k
    reduced = carve(img, k)["image"]
    t0 = time.perf_counter(); ours = enlarge_by_seams(reduced, k); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); naive = enlarge_naive(reduced, k); t2 = time.perf_counter() - t0
    assert ours["image"].shape == naive["image"].shape == (H, W, 3)
    cnt = lambda src: np.stack([np.bincount(src[i], minlength=Wr) for i in range(H)])
    co, cn = cnt(ours["src"]), cnt(naive["src"])
    print(f"[{name}] enlargement test {Wr} -> {W} (k = {k}): k seams from a copy {t1:.1f} s, cheapest seam repeated {t2:.1f} s; "
          f"a pixel is written at most {co.max()}x (ours) vs {cn.max()}x (naive); written more than twice: {int((co > 2).sum())} vs {int((cn > 2).sum())}; "
          f"distinct stretched columns per row: {np.mean((co > 1).sum(axis=1)):.1f} vs {np.mean((cn > 1).sum(axis=1)):.1f}")
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw=dict(width_ratios=[Wr, W, W]))
    for a, im, t in [(ax[0], reduced, f"reduced ({Wr} wide)"), (ax[1], ours["image"], f"{k} seams from a copy, inserted together"),
                     (ax[2], naive["image"], f"cheapest seam inserted {k} times")]:
        a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=11); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/fig_bonus_naive_vs_copy_{name}.png", dpi=70); plt.close(fig)


if __name__ == "__main__":
    img = load_image(f"{IMAGES}/part3_seams/broadway_tower.png").astype(np.float64)
    m = np.zeros(img.shape[:2], dtype=np.uint8)
    poly = np.array([(59, 360), (68, 360), (70, 368), (72, 376), (73, 384), (73, 393), (72, 401),
                     (57, 401), (56, 393), (55, 386), (54, 381), (56, 375), (57, 367)], dtype=np.int32)
    cv2.fillPoly(m, [poly], 1)                                                      # mask drawn in code, sharp edge
    print(f"mask: cv2.fillPoly with {len(poly)} vertices around the walker (x 54-73, y 360-401)")
    object_removal_pipeline("tower_walker", img, m.astype(np.float64))
    enlargement_comparison("starry", load_image(f"{IMAGES}/part3_seams/starry_night.jpg").astype(np.float64))
