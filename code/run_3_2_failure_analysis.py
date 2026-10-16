"""Part 3.2 - the failure case: Starry Night with DoG energy squeezes the cypress.

Re-runs the comparison for starry_night (so this script is self-contained), measures how much of the dark cypress each energy
function removes, and draws the 2x2 failure figure.  The cypress is located by a ROUGH proxy: very dark pixels (gray < 0.22)
in the left 330 columns.  Treat the percentages as indicative, not exact.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_image
from paths import IMAGES, RESULTS
from freq import to_gray
from run_3_2_comparisons import compare

DARK, LEFT = 0.22, 330
img = load_image(f"{IMAGES}/part3_seams/starry_night.jpg").astype(np.float64)
stats, res = compare("starry_night", img)
H, W = img.shape[:2]
tree = to_gray(img) < DARK
tree[:, LEFT:] = False
print(f"cypress proxy (gray < {DARK}, columns < {LEFT}): {tree.sum()} px = {100 * tree.mean():.1f}% of the image")
for key, label in (("fd", "finite differences"), ("dog", "DoG (sigma=2)")):
    removed = np.zeros((H, W), bool)
    for seam in res[key]["seams"]:
        removed[np.arange(H), seam] = True
    print(f"  {label:20s}: removed {100 * (removed & tree).sum() / tree.sum():.1f}% of the cypress pixels (a uniform 30% resize would remove 30.0%); "
          f"{100 * (removed & tree).sum() / removed.sum():.1f}% of ALL removed pixels lie on the cypress")
    for r in (350, 420, 500):
        w0 = int(tree[r].sum()); w1 = int((tree[r] & ~removed[r]).sum())
        print(f"      row {r}: cypress width {w0} px -> {w1} px  (x{w1 / max(w0, 1):.2f})")

panels = [(img, "original (720x568)"),
          (load_image(f"{RESULTS}/starry_night_resized.png"), "resized to 504 wide (uniform squeeze, 30%)"),
          (res["fd"]["image"], "seam carved, finite differences: cypress slimmer but recognizable"),
          (res["dog"]["image"], "seam carved, DoG (sigma=2): cypress reduced to a sliver")]
fig, ax = plt.subplots(2, 2, figsize=(14, 12.2))
for a, (im, t) in zip(ax.ravel(), panels):
    a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=11); a.axis("off")
plt.tight_layout(); plt.savefig(f"{RESULTS}/fig_3_2_failure_starry_night.png", dpi=75); plt.close(fig)
print("saved fig_3_2_failure_starry_night.png")
