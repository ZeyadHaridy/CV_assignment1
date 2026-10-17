"""Runs ONLY the parts that use YOUR OWN images (images/own/). Run from the code/ folder:  python run_own.py
Outputs go to ../results/ ."""
import matplotlib; matplotlib.use("Agg")
import os, sys, runpy, subprocess, shutil
import numpy as np, cv2
Image=display=lambda *a,**k:None
sys.path.insert(0, os.getcwd())

IMG, RES = "../images", "../results"
OWN = IMG + "/own"                       # put YOUR images here (<= 1000 px on the longer side)
os.makedirs(RES, exist_ok=True); os.makedirs(OWN, exist_ok=True)

# ---- switches: set to False to skip a section (1.1 and 3.x are the slow ones) ----
RUN = dict(tests=False, bench_1_1=True, edges_1_2_1_3=True, hybrids_2_1=True, aliasing_2_2=True,
           pyramids_2_3=True, blends_2_4=True, seams_3_1=True, compare_3_2=True, bonus=True)

# ---- name of each own image (change them here if you use other file names) ----
OWN_FILES = dict(
    hybrid1_low=OWN+"/hybrid1_low.jpg",  hybrid1_high=OWN+"/hybrid1_high.jpg",
    hybrid2_low=OWN+"/hybrid2_low.jpg",  hybrid2_high=OWN+"/hybrid2_high.jpg",
    blend1_base=OWN+"/blend1_base.jpg",  blend1_obj=OWN+"/blend1_obj.jpg",
    blend2_base=OWN+"/blend2_base.jpg",  blend2_obj=OWN+"/blend2_obj.jpg",     # blend2_obj = the KITE photo (converted to .jpg)
    seam_person=OWN+"/seam_person.jpg",  seam_landscape=OWN+"/seam_landscape.jpg", seam_fail=OWN+"/seam_fail.jpg",
    bonus_photo=OWN+"/bonus_photo.jpg",
)

def show_png(*names, width=None):
    """Display figures saved by the scripts (names relative to ../results, or full paths)."""
    for n in names:
        p = n if os.path.exists(n) else f"{RES}/{n}"
        if os.path.exists(p): display(Image(filename=p, width=width))
        else: print("missing figure:", p)

def have(*paths):
    miss = [p for p in paths if not os.path.exists(p)]
    if miss: print("SKIPPED - file(s) not found:", miss)
    return not miss

def run_script(name, *args):
    """Run one of the run_*.py scripts from this folder (they use ../images and ../results)."""
    old = sys.argv; sys.argv = [name, *map(str, args)]
    try: runpy.run_path(name)
    finally: sys.argv = old

def shrink_own_images(max_side=1000):
    """Resize every image in ../images/own so its longer side is <= max_side (originals backed up in ../images/own_original)."""
    bak = IMG + "/own_original"; os.makedirs(bak, exist_ok=True)
    for f in sorted(os.listdir(OWN)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
        p = f"{OWN}/{f}"; im = cv2.imread(p)
        if im is None: continue
        s = max(im.shape[:2])
        if s > max_side:
            if not os.path.exists(f"{bak}/{f}"): shutil.copy(p, f"{bak}/{f}")
            im = cv2.resize(im, None, fx=max_side/s, fy=max_side/s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(p, im); print(f"{f}: shrunk to {im.shape[1]}x{im.shape[0]}")
        else:
            print(f"{f}: {im.shape[1]}x{im.shape[0]} (ok)")

def grid(path, step=20):
    """Show an image with a pixel grid so you can read (x, y) coordinates for the alignment points / mask polygons."""
    get_ipython().run_line_magic("matplotlib", "inline")
    from align_images import show_with_grid
    from utils import load_image
    show_with_grid(load_image(path), step)

shrink_own_images()

from run_2_1_hybrid import process_pair
D2 = IMG + "/part2_frequencies/"
HYBRIDS = [
    dict(name="man_cat", low=D2+"hybrid_man.jpg", high=D2+"hybrid_cat.jpg",
         pts_low=[(299, 341), (443, 332)], pts_high=[(605, 290), (760, 372)], sigma_low=6, sigma_high=4),
    # OWN: fill pts_low / pts_high (two points each) and tune the sigmas
    # own1: LOW = the face with glasses (hybrid1_high.jpg, seen from far), HIGH = the other face (hybrid1_low.jpg, seen up close).
    #       (the roles are swapped on purpose: the HIGH image gets enlarged by align_images, so no empty borders appear)
    dict(name="own1", low=OWN_FILES["hybrid1_high"], high=OWN_FILES["hybrid1_low"],
         pts_low=[(310, 452), (477, 455)], pts_high=[(316, 401), (430, 402)], sigma_low=8, sigma_high=3),
    # own2: LOW = the car (hybrid2_high.jpg), HIGH = the face (hybrid2_low.jpg). The face matches the car's headlights (eyes <-> headlights).
    #       This is the pair that may fail: explain why in the report (the shapes do not match).
    dict(name="own2", low=OWN_FILES["hybrid2_high"], high=OWN_FILES["hybrid2_low"],
         pts_low=[(115, 158), (418, 158)], pts_high=[(310, 452), (477, 455)], sigma_low=6, sigma_high=3),
]
if RUN["hybrids_2_1"]:
    for h in HYBRIDS[1:]:   # own pairs only (man_cat is run by run_all.py)
        if h["pts_low"] is None or h["pts_high"] is None:
            print(f"[{h['name']}] SKIPPED - set pts_low and pts_high first"); continue
        if not have(h["low"], h["high"]): continue
        process_pair(h["name"], h["low"], h["high"], h["pts_low"], h["pts_high"], h["sigma_low"], h["sigma_high"])

from run_2_4_blending import blend_pair, level_by_level, depth_sweep
from utils import load_image
from align_images import align_images, similarity_from_two_points

D2 = IMG + "/part2_frequencies/"

# Kite mask: two polygons drawn ON THE KITE IMAGE (object coordinates), then warped with the same transform as the image.
# These vertices are estimates: check them with the overlay in the notebook (cv2.polylines + show) and adjust.
KITE_BODY = [(329, 92), (405, 103), (374, 193), (295, 157)]                                   # diamond, slightly inside the edge
KITE_TAIL = [(378, 200), (392, 228), (420, 260), (455, 282), (487, 300), (480, 310),
             (445, 298), (415, 278), (395, 255), (380, 230), (372, 205)]                       # ribbons, rough

BLENDS = [  # pts_base / pts_obj: two matching points
            # poly     : mask vertices in BASE coordinates          (own1)
            # poly_obj : list of polygons in OBJECT coordinates, warped with the alignment transform  (own2)
    # own1: face (eyes) on the watermelon; irregular face-shaped mask (14 vertices)
    dict(name="own1", base=OWN_FILES["blend1_base"], obj=OWN_FILES["blend1_obj"],
         pts_base=[(245, 265), (355, 265)], pts_obj=[(310, 452), (477, 455)],
         poly=[(301, 98), (367, 107), (414, 155), (425, 228), (417, 307), (395, 386), (360, 446), (308, 477),
               (255, 451), (221, 396), (197, 324), (189, 245), (197, 179), (236, 122)],
         poly_obj=None, levels=5, favorite=True),
    # own2: the kite in the sky. pts_obj = top tip and bottom tip of the kite; pts_base keeps the same offsets (48, 107) -> scale 1.0.
    #       Move pts_base to put the kite elsewhere (keep the two points the same distance apart to keep its size).
    dict(name="own2", base=OWN_FILES["blend2_base"], obj=OWN_FILES["blend2_obj"],
         pts_base=[(250, 70), (298, 177)], pts_obj=[(328, 89), (376, 196)],
         poly=None, poly_obj=[KITE_BODY, KITE_TAIL], levels=3, favorite=False),
]
if RUN["blends_2_4"]:
    # --- oraple: apple (left half of the mask = white) + orange
    if have(D2+"apple.jpg", D2+"orange.jpg"):
        apple, orange = load_image(D2+"apple.jpg").astype(np.float64), load_image(D2+"orange.jpg").astype(np.float64)
        m = np.zeros(apple.shape[:2]); m[:, :apple.shape[1] // 2] = 1.0
        blend_pair("oraple", apple, orange, m, 6, ("apple", "orange"))
    # --- own blends
    for b in BLENDS:
        if b["pts_base"] is None or b["pts_obj"] is None or (b["poly"] is None and b["poly_obj"] is None):
            print(f"[{b['name']}] SKIPPED - set pts_base, pts_obj and poly / poly_obj first"); continue
        if not have(b["base"], b["obj"]): continue
        base = load_image(b["base"]).astype(np.float64)
        obj_raw = load_image(b["obj"])
        obj = align_images(base, obj_raw, b["pts_base"], b["pts_obj"]).astype(np.float64)
        if b["poly"] is not None:                                      # polygon given in BASE coordinates
            m = np.zeros(base.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.array(b["poly"], np.int32)], 1)        # sharp mask; its Gaussian pyramid softens it
        else:                                                          # polygons drawn on the object, then warped
            mo = np.zeros(obj_raw.shape[:2], np.uint8)
            for P in b["poly_obj"]:
                cv2.fillPoly(mo, [np.array(P, np.int32)], 1)
            M = similarity_from_two_points(b["pts_obj"], b["pts_base"])     # same transform align_images applies to the image
            m = cv2.warpAffine(mo, M, (base.shape[1], base.shape[0]), flags=cv2.INTER_NEAREST)
        m = m.astype(np.float64)
        blend_pair(b["name"], obj, base, m, b["levels"], ("object (aligned)", "background"))
        if b["name"] == "own2":                                        # use this figure to choose the number of levels
            depth_sweep("own2", obj, base, m, [1, 2, 3, 4, 5], (slice(20, 320), slice(180, 450)))
        if b["favorite"]:
            level_by_level(b["name"], obj, base, m, b["levels"])

from run_3_2_comparisons import compare
D3 = IMG + "/part3_seams/"
STARTER_3_2 = [("rocket", D3+"rocket.jpg"), ("home", D3+"home.jpg"), ("starry_night", D3+"starry_night.jpg")]
OWN_3_2 = [("own_person", OWN_FILES["seam_person"]), ("own_landscape", OWN_FILES["seam_landscape"]), ("own_fail", OWN_FILES["seam_fail"])]
STATS = {}
if RUN["compare_3_2"]:
    for name, path in OWN_3_2:
        if not have(path): continue
        STATS[name], _ = compare(name, load_image(path).astype(np.float64))

from run_bonus import object_removal_pipeline
BONUS = dict(name="own", path=OWN_FILES["bonus_photo"],      # the water bottle
             poly=[(172, 72), (238, 72), (239, 105), (236, 124), (248, 140), (263, 160), (264, 200), (264, 330), (262, 396),
                   (258, 418), (205, 422), (150, 416), (146, 390), (146, 330), (147, 200), (152, 160), (164, 140),
                   (180, 124), (178, 105), (172, 100)])
if RUN["bonus"]:
    if BONUS["poly"] is None:
        print("SKIPPED - set BONUS['poly'] (polygon around the object) first")
    elif have(BONUS["path"]):
        photo = load_image(BONUS["path"]).astype(np.float64)
        mk = np.zeros(photo.shape[:2], np.uint8)
        cv2.fillPoly(mk, [np.array(BONUS["poly"], np.int32)], 1)         # sharp mask, drawn in code
        mk = cv2.dilate(mk, np.ones((3, 3), np.uint8), iterations=3)    # grow the mask ~3 px: removes the blue cap speck and the bright streak at the shadow
        out = object_removal_pipeline(BONUS["name"], photo, mk.astype(np.float64))
        n = BONUS["name"]
        show_png(f"fig_bonus_{n}.png", f"fig_bonus_{n}_zoom.png", f"fig_bonus_{n}_naive_vs_copy.png")
print("\nDONE - own-image figures are in ../results")