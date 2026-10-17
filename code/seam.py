"""Part 3.1 - seam carving from scratch (Avidan & Shamir, 2007).

The seam-finding dynamic program, seam removal and coordinate tracking are all written here.
Energies reuse OUR Part 1 filters (finite differences / DoG through conv2d_separable).
No seam-carving library is used.  (cv2.resize is used ONLY in the comparison script, never here.)
"""
import numpy as np
from edges import finite_difference_gradient, dog_gradient
from freq import to_gray


# ------------------------------------------------------------------ energy functions
def energy_fd(image, pad_mode="reflect"):
    """Gradient magnitude from the finite differences Dx = [1 0 -1], Dy = Dx^T of 1.2 (on the grayscale image).
    pad_mode defaults to 'reflect': zero padding would put a strong false 'edge' on the image frame (see 1.2)."""
    return finite_difference_gradient(to_gray(image), pad_mode)[2]


def energy_dog(image, sigma=2.0, pad_mode="reflect"):
    """Gradient magnitude from the derivative-of-Gaussian filters of 1.3 (blur folded into the derivative)."""
    return dog_gradient(to_gray(image), sigma, pad_mode=pad_mode)[2]


# ------------------------------------------------------------------ dynamic programming
def cumulative_energy(energy):
    """M[i, j] = energy[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])   (neighbors outside the image ignored).

    Also returns `back`, with back[i, j] in {-1, 0, +1}: the column offset of the best predecessor in row i-1.
    Ties are broken in the order  straight up, up-left, up-right  (so flat regions give straight seams).
    The loop runs over ROWS only; each row is processed for all columns at once.
    """
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 2 or energy.size == 0:
        raise ValueError(f"energy must be a non-empty 2D array, got shape {energy.shape}")
    H, W = energy.shape
    M = energy.copy()
    back = np.zeros((H, W), dtype=np.int8)
    offsets = np.array([0, -1, 1], dtype=np.int8)
    cols = np.arange(W)
    for i in range(1, H):
        prev = M[i - 1]
        up_left = np.concatenate(([np.inf], prev[:-1]))        # predecessor at column j-1
        up_right = np.concatenate((prev[1:], [np.inf]))        # predecessor at column j+1
        candidates = np.stack([prev, up_left, up_right])       # (3, W)
        k = np.argmin(candidates, axis=0)                      # first minimum wins ties
        M[i] += candidates[k, cols]
        back[i] = offsets[k]
    return M, back


def find_vertical_seam(energy):
    """Cheapest top-to-bottom seam. Returns (seam, cost): seam[i] = column of the seam in row i."""
    M, back = cumulative_energy(energy)
    H, W = M.shape
    seam = np.empty(H, dtype=np.int64)
    j = int(np.argmin(M[-1]))                                  # cheapest end point in the bottom row (leftmost on ties)
    cost = float(M[-1, j])
    seam[-1] = j
    for i in range(H - 1, 0, -1):                              # trace back, one row at a time
        j += int(back[i, j])
        seam[i - 1] = j
    return seam, cost


def find_vertical_seam_loop(energy):
    """Plain-Python version of the same dynamic program (every pixel and predecessor in explicit loops).
    Used to check the row-vectorized version; returns (seam, cost)."""
    energy = np.asarray(energy, dtype=np.float64)
    H, W = energy.shape
    M = energy.copy()
    back = np.zeros((H, W), dtype=np.int64)
    for i in range(1, H):
        for j in range(W):
            best, best_off = M[i - 1, j], 0                    # straight up first (same tie order)
            if j > 0 and M[i - 1, j - 1] < best:
                best, best_off = M[i - 1, j - 1], -1
            if j < W - 1 and M[i - 1, j + 1] < best:
                best, best_off = M[i - 1, j + 1], 1
            M[i, j] += best
            back[i, j] = best_off
    j = int(np.argmin(M[-1])); cost = float(M[-1, j])
    seam = [j]
    for i in range(H - 1, 0, -1):
        j += int(back[i, j]); seam.append(j)
    return np.array(seam[::-1], dtype=np.int64), cost


# ------------------------------------------------------------------ removal and drawing
def remove_vertical_seam(array, seam):
    """Remove one pixel per row (all channels).  array: (H, W) or (H, W, C); seam: (H,) columns."""
    H, W = array.shape[:2]
    if len(seam) != H or W < 2:
        raise ValueError(f"seam of length {len(seam)} does not fit array {array.shape} (or width < 2)")
    keep = np.ones((H, W), dtype=bool)
    keep[np.arange(H), seam] = False
    return array[keep].reshape((H, W - 1) + array.shape[2:])


def draw_seams(image, seams, color=(1.0, 0.0, 0.0), direction="vertical"):
    """Copy of `image` (float, (H, W) or (H, W, 3)) with every seam pixel painted `color`.
    seams: (n, H) columns for vertical seams, or (n, W) rows for horizontal seams, in ORIGINAL coordinates."""
    out = np.array(image, dtype=np.float64)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    H, W = out.shape[:2]
    seams = np.asarray(seams)
    for s in seams:
        if direction == "vertical":
            out[np.arange(H), s] = color
        else:
            out[s, np.arange(W)] = color
    return out


# ------------------------------------------------------------------ carving
def carve(image, n_remove, energy_fn=energy_fd, direction="vertical"):
    """Remove `n_remove` seams one at a time, recomputing the energy after every removal.

    direction="vertical":   removes columns -> width becomes W - n_remove.
    direction="horizontal": removes rows    -> height becomes H - n_remove (the image is transposed, carved, transposed back).
    Returns a dict:
      image   : the carved image
      seams   : (n_remove, H) array [vertical] or (n_remove, W) [horizontal]: where each seam sat in the ORIGINAL image
      costs   : mean energy per pixel of each removed seam (measured at the moment it was removed)
    The input image is never modified.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError(f"image must be (H, W) or (H, W, C) and non-empty, got shape {image.shape}")
    if direction not in ("vertical", "horizontal"):
        raise ValueError("direction must be 'vertical' or 'horizontal'")
    if direction == "horizontal":
        t = (1, 0) + tuple(range(2, image.ndim))
        res = carve(image.transpose(t), n_remove, energy_fn, "vertical")
        res["image"] = res["image"].transpose(t)
        return res
    if int(n_remove) != n_remove or n_remove < 0:
        raise ValueError(f"n_remove must be a non-negative integer, got {n_remove}")
    n_remove = int(n_remove)
    H, W = image.shape[:2]
    if n_remove > W - 1:
        raise ValueError(f"cannot remove {n_remove} seams from width {W}: at least one column must remain")

    current = image.copy()
    cols = np.tile(np.arange(W), (H, 1))                       # original column of every remaining pixel
    rows = np.arange(H)
    seams = np.empty((n_remove, H), dtype=np.int64)
    costs = np.empty(n_remove)
    for k in range(n_remove):
        seam, cost = find_vertical_seam(energy_fn(current))   # energy RECOMPUTED on the current image
        seams[k] = cols[rows, seam]                            # where this seam sat in the original
        costs[k] = cost / H
        current = remove_vertical_seam(current, seam)          # all channels
        cols = remove_vertical_seam(cols, seam)                # the same seam removed from the coordinate array
    return dict(image=current, seams=seams, costs=costs)


def carve_to_width(image, target_width, energy_fn=energy_fd):
    """Convenience: carve vertical seams until the width is exactly `target_width` (checked)."""
    W = np.asarray(image).shape[1]
    if int(target_width) != target_width or not 1 <= target_width <= W:
        raise ValueError(f"target width must be an integer in [1, {W}], got {target_width}")
    res = carve(image, W - int(target_width), energy_fn)
    assert res["image"].shape[1] == target_width
    return res


def seam_search_space(H, W):
    """(lower, upper) bounds on the number of distinct vertical seams, as log10, for an H x W image:
    each step has 2 (at a border) or 3 choices, so W*2^(H-1) <= #seams <= W*3^(H-1)."""
    return np.log10(W) + (H - 1) * np.log10(2), np.log10(W) + (H - 1) * np.log10(3)


# =====================================================================================================
# Bonus - object removal (Section 4.6 of the paper) and seam insertion (Section 4.3)
# =====================================================================================================
def remove_object(image, mask, energy_fn=energy_fd, penalty=-1.0e6):
    """Remove the masked object by repeatedly removing the cheapest seam, with the energy of masked pixels set to a
    huge NEGATIVE value (so the cheapest seam goes through the mask).  Repeats until no masked pixel is left.

    The energy is computed in float64 on the CURRENT image after every removal; the mask is shrunk with the same seam.
    Returns dict(image = photo without the object, k = number of seams removed, seams = (k, H) original column of each
    removed seam, so they can be drawn on the original).
    """
    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask)
    if image.ndim not in (2, 3) or mask.shape != image.shape[:2]:
        raise ValueError(f"mask {mask.shape} must have the same height and width as the image {image.shape}")
    m = mask > 0.5
    H, W = image.shape[:2]
    current, cols, rows = image.copy(), np.tile(np.arange(W), (H, 1)), np.arange(H)
    seams = []
    while m.any():
        if current.shape[1] < 2:
            raise ValueError("the mask is too wide: removing it would leave no image")
        e = np.asarray(energy_fn(current), dtype=np.float64)
        e[m] = penalty                                          # cheapest seams now run through the object
        seam, _ = find_vertical_seam(e)
        seams.append(cols[rows, seam])
        current = remove_vertical_seam(current, seam)
        m = remove_vertical_seam(m, seam)                       # the mask loses the same pixels
        cols = remove_vertical_seam(cols, seam)
    return dict(image=current, k=len(seams), seams=np.array(seams, dtype=np.int64).reshape(len(seams), H))


def insert_seams(image, seams):
    """Widen `image` by k = len(seams) columns: every pixel is written once, but each pixel on a seam is written TWICE --
    the first copy averaged with its LEFT neighbor, the second with its RIGHT neighbor (a missing neighbor at the border
    is replaced by the pixel itself).  Neighbors are taken from `image`, not from the output.

    seams: (k, H) integer array, columns in `image`; no pixel may be used by two seams.
    Returns (out, src, inserted): the widened image, src[i, j] = column of `image` that out[i, j] comes from, and a boolean
    mask of the pixels that belong to inserted seams (both copies).
    """
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape[:2]
    seams = np.asarray(seams, dtype=np.int64).reshape(-1, H)
    k = seams.shape[0]
    if seams.size and (seams.min() < 0 or seams.max() >= W):
        raise ValueError("seam column outside the image")
    on_seam = np.zeros((H, W), dtype=bool)
    for s in seams:
        if on_seam[np.arange(H), s].any():
            raise ValueError("two seams use the same pixel")
        on_seam[np.arange(H), s] = True
    reps = 1 + on_seam.astype(np.int64)                                         # 2 for seam pixels, else 1
    rows = np.arange(H)[:, None]
    src = np.repeat(np.tile(np.arange(W), (H, 1)).ravel(), reps.ravel()).reshape(H, W + k)
    inserted = np.repeat(on_seam.ravel(), reps.ravel()).reshape(H, W + k)
    second = np.zeros((H, W + k), dtype=bool)
    second[:, 1:] = inserted[:, 1:] & (src[:, 1:] == src[:, :-1])               # the second of a pair of identical sources
    first = inserted & ~second
    left, right = np.maximum(src - 1, 0), np.minimum(src + 1, W - 1)            # missing neighbor -> the pixel itself
    pix = image[rows, src]
    pick = lambda a: a[..., None] if image.ndim == 3 else a
    out = np.where(pick(first), 0.5 * (pix + image[rows, left]),
                   np.where(pick(second), 0.5 * (pix + image[rows, right]), pix))
    return out, src, inserted


def enlarge_by_seams(reduced, k, energy_fn=energy_fd):
    """Grow `reduced` back by k columns.  The k seams are found ONCE, by removing k seams from a COPY of the photo (the copy
    is only used to find them; the coordinate tracking of carve() tells where each seam sat in `reduced`), then all k are
    inserted at the same time.  Returns dict(image, seams, inserted)."""
    found = carve(reduced, k, energy_fn)                                        # works on a copy; `reduced` is untouched
    out, src, inserted = insert_seams(reduced, found["seams"])
    return dict(image=out, seams=found["seams"], inserted=inserted, src=src)


def enlarge_naive(reduced, k, energy_fn=energy_fd):
    """For comparison ONLY: insert the single cheapest seam, k times in a row, recomputing the energy on the growing image."""
    cur = np.asarray(reduced, dtype=np.float64)
    H, W = cur.shape[:2]
    src = np.tile(np.arange(W), (H, 1))                                         # which ORIGINAL column each pixel descends from
    rows = np.arange(H)[:, None]
    for _ in range(k):
        seam, _ = find_vertical_seam(energy_fn(cur))
        cur, s, _ = insert_seams(cur, seam[None, :])
        src = src[rows, s]
    return dict(image=cur, src=src)
