"""Tests for seam.py. Run from code/:   python tests/test_seam.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from seam import (cumulative_energy, find_vertical_seam, find_vertical_seam_loop, remove_vertical_seam, draw_seams,
                  carve, carve_to_width, energy_fd, energy_dog, seam_search_space)

rng = np.random.default_rng(31)

TINY = np.array([[3, 8, 5, 4],
                 [7, 1, 9, 6],
                 [2, 5, 3, 8],
                 [9, 4, 6, 1]], dtype=float)


def brute_force_min_cost(e):
    """Try EVERY seam (all column choices with |step| <= 1) and return the smallest total energy (tests only)."""
    H, W = e.shape
    best = np.inf
    def rec(i, j, total):
        nonlocal best
        total += e[i, j]
        if i == H - 1:
            best = min(best, total); return
        for dj in (-1, 0, 1):
            if 0 <= j + dj < W:
                rec(i + 1, j + dj, total)
    for j in range(W):
        rec(0, j, 0.0)
    return best


def test_tiny_example_by_hand():
    M, back = cumulative_energy(TINY)
    expected = np.array([[3, 8, 5, 4], [10, 4, 13, 10], [6, 9, 7, 18], [15, 10, 13, 8]], dtype=float)
    assert np.array_equal(M, expected), M
    seam, cost = find_vertical_seam(TINY)
    assert seam.tolist() == [0, 1, 2, 3] and cost == 8.0
    assert sum(TINY[i, j] for i, j in enumerate(seam)) == cost
    assert back[3, 3] == -1 and back[2, 2] == -1 and back[1, 1] == -1          # three up-left steps
    # a greedy walk (start at the smallest pixel of row 0, always take the cheapest neighbor) is WORSE: 3+1+2+4 = 10
    j = int(np.argmin(TINY[0])); total = TINY[0, j]
    for i in range(1, 4):
        nb = [c for c in (j - 1, j, j + 1) if 0 <= c < 4]
        j = min(nb, key=lambda c: TINY[i, c]); total += TINY[i, j]
    assert total == 10.0 > cost
    print("  tiny example: M =", expected.astype(int).tolist(), "-> seam", seam.tolist(), "cost", cost, "(greedy gives", total, ")")


def test_dp_equals_brute_force_on_random_matrices():
    n = 0
    for _ in range(300):
        H, W = rng.integers(1, 7, size=2)
        e = rng.integers(0, 10, size=(H, W)).astype(float)
        seam, cost = find_vertical_seam(e)
        assert cost == brute_force_min_cost(e), (e, cost)
        assert np.all(np.abs(np.diff(seam)) <= 1) and seam.min() >= 0 and seam.max() < W
        assert sum(e[i, seam[i]] for i in range(H)) == cost
        n += 1
    print(f"  DP cost equals the brute-force optimum on {n} random matrices (1-6 rows/columns)")


def test_vectorized_equals_plain_loop_version():
    for _ in range(60):
        H, W = rng.integers(1, 15, size=2)
        e = rng.random((H, W))
        s1, c1 = find_vertical_seam(e); s2, c2 = find_vertical_seam_loop(e)
        assert np.array_equal(s1, s2) and np.isclose(c1, c2)
    e = rng.integers(0, 3, size=(12, 9)).astype(float)                              # many ties
    assert np.array_equal(find_vertical_seam(e)[0], find_vertical_seam_loop(e)[0])


def test_ties_give_a_straight_seam():
    seam, cost = find_vertical_seam(np.ones((6, 5)))
    assert seam.tolist() == [0] * 6 and cost == 6.0


def test_seam_avoids_a_high_energy_wall():
    e = np.ones((8, 7)); e[:, 3] = 100.0                                            # a wall of strong edges in column 3
    seam, cost = find_vertical_seam(e)
    assert 3 not in seam and cost == 8.0


def test_borders_and_tiny_energy_matrices():
    assert find_vertical_seam(np.array([[5.0]]))[0].tolist() == [0]
    assert find_vertical_seam(np.array([[4.0, 2.0, 9.0]]))[0].tolist() == [1]       # one row: just the cheapest pixel
    s, c = find_vertical_seam(np.array([[1.0], [2.0], [3.0]]))                      # one column
    assert s.tolist() == [0, 0, 0] and c == 6.0
    for bad in (np.zeros(4), np.zeros((0, 3)), np.zeros((2, 2, 2))):
        try:
            find_vertical_seam(bad)
        except ValueError:
            continue
        raise AssertionError(bad.shape)


def test_remove_seam_gray_and_color():
    img = np.arange(12.0).reshape(3, 4)
    out = remove_vertical_seam(img, np.array([1, 2, 0]))
    assert out.tolist() == [[0, 2, 3], [4, 5, 7], [9, 10, 11]]
    col = rng.random((5, 6, 3)); seam = np.array([0, 1, 2, 3, 5])
    out = remove_vertical_seam(col, seam)
    assert out.shape == (5, 5, 3)
    for i in range(5):
        assert np.array_equal(out[i], np.delete(col[i], seam[i], axis=0))          # every channel, correct pixel
    try:
        remove_vertical_seam(img, np.array([0, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong seam length not rejected")


def test_carve_shape_tracking_and_no_input_change():
    img = rng.random((20, 30, 3)); before = img.copy()
    n = 7
    res = carve(img, n)
    assert np.array_equal(img, before)
    assert res["image"].shape == (20, 23, 3) and res["seams"].shape == (n, 20) and res["costs"].shape == (n,)
    # coordinate tracking: the carved image is exactly the original with the recorded seam pixels deleted
    removed = np.zeros(img.shape[:2], dtype=bool)
    for s in res["seams"]:
        assert not removed[np.arange(20), s].any()                                  # no pixel removed twice
        removed[np.arange(20), s] = True
    assert removed.sum() == n * 20 and np.all(removed.sum(axis=1) == n)             # exactly n per row
    rebuilt = np.stack([img[i][~removed[i]] for i in range(20)])
    assert np.array_equal(rebuilt, res["image"])
    print(f"  tracking: carved 20x30 -> 20x23; deleting the {n} recorded seams from the original reproduces it exactly")


def test_first_seam_is_the_cheapest_and_seams_are_connected_when_removed():
    img = rng.random((15, 12))
    res = carve(img, 1)
    seam, cost = find_vertical_seam(energy_fd(img))
    assert np.array_equal(res["seams"][0], seam) and np.isclose(res["costs"][0], cost / 15)
    assert np.all(np.abs(np.diff(res["seams"][0])) <= 1)


def test_energy_is_recomputed_after_each_removal():
    """Carving 2 seams in one call must equal: carve 1, then carve 1 more from the result."""
    img = rng.random((18, 14, 3))
    two = carve(img, 2)["image"]
    one = carve(img, 1)["image"]; again = carve(one, 1)["image"]
    assert np.array_equal(two, again)


def test_horizontal_seams_by_transposing():
    img = rng.random((16, 22, 3))
    h = carve(img, 5, direction="horizontal")
    assert h["image"].shape == (11, 22, 3) and h["seams"].shape == (5, 22)
    v = carve(img.transpose(1, 0, 2), 5)
    assert np.array_equal(h["image"], v["image"].transpose(1, 0, 2)) and np.array_equal(h["seams"], v["seams"])
    removed = np.zeros(img.shape[:2], bool)
    for s in h["seams"]:
        removed[s, np.arange(22)] = True
    assert removed.sum() == 5 * 22 and np.all(removed.sum(axis=0) == 5)             # one pixel per column per seam
    assert np.array_equal(np.stack([img[:, j][~removed[:, j]] for j in range(22)], axis=1), h["image"])


def test_validation_and_tiny_images():
    img = rng.random((6, 5))
    for bad in (-1, 5, 6, 2.5):
        try:
            carve(img, bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    for bad_img in (np.zeros(5), np.zeros((0, 4)), np.zeros((2, 2, 2, 2))):
        try:
            carve(bad_img, 1)
        except ValueError:
            continue
        raise AssertionError(bad_img.shape)
    assert carve(rng.random((4, 1)), 0)["image"].shape == (4, 1)                    # nothing to remove
    assert carve(rng.random((4, 2)), 1)["image"].shape == (4, 1)                    # down to one column
    assert carve(rng.random((1, 6)), 3)["image"].shape == (1, 3)                    # one-row image
    assert carve(rng.random((1, 1)), 0)["image"].shape == (1, 1)
    for tw in (0, 6, 2.5):
        try:
            carve_to_width(img, tw)
        except ValueError:
            continue
        raise AssertionError(tw)
    assert carve_to_width(img, 5)["image"].shape == (6, 5) and carve_to_width(img, 2)["image"].shape == (6, 2)
    try:
        carve(img, 1, direction="diagonal")
    except ValueError:
        return
    raise AssertionError("bad direction accepted")


def test_flat_region_is_carved_before_a_textured_one():
    img = np.zeros((40, 60)); img[:, :30] = rng.random((40, 30))                    # noise on the left, flat on the right
    res = carve(img, 15)
    assert res["seams"].min() >= 29, res["seams"].min()                              # all seams run through the flat part
    print("  textured-left / flat-right image: all 15 seams were taken from the flat half (original columns >= 29)")


def test_energy_functions():
    flat = np.full((20, 25, 3), 0.4)
    assert np.allclose(energy_fd(flat), 0, atol=1e-12) and np.allclose(energy_dog(flat), 0, atol=1e-12)
    assert energy_fd(flat, pad_mode="zeros")[10, 0] > 0.3                             # zero padding invents a frame edge
    step = np.zeros((20, 30)); step[:, 15:] = 1.0
    e = energy_fd(step); assert e.shape == (20, 30) and e[:, 14:16].min() > 0.99 and e[:, :12].max() < 1e-12
    d = energy_dog(step); assert d[:, 15].max() > 0.05 and (d[:, 9:21] > 1e-6).all()   # DoG spreads the edge over a wide band
    assert np.allclose(energy_fd(step[..., None].repeat(3, axis=2)), e)               # color -> grayscale first
    lo, hi = seam_search_space(427, 640)
    assert 130 < lo < 132 and 205 < hi < 207
    print(f"  number of vertical seams in a 427x640 image: between 10^{lo:.0f} and 10^{hi:.0f}")


def test_draw_seams():
    img = np.zeros((4, 5, 3)); seams = np.array([[0, 1, 1, 2], [4, 4, 3, 3]])
    out = draw_seams(img, seams)
    assert (out[0, 0] == [1, 0, 0]).all() and (out[3, 3] == [1, 0, 0]).all() and out.sum() == 8
    h = draw_seams(np.zeros((3, 4)), np.array([[1, 1, 2, 2]]), direction="horizontal")
    assert h.shape == (3, 4, 3) and h[1, 0, 0] == 1 and h[2, 3, 0] == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
