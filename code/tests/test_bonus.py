"""Tests for the Bonus functions in seam.py. Run from code/:   python tests/test_bonus.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from seam import insert_seams, remove_object, enlarge_by_seams, enlarge_naive

rng = np.random.default_rng(41)


def test_insert_hand_examples():
    row = np.array([[10., 20., 30., 40.]])
    cases = [([[1]], [10, 15, 25, 30, 40]),                 # first copy: (20+10)/2, second copy: (20+30)/2
             ([[0]], [10, 15, 20, 30, 40]),                 # left border: first copy averages with itself
             ([[3]], [10, 20, 30, 35, 40]),                 # right border: second copy averages with itself
             ([[1], [2]], [10, 15, 25, 25, 35, 40])]        # neighbors come from the INPUT row, not the output
    for seams, expected in cases:
        out, src, ins = insert_seams(row, np.array(seams))
        assert out[0].tolist() == expected, (seams, out)
    print("  insertion hand examples (middle, both borders, adjacent seams) all match")


def test_insert_properties_gray_and_color():
    for shape in [(9, 12), (7, 10, 3)]:
        img = rng.random(shape); H, W = shape[:2]; k = 3
        seams = np.array([rng.permutation(W)[:k] for _ in range(H)]).T          # k distinct pixels per row
        out, src, ins = insert_seams(img, seams)
        assert out.shape == (H, W + k) + shape[2:] and ins.sum() == 2 * k * H
        assert np.array_equal(out[~ins], img[np.arange(H)[:, None], src][~ins])  # non-seam pixels are plain copies
        for i in range(H):                                                       # each source column appears once, seam columns twice
            counts = np.bincount(src[i], minlength=W)
            assert set(counts) <= {1, 2} and np.array_equal(np.where(counts == 2)[0], np.sort(seams[:, i]))
    img = rng.random((4, 5))
    for b in (np.array([[-1, 0, 0, 0]]), np.array([[99, 0, 0, 0]])):                # outside the image
        try:
            insert_seams(img, b)
        except ValueError:
            continue
        raise AssertionError("bad seam accepted")
    try:
        insert_seams(img, np.array([[1, 1, 1, 1], [1, 2, 2, 2]]))
    except ValueError:
        return
    raise AssertionError("shared pixel not rejected")


def textured(h, w):
    return rng.random((h, w, 3))


def test_remove_object_rectangle_needs_exactly_its_width_in_seams():
    img = textured(30, 40); img[8:20, 15:19] = 5.0                  # a 4-px-wide "object" with a unique value
    mask = np.zeros((30, 40)); mask[8:20, 15:19] = 1.0
    before = img.copy()
    res = remove_object(img, mask)
    assert np.array_equal(img, before)
    assert res["k"] == 4 and res["image"].shape == (30, 36, 3) and res["seams"].shape == (4, 30)
    assert not (res["image"] == 5.0).any()                          # every object pixel is gone
    removed = np.zeros((30, 40), bool)
    for s in res["seams"]:
        removed[np.arange(30), s] = True
    assert removed.sum() == 4 * 30 and removed[mask > 0].all()      # the removed pixels include the whole mask
    print(f"  4-px-wide object -> exactly k = {res['k']} seams, all {int(mask.sum())} object pixels removed")


def test_remove_object_wider_in_some_rows_and_nothing_to_do():
    img = textured(25, 30); mask = np.zeros((25, 30))
    mask[5:10, 10:12] = 1; mask[10:16, 10:15] = 1; mask[16:20, 12:14] = 1       # widest row: 5 px
    img[mask > 0] = 7.0
    res = remove_object(img, mask)
    assert res["k"] == 5 and not (res["image"] == 7.0).any() and res["image"].shape == (25, 25, 3)
    res0 = remove_object(img, np.zeros((25, 30)))
    assert res0["k"] == 0 and np.array_equal(res0["image"], img)


def test_remove_object_validation():
    img = textured(10, 12)
    for bad in (np.zeros((10, 11)), np.zeros((9, 12))):
        try:
            remove_object(img, bad)
        except ValueError:
            continue
        raise AssertionError("mask shape mismatch accepted")
    try:
        remove_object(img, np.ones((10, 12)))                        # object as wide as the picture
    except ValueError:
        return
    raise AssertionError("whole-image mask accepted")


def test_enlarge_shape_and_copy_is_not_modified():
    red = textured(22, 26); before = red.copy()
    res = enlarge_by_seams(red, 5)
    assert np.array_equal(red, before)
    assert res["image"].shape == (22, 31, 3) and res["seams"].shape == (5, 22) and res["inserted"].sum() == 2 * 5 * 22
    assert np.array_equal(enlarge_by_seams(red, 0)["image"], red)


def test_k_seams_from_a_copy_spread_the_stretch_but_repeating_the_cheapest_seam_does_not():
    """Noise on the left, a perfectly flat band on the right: the cheapest seam is always in the flat band."""
    img = np.zeros((40, 50)); img[:, :20] = rng.random((40, 20))
    k = 8
    ours = enlarge_by_seams(img, k); naive = enlarge_naive(img, k)
    assert ours["image"].shape == naive["image"].shape == (40, 58)
    mult = lambda src: max(np.bincount(src[i], minlength=50).max() for i in range(40))
    m_ours, m_naive = mult(ours["src"]), mult(naive["src"])
    assert m_ours <= 2 and m_naive >= 4, (m_ours, m_naive)
    print(f"  flat-band test, k = {k}: a source pixel appears at most {m_ours}x with k seams from a copy, "
          f"but up to {m_naive}x when the cheapest seam is inserted k times")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
