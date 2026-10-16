"""Tests for blend.py. Run from code/:   python tests/test_blend.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from blend import multires_blend, hard_cut, effective_alpha
from pyramids import max_levels

rng = np.random.default_rng(21)


def half_mask(h, w):
    m = np.zeros((h, w)); m[:, : w // 2] = 1.0
    return m


def test_mask_all_ones_or_zeros_returns_one_input():
    a, b = rng.random((40, 52, 3)), rng.random((40, 52, 3))
    for lv in (1, 3, 5):
        assert np.allclose(multires_blend(a, b, np.ones((40, 52)), lv), a, atol=1e-12)
        assert np.allclose(multires_blend(a, b, np.zeros((40, 52)), lv), b, atol=1e-12)


def test_identical_images_blend_to_themselves():
    a = rng.random((33, 47)); m = (rng.random((33, 47)) > 0.5).astype(float)
    assert np.allclose(multires_blend(a, a, m, 4), a, atol=1e-12)


def test_one_level_equals_hard_cut():
    a, b = rng.random((30, 41, 3)), rng.random((30, 41, 3))
    m = (rng.random((30, 41)) > 0.5).astype(float)
    assert np.allclose(multires_blend(a, b, m, 1), hard_cut(a, b, m), atol=1e-14)


def test_swapping_images_and_complementing_mask_gives_the_same_result():
    a, b = rng.random((37, 29)), rng.random((37, 29)); m = rng.random((37, 29))
    assert np.allclose(multires_blend(a, b, m, 4), multires_blend(b, a, 1 - m, 4), atol=1e-12)


def test_shapes_odd_sizes_gray_and_color_and_errors():
    for shape in [(65, 47), (64, 64), (7, 9), (31, 100, 3)]:
        a, b = rng.random(shape), rng.random(shape)
        m = rng.random(shape[:2])
        out = multires_blend(a, b, m, min(4, max_levels(shape)))
        assert out.shape == a.shape and out.dtype == np.float64
    try:
        multires_blend(rng.random((8, 8)), rng.random((8, 9)), np.ones((8, 8)), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch not rejected")
    try:
        multires_blend(rng.random((8, 8)), rng.random((8, 8)), np.ones((8, 9)), 2)
    except ValueError:
        return
    raise AssertionError("mask mismatch not rejected")


def test_parts_add_up_and_follow_the_formula():
    a, b = rng.random((40, 40, 3)), rng.random((40, 40, 3)); m = half_mask(40, 40)
    out, p = multires_blend(a, b, m, 4, return_parts=True)
    for i in range(4):
        gm = p["GM"][i][..., None]
        assert np.allclose(p["A"][i], gm * p["L1"][i]) and np.allclose(p["B"][i], (1 - gm) * p["L2"][i])
        assert np.allclose(p["combined"][i], p["A"][i] + p["B"][i])


def transition_width(alpha_row):
    """Number of pixels where alpha is between 10% and 90%."""
    return int(np.sum((alpha_row > 0.1) & (alpha_row < 0.9)))


def test_more_levels_means_a_softer_seam():
    m = half_mask(256, 256)
    widths = []
    for lv in range(1, 8):
        widths.append(transition_width(effective_alpha(m, lv)[128]))
    assert widths[0] == 0                                     # 1 level = hard cut: no pixel between 10% and 90%
    assert all(w2 >= w1 for w1, w2 in zip(widths, widths[1:])) and widths[-1] > widths[1] > 0
    print("  seam width (pixels between 10% and 90% of image 1) for 1..7 levels:", widths)


def test_seam_is_centered_and_alpha_is_between_0_and_1_roughly():
    m = half_mask(128, 128)
    al = effective_alpha(m, 5)[64]
    assert abs(np.argmin(np.abs(al - 0.5)) - 63.5) <= 1.5       # the 50% point stays at the mask edge
    assert al[:8].min() > 0.99 and al[-8:].max() < 0.01
    assert al.min() > -0.02 and al.max() < 1.02                 # tiny over/undershoot at most


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
