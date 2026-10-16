"""Tests for pyramids.py. Run from code/:   python tests/test_pyramids.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.signal import convolve2d
from pyramids import (reduce, expand, gaussian_pyramid, laplacian_pyramid, reconstruct, max_levels,
                      lecture_style_pyramid, KERNEL_1D, KERNEL_2D)

rng = np.random.default_rng(5)


def ref_blur(a):                                   # independent reference: np.pad(reflect) + scipy 'valid'
    return convolve2d(np.pad(a, 2, mode="reflect"), KERNEL_2D, mode="valid")


def test_kernel():
    assert np.allclose(KERNEL_1D * 16, [1, 4, 6, 4, 1]) and abs(KERNEL_2D.sum() - 1) < 1e-15
    assert abs((KERNEL_1D * (np.arange(5) - 2) ** 2).sum() - 1.0) < 1e-15           # variance 1 -> sigma = 1


def test_reduce_shapes_odd_and_even():
    for H, W, eh, ew in [(8, 8, 4, 4), (7, 9, 4, 5), (1, 1, 1, 1), (2, 3, 1, 2), (670, 736, 335, 368), (5, 1, 3, 1)]:
        assert reduce(rng.random((H, W))).shape == (eh, ew), (H, W)
    assert reduce(rng.random((7, 9, 3))).shape == (4, 5, 3)


def test_reduce_matches_independent_reference():
    for H, W in [(9, 11), (16, 16), (6, 5), (30, 41)]:
        a = rng.standard_normal((H, W))
        assert np.allclose(reduce(a), ref_blur(a)[::2, ::2], atol=1e-12)


def test_expand_matches_independent_reference():
    for h, w in [(5, 6), (8, 8), (3, 4), (15, 10)]:
        a = rng.standard_normal((h, w))
        up = np.zeros((2 * h, 2 * w)); up[::2, ::2] = a
        assert np.allclose(expand(a), 4 * ref_blur(up), atol=1e-12)


def test_reduce_hand_values_impulse():
    a = np.zeros((9, 9)); a[4, 4] = 1.0
    r = reduce(a)                                  # shape 5x5, impulse lands on r[2, 2]
    assert r.shape == (5, 5)
    assert np.isclose(r[2, 2], 36 / 256)           # center weight 6/16 in x times 6/16 in y
    assert np.isclose(r[2, 1], 6 / 256)            # input column 2: offset 2 -> weight 1/16 in x times 6/16 in y
    assert np.isclose(r[1, 1], 1 / 256)            # offset (2, 2) -> 1/16 * 1/16


def test_expand_hand_values_impulse():
    a = np.zeros((9, 9)); a[4, 4] = 1.0
    e = expand(a)                                  # 18x18, the pixel lands at (8, 8) of the zero-filled grid
    assert e.shape == (18, 18)
    assert np.isclose(e[8, 8], 4 * 36 / 256)       # 0.5625
    assert np.isclose(e[8, 9], 4 * 6 * 4 / 256)    # 0.375
    assert np.isclose(e[9, 9], 4 * 16 / 256)       # 0.25
    assert np.isclose(e[8, 10], 4 * 6 * 1 / 256)   # 0.09375
    assert np.isclose(e.sum(), 4.0)                # one source pixel now covers a 2x2 block: total mass x4


def test_constant_image_is_preserved_everywhere():
    for shape in [(8, 8), (7, 9), (33, 20)]:
        c = np.full(shape, 0.37)
        assert np.allclose(reduce(c), 0.37, atol=1e-14)
        assert np.allclose(expand(c), 0.37, atol=1e-14), shape          # holds at the borders too (the factor 4)


def test_expand_crop_rules():
    a = rng.random((4, 5))
    assert expand(a).shape == (8, 10)
    assert expand(a, (7, 9)).shape == (7, 9)
    assert np.array_equal(expand(a, (7, 9)), expand(a)[:7, :9])       # "drop the last one"
    for bad in [(6, 10), (8, 12), (9, 10)]:
        try:
            expand(a, bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_pyramid_sizes_and_storage():
    img = rng.random((670, 736))
    gp = gaussian_pyramid(img, 7)
    assert [g.shape for g in gp] == [(670, 736), (335, 368), (168, 184), (84, 92), (42, 46), (21, 23), (11, 12)]
    lp = laplacian_pyramid(gp)
    assert [l.shape for l in lp] == [g.shape for g in gp]
    assert np.array_equal(lp[-1], gp[-1])
    ratio = sum(g.size for g in gp) / img.size
    assert 1.33 < ratio < 1.34, ratio                      # odd sides round UP, so it can land a hair above 4/3
    # power-of-two image: exactly the geometric series 1 + 1/4 + 1/16 + ... (n terms) < 4/3
    p2 = gaussian_pyramid(rng.random((256, 256)), 8)
    ratio2 = sum(g.size for g in p2) / 256 ** 2
    assert abs(ratio2 - sum(0.25 ** k for k in range(8))) < 1e-12 and ratio2 < 4 / 3
    print(f"  total size / image: 670x736, 7 levels = {ratio:.5f};  256x256, 8 levels = {ratio2:.5f};  limit 4/3 = {4/3:.5f}")
    assert max_levels((670, 736)) == 11 and max_levels((1, 1)) == 1 and max_levels((8, 8)) == 4


def test_reconstruction_is_exact():
    worst = 0.0
    shapes = [(64, 64), (65, 33), (7, 9), (2, 3), (5, 1), (1, 1), (100, 37), (670, 736)]
    for H, W in shapes:
        img = rng.random((H, W))
        for levels in {1, 2, max_levels((H, W)), max(1, max_levels((H, W)) - 2)}:
            rec = reconstruct(laplacian_pyramid(gaussian_pyramid(img, levels)))
            assert rec.shape == img.shape
            err = np.abs(rec - img).max(); worst = max(worst, err)
            assert err < 1e-12, (H, W, levels, err)
    col = rng.random((45, 31, 3))
    err = np.abs(reconstruct(laplacian_pyramid(gaussian_pyramid(col, 5))) - col).max()
    assert err < 1e-12; worst = max(worst, err)
    print(f"  exact rebuild on {len(shapes)} shapes (odd, even, tiny, 670x736) + color: worst max|error| = {worst:.2e}")


def test_exactness_does_not_depend_on_the_image_content():
    """The rebuild is exact even for white noise (nothing smooth to exploit) -> it's by construction."""
    noise = rng.standard_normal((50, 51))
    assert np.abs(reconstruct(laplacian_pyramid(gaussian_pyramid(noise, 5))) - noise).max() < 1e-12


def test_lecture_style_levels_are_not_enough_to_rebuild_exactly():
    """'Level minus its own blur': rebuilding with expand(next) + (G - blur G) is NOT exact."""
    img = rng.random((64, 64))
    gp = gaussian_pyramid(img, 4)
    lec = lecture_style_pyramid(gp)
    rec = gp[-1]                                       # smallest level: nothing coarser, keep as is
    for lv in reversed(range(len(gp) - 1)):
        rec = lec[lv] + expand(rec, gp[lv].shape)      # same rebuild recipe, but with 'level minus blur' details
    err = np.abs(rec - img).max()
    assert err > 1e-2, err
    print(f"  'level minus its blur' rebuild error (random 64x64, 4 levels): max|error| = {err:.3f}  vs ~1e-16 for ours")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
