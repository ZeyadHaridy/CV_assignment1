"""Tests for conv.py. Run from the code/ folder:   python tests/test_conv.py

SciPy is used here ONLY as a reference to check our own implementation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.signal import convolve2d, correlate2d
from conv import conv2d_loop

rng = np.random.default_rng(0)


def test_hand_example_shift():
    """A single 1 to the RIGHT of the kernel center shifts the image to the RIGHT."""
    img = np.arange(1, 10, dtype=float).reshape(3, 3)       # [[1,2,3],[4,5,6],[7,8,9]]
    k = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    expected = np.array([[0, 1, 2], [0, 4, 5], [0, 7, 8]], dtype=float)
    out = conv2d_loop(img, k)
    assert np.array_equal(out, expected), out
    # Plain correlation (no flip) would shift the OTHER way, so this proves we flip:
    corr = correlate2d(img, k, mode="same")
    assert not np.array_equal(out, corr)
    print("  hand example (shift right):", out.tolist())


def test_hand_example_sum():
    """3x3 box of ones on a tiny image, worked out by hand with zero padding."""
    img = np.array([[1, 2], [3, 4]], dtype=float)
    k = np.ones((3, 3))
    # every output pixel sees the whole 2x2 image (the rest is zeros) -> 10
    assert np.array_equal(conv2d_loop(img, k), np.full((2, 2), 10.0))


def test_shape_dtype():
    img = rng.random((7, 11)).astype(np.float32)
    out = conv2d_loop(img, rng.random((3, 5)))
    assert out.shape == img.shape and out.dtype == np.float64


def test_identity_kernel():
    img = rng.random((6, 9))
    k = np.zeros((5, 3)); k[2, 1] = 1.0
    assert np.allclose(conv2d_loop(img, k), img)


def test_random_vs_scipy():
    n_cases = 0
    for _ in range(60):
        H, W = rng.integers(1, 13, size=2)                 # includes 1-pixel-wide images
        kh, kw = 2 * rng.integers(0, 4, size=2) + 1        # odd: 1,3,5,7 (incl. non-square)
        img = rng.standard_normal((H, W))
        k = rng.standard_normal((kh, kw))
        ours = conv2d_loop(img, k)
        ref = convolve2d(img, k, mode="same", boundary="fill", fillvalue=0)
        assert ours.shape == ref.shape == (H, W)
        assert np.allclose(ours, ref, atol=1e-12), (H, W, kh, kw)
        n_cases += 1
    print(f"  {n_cases} random (image, kernel) pairs match scipy; max-size 12x12 / 7x7")


def test_separable_kernel_vs_scipy():
    col = rng.standard_normal((5, 1)); row = rng.standard_normal((1, 7))
    k = col @ row                                          # rank-1 (separable) kernel
    img = rng.standard_normal((10, 12))
    assert np.allclose(conv2d_loop(img, k), convolve2d(img, k, mode="same"), atol=1e-12)


def test_kernel_bigger_than_image():
    img = rng.standard_normal((2, 3)); k = rng.standard_normal((5, 7))
    assert np.allclose(conv2d_loop(img, k), convolve2d(img, k, mode="same"), atol=1e-12)


def test_even_kernel_rejected():
    for shape in [(2, 3), (3, 4), (4, 4)]:
        try:
            conv2d_loop(np.zeros((5, 5)), np.ones(shape))
        except ValueError:
            continue
        raise AssertionError(f"even kernel {shape} was not rejected")


def test_input_not_modified():
    img = rng.random((5, 5)); before = img.copy()
    conv2d_loop(img, rng.random((3, 3)))
    assert np.array_equal(img, before)


def test_reflect_padding_constant_image():
    """Reflect padding keeps a constant image constant (zero padding would darken borders)."""
    img = np.full((6, 6), 5.0); k = np.ones((3, 3)) / 9
    assert np.allclose(conv2d_loop(img, k, pad_mode="reflect"), 5.0)
    assert conv2d_loop(img, k, pad_mode="zeros")[0, 0] < 5.0


# ---------------------------------------------------------------- vectorized version
from conv import conv2d_vectorized


def test_vec_hand_example_shift():
    img = np.arange(1, 10, dtype=float).reshape(3, 3)
    k = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    expected = np.array([[0, 1, 2], [0, 4, 5], [0, 7, 8]], dtype=float)
    assert np.array_equal(conv2d_vectorized(img, k), expected)


def test_vec_random_vs_scipy_and_loop():
    for _ in range(100):
        H, W = rng.integers(1, 16, size=2)
        kh, kw = 2 * rng.integers(0, 5, size=2) + 1        # 1,3,5,7,9
        img = rng.standard_normal((H, W)); k = rng.standard_normal((kh, kw))
        vec = conv2d_vectorized(img, k)
        assert vec.shape == (H, W)
        assert np.allclose(vec, convolve2d(img, k, mode="same"), atol=1e-12), (H, W, kh, kw)
        assert np.allclose(vec, conv2d_loop(img, k), atol=1e-12), (H, W, kh, kw)
    print("  100 random cases: vectorized == scipy == loop (kernels up to 9x9, images up to 15x15)")


def test_vec_medium_image():
    img = rng.random((40, 55)); k = rng.standard_normal((9, 5))
    assert np.allclose(conv2d_vectorized(img, k), convolve2d(img, k, mode="same"), atol=1e-12)


def test_vec_reflect_padding_independent_check():
    """Independent reference: reflect-pad with np.pad, then 'valid' convolution with SciPy."""
    for _ in range(30):
        H, W = rng.integers(4, 15, size=2)
        kh, kw = 2 * rng.integers(0, 3, size=2) + 1
        img = rng.standard_normal((H, W)); k = rng.standard_normal((kh, kw))
        ref = convolve2d(np.pad(img, ((kh // 2,) * 2, (kw // 2,) * 2), mode="reflect"), k, mode="valid")
        assert np.allclose(conv2d_vectorized(img, k, pad_mode="reflect"), ref, atol=1e-12)
        assert np.allclose(conv2d_loop(img, k, pad_mode="reflect"), ref, atol=1e-12)


def test_vec_uint8_input_no_overflow():
    img = rng.integers(0, 256, size=(8, 8), dtype=np.uint8); k = np.ones((3, 3))
    out = conv2d_vectorized(img, k)
    assert out.dtype == np.float64 and np.allclose(out, convolve2d(img.astype(float), k, mode="same"))


def test_vec_input_not_modified_and_even_rejected():
    img = rng.random((5, 5)); before = img.copy()
    conv2d_vectorized(img, rng.random((3, 3)))
    assert np.array_equal(img, before)
    try:
        conv2d_vectorized(img, np.ones((2, 2)))
    except ValueError:
        return
    raise AssertionError("even kernel not rejected")


# ---------------------------------------------------------------- separable version
from conv import conv2d_separable, separable_factors


def test_sep_factors_known_kernels():
    box = np.ones((3, 3))
    g = np.array([1, 4, 6, 4, 1.0]); gauss = np.outer(g, g) / 256
    plus = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    for name, k, expect in [("box", box, True), ("outer([1 4 6 4 1])", gauss, True),
                            ("plus", plus, False), ("zeros", np.zeros((3, 3)), True),
                            ("1x3 [1 0 -1]", np.array([[1, 0, -1.0]]), True)]:
        f = separable_factors(k)
        assert (f is not None) == expect, name
        if f is not None:
            col, row = f
            assert col.shape == (k.shape[0], 1) and row.shape == (1, k.shape[1])
            assert np.allclose(col @ row, k, atol=1e-12), name
    print("  SVD test: box/Gaussian/[1 0 -1]/zeros separable, plus-shape not")


def test_sep_random_vs_scipy():
    n_sep = n_non = 0
    for _ in range(100):
        H, W = rng.integers(1, 16, size=2)
        kh, kw = 2 * rng.integers(0, 5, size=2) + 1
        if rng.random() < 0.5:
            k = rng.standard_normal((kh, 1)) @ rng.standard_normal((1, kw)); n_sep += 1
        else:
            k = rng.standard_normal((kh, kw))
            if min(kh, kw) > 1: n_non += 1
            else: n_sep += 1
        img = rng.standard_normal((H, W))
        out = conv2d_separable(img, k)
        assert np.allclose(out, convolve2d(img, k, mode="same"), atol=1e-11), (H, W, kh, kw)
    print(f"  separable version == scipy on 100 random cases (~{n_sep} rank-1, ~{n_non} full-rank kernels)")


def test_sep_reflect_matches_vectorized():
    g = np.array([1, 4, 6, 4, 1.0]); gauss = np.outer(g, g) / 256
    img = rng.standard_normal((12, 9))
    assert np.allclose(conv2d_separable(img, gauss, pad_mode="reflect"),
                       conv2d_vectorized(img, gauss, pad_mode="reflect"), atol=1e-12)


def test_sep_nearly_rank1_is_not_misclassified():
    k = np.outer([1, 2, 1.0], [1, 2, 1.0]); k[1, 1] += 1e-3          # slightly broken
    assert separable_factors(k) is None
    img = rng.standard_normal((8, 8))
    assert np.allclose(conv2d_separable(img, k), convolve2d(img, k, mode="same"), atol=1e-12)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
