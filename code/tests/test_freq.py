"""Tests for freq.py. Run from code/:   python tests/test_freq.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from freq import (gaussian_blur, low_pass, high_pass, hybrid_image, to_gray,
                  log_fft_magnitude, gaussian_cutoff_frequency)
from align_images import similarity_from_two_points

rng = np.random.default_rng(11)


def test_blur_constant_image_stays_constant_with_reflect():
    img = np.full((30, 40), 0.7)
    assert np.allclose(gaussian_blur(img, 2.5), 0.7, atol=1e-12)          # zeros would darken the border
    assert np.allclose(high_pass(img, 2.5), 0.0, atol=1e-12)


def test_blur_preserves_shape_dtype_and_color_is_per_channel():
    img = rng.random((25, 31, 3))
    out = gaussian_blur(img, 1.5)
    assert out.shape == img.shape and out.dtype == np.float64
    for c in range(3):
        assert np.allclose(out[..., c], gaussian_blur(img[..., c], 1.5))


def test_low_plus_high_is_the_image():
    img = rng.random((20, 20))
    assert np.allclose(low_pass(img, 2) + high_pass(img, 2), img, atol=1e-12)


def test_blur_attenuates_a_sine_like_the_gaussian_frequency_response():
    """A sine of frequency f (cycles/pixel) comes out of a Gaussian blur with amplitude H(f).
    Amplitude is measured by projecting onto the sine over a whole number of periods (the largest
    SAMPLE of a sampled sine is not its amplitude).  H(f) is checked two ways:
      * exactly, for OUR truncated (+-3 sigma, renormalized) kernel:  sum_k g[k] cos(2 pi f k)
      * approximately, against the continuous formula exp(-2 pi^2 sigma^2 f^2)."""
    from filters import gaussian_kernel_1d, default_size
    x = np.arange(400)
    for sigma, f in [(2.0, 0.05), (2.0, 0.10), (3.0, 0.0625), (1.0, 0.20)]:
        img = np.tile(np.sin(2 * np.pi * f * x), (60, 1))
        out = gaussian_blur(img, sigma)[30, 40:360]                      # 320 samples = whole periods
        xs = x[40:360]
        measured = 2 * abs(np.mean(out * np.exp(-2j * np.pi * f * xs)))
        n = default_size(sigma); g = gaussian_kernel_1d(n, sigma); k = np.arange(n) - n // 2
        exact = abs((g * np.cos(2 * np.pi * f * k)).sum())
        continuous = np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2)
        assert abs(measured - exact) < 1e-9, (sigma, f, measured, exact)
        assert abs(measured - continuous) < 2e-3, (sigma, f, measured, continuous)
        print(f"  sigma={sigma}, f={f}: measured {measured:.4f} | our kernel {exact:.4f} | continuous formula {continuous:.4f}")


def test_high_pass_removes_slow_variation_keeps_fast():
    n = 400; x = np.arange(n)
    slow = np.tile(np.sin(2 * np.pi * 0.01 * x), (40, 1))
    fast = np.tile(np.sin(2 * np.pi * 0.25 * x), (40, 1))
    assert np.abs(high_pass(slow, 3)[20, 100:300]).max() < 0.05
    assert np.abs(high_pass(fast, 3)[20, 100:300]).max() > 0.99


def test_hybrid_is_sum_and_cutoff_formula():
    a, b = rng.random((16, 16)), rng.random((16, 16))
    assert np.allclose(hybrid_image(a, b, 2, 1), low_pass(a, 2) + high_pass(b, 1))
    assert abs(np.exp(-2 * np.pi ** 2 * 3.0 ** 2 * gaussian_cutoff_frequency(3.0) ** 2) - 0.5) < 1e-12


def test_log_fft_dc_is_at_center_and_is_max_for_positive_image():
    img = rng.random((33, 40)) + 0.5
    L = log_fft_magnitude(img)
    assert L.shape == img.shape
    assert np.unravel_index(L.argmax(), L.shape) == (33 // 2, 40 // 2)


def test_to_gray():
    rgb = rng.random((5, 6, 3))
    g = to_gray(rgb)
    assert g.shape == (5, 6) and 0 <= g.min() and g.max() <= 1
    assert np.allclose(to_gray(g), g)


def test_alignment_points_land_on_target_with_starter_coordinates():
    pts1 = [(299, 341), (443, 332)]; pts2 = [(605, 290), (760, 372)]       # man eyes, cat eyes (README)
    M = similarity_from_two_points(pts2, pts1)
    for (x, y), (tx, ty) in zip(pts2, pts1):
        assert np.allclose(M @ [x, y, 1], [tx, ty], atol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
