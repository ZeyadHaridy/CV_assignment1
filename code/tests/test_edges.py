"""Tests for filters.py / edges.py. Run from code/:   python tests/test_edges.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.signal import convolve2d
from conv import conv2d_separable, conv2d_vectorized, separable_factors
from filters import DX, DY, gaussian_kernel, gaussian_kernel_1d, default_size
from edges import (finite_difference_gradient, dog_filters, dog_gradient, threshold_edges,
                   signed_for_display, magnitude_for_display)

rng = np.random.default_rng(7)


def test_filter_shapes():
    assert DX.shape == (1, 3) and DY.shape == (3, 1) and np.array_equal(DY, DX.T)


def test_gaussian_kernel_properties():
    for sigma in (0.5, 1, 2, 3.7):
        k = gaussian_kernel(default_size(sigma), sigma)
        assert k.shape[0] == k.shape[1] == default_size(sigma) and k.shape[0] % 2 == 1
        assert abs(k.sum() - 1) < 1e-12
        assert np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])      # symmetric
        assert k[k.shape[0] // 2, k.shape[1] // 2] == k.max()             # peak at the center
        assert separable_factors(k) is not None                           # rank 1
    assert default_size(2) == 13


def test_step_edge_sign_and_location():
    """Dark (0) | bright (1) vertical step.  Convolution FLIPS [1 0 -1] into [-1 0 1], so the output is
    I[x+1] - I[x-1] (the usual central difference): +1 on the two pixels that straddle the step."""
    img = np.zeros((6, 8)); img[:, 4:] = 1.0                               # step between col 3 and 4
    gx, gy, mag = finite_difference_gradient(img, pad_mode="reflect")
    assert np.allclose(gx[:, 3], 1) and np.allclose(gx[:, 4], 1)
    assert np.allclose(gx[:, [0, 1, 2, 5, 6, 7]], 0)
    assert np.allclose(gy, 0)
    assert np.allclose(mag[:, 3], 1)
    # a horizontal step (dark on top, bright below) gives gy = +1 on the two straddling rows
    img2 = img.T.copy()
    gx2, gy2, _ = finite_difference_gradient(img2, pad_mode="reflect")
    assert np.allclose(gy2[3], 1) and np.allclose(gy2[4], 1) and np.allclose(gx2, 0)


def test_matches_scipy_reference():
    img = rng.random((15, 17))
    gx, gy, mag = finite_difference_gradient(img)
    assert np.allclose(gx, convolve2d(img, DX, mode="same"), atol=1e-12)
    assert np.allclose(gy, convolve2d(img, DY, mode="same"), atol=1e-12)
    assert np.allclose(mag, np.hypot(gx, gy))


def test_zero_padding_creates_frame_artifact():
    """Constant image: interior gradient is 0, but with zero padding the frame lights up."""
    img = np.full((9, 9), 0.6)
    _, _, m_zero = finite_difference_gradient(img, "zeros")
    _, _, m_refl = finite_difference_gradient(img, "reflect")
    assert np.allclose(m_zero[2:-2, 2:-2], 0)
    assert np.allclose(m_zero[4, 0], 0.6) and np.allclose(m_zero[4, -1], 0.6)   # |0 - I| on the border
    assert np.allclose(m_zero[0, 4], 0.6) and np.allclose(m_zero[-1, 4], 0.6)
    assert np.allclose(m_refl, 0)                                              # reflect: no frame


def test_dog_filter_properties():
    for sigma in (1, 2, 3):
        dx, dy = dog_filters(sigma)
        n = default_size(sigma)
        assert dx.shape == dy.shape == (n, n)
        assert np.allclose(dy, dx.T)                                           # isotropic Gaussian
        assert np.allclose(dx, -dx[:, ::-1])                                   # antisymmetric in x
        assert np.allclose(dx, dx[::-1, :])                                    # symmetric in y
        assert abs(dx.sum()) < 1e-12                                           # no response to flat areas
        assert separable_factors(dx) is not None and separable_factors(dy) is not None
        # the two lobes have opposite signs
        assert dx[n // 2, : n // 2].sum() * dx[n // 2, n // 2 + 1:].sum() < 0  # opposite signs


def test_dog_equals_blur_then_derivative():
    """Associativity: (I * G) * Dx  ==  I * (G * Dx)   (exact when the DoG keeps all its taps)."""
    sigma = 2; n = default_size(sigma)
    G = gaussian_kernel(n, sigma)
    img = rng.random((40, 45))
    blurred = conv2d_separable(img, G)
    ref_x = conv2d_separable(blurred, DX)
    ref_y = conv2d_separable(blurred, DY)
    m = n // 2 + 2                                                            # ignore the border zone
    # exact DoG: full convolution of G with Dx has 2 extra columns -> pad G with zero columns first
    dog_x_full = conv2d_separable(np.pad(G, ((0, 0), (1, 1))), DX)
    dog_y_full = conv2d_separable(np.pad(G, ((1, 1), (0, 0))), DY)
    ex = conv2d_separable(img, dog_x_full); ey = conv2d_separable(img, dog_y_full)
    assert np.allclose(ex[m:-m, m:-m], ref_x[m:-m, m:-m], atol=1e-12)
    assert np.allclose(ey[m:-m, m:-m], ref_y[m:-m, m:-m], atol=1e-12)
    # the assignment's same-size DoG drops those 2 tiny edge taps -> tiny, non-zero difference
    gx, gy, _ = dog_gradient(img, sigma)
    diff = np.abs(gx[m:-m, m:-m] - ref_x[m:-m, m:-m]).max()
    print(f"  same-size DoG vs blur-then-derivative, interior max |diff| = {diff:.2e} (exact when taps kept: <1e-12)")
    assert diff < 5e-3


def test_dog_suppresses_noise():
    """On a noisy flat image the DoG response is much smaller than the finite difference response."""
    noise = 0.05 * rng.standard_normal((120, 120)) + 0.5
    _, _, m12 = finite_difference_gradient(noise, "reflect")
    _, _, m13 = dog_gradient(noise, 2, pad_mode="reflect")
    assert m13.mean() < 0.35 * m12.mean()
    print(f"  noise-only image: mean magnitude 1.2 = {m12.mean():.4f}, DoG(sigma=2) = {m13.mean():.4f}")


def test_threshold_and_display_helpers():
    m = np.array([[0.1, 0.3], [0.5, 0.2]])
    assert np.array_equal(threshold_edges(m, 0.25), np.array([[False, True], [True, False]]))
    d = np.array([[-2.0, 0.0], [1.0, 2.0]])
    disp = signed_for_display(d)
    assert disp[0, 0] == 0.0 and disp[0, 1] == 0.5 and disp[1, 1] == 1.0       # zero -> mid-gray
    assert np.array_equal(d, np.array([[-2.0, 0.0], [1.0, 2.0]]))              # original untouched
    assert magnitude_for_display(m).max() == 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
