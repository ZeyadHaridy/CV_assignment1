"""Tests for the 2.2 aliasing tools. Run from code/:   python tests/test_aliasing.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from freq import downsample, blur_then_downsample

N, FACTOR = 512, 4


def sine_image(f):
    x = np.arange(N)
    return np.tile(np.sin(2 * np.pi * f * x), (N, 1))              # f in cycles per ORIGINAL pixel


def peak_frequency(row):
    """Frequency (cycles per sample) of the strongest non-DC component of a 1D signal."""
    spec = np.abs(np.fft.rfft(row - row.mean()))
    return np.fft.rfftfreq(len(row))[spec.argmax()]


def amplitude(row, f):
    """Amplitude of the component of `row` at frequency f (cycles/sample), by projection."""
    n = np.arange(len(row))
    return 2 * abs(np.mean(row * np.exp(-2j * np.pi * f * n)))


def test_shapes_and_values():
    img = np.arange(64.0).reshape(8, 8)
    assert np.array_equal(downsample(img, 4), img[::4, ::4]) and downsample(img, 4).shape == (2, 2)
    assert np.allclose(blur_then_downsample(np.full((16, 16), 0.3), 4, 2.0), 0.3)


def test_naive_subsampling_creates_a_false_low_frequency():
    """f = 0.2 cycles/px is above the new Nyquist (1/8). After keeping every 4th pixel it shows up as
    a FALSE pattern at |4*0.2 - 1| = 0.2 cycles per new pixel, at full strength (amplitude 1)."""
    out = downsample(sine_image(0.2), FACTOR)[0]                  # one row, 128 samples
    f_alias = peak_frequency(out)
    assert abs(f_alias - 0.2) < 1.0 / 128, f_alias
    assert abs(amplitude(out, 0.2) - 1.0) < 0.05
    print(f"  naive: sine at 0.2 cyc/px -> false pattern at {f_alias:.3f} cyc/new px, amplitude {amplitude(out, 0.2):.3f}")


def test_prefilter_kills_the_false_pattern_in_proportion_to_the_gaussian_response():
    """Blur with sigma=2 first: amplitude of the false pattern ~ exp(-2 pi^2 sigma^2 f^2) = 0.04."""
    sigma, f = 2.0, 0.2
    out = blur_then_downsample(sine_image(f), FACTOR, sigma)[0]
    amp = amplitude(out, 0.2)
    expected = np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2)
    assert abs(amp - expected) < 0.01, (amp, expected)
    print(f"  sigma=2: false pattern amplitude {amp:.3f} (theory {expected:.3f}) vs 1.000 without prefilter")


def test_legitimate_low_frequency_survives_prefiltering():
    """f = 0.05 cyc/px (below the new Nyquist) is only mildly attenuated, and appears at 4*0.05 = 0.2 cyc/new px."""
    sigma, f = 2.0, 0.05
    out = blur_then_downsample(sine_image(f), FACTOR, sigma)[0]
    expected = np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2)
    assert abs(peak_frequency(out) - 0.2) < 1.0 / 128
    assert abs(amplitude(out, 0.2) - expected) < 0.01
    print(f"  sigma=2: real detail at 0.05 cyc/px keeps amplitude {amplitude(out, 0.2):.3f} (theory {expected:.3f})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
