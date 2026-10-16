"""Part 1.1: sanity check of the three versions against SciPy, then the timing table."""
import time, csv, os
import numpy as np
from scipy.signal import convolve2d
from conv import conv2d_loop, conv2d_vectorized, conv2d_separable, separable_factors
from filters import gaussian_kernel
from paths import RESULTS

rng = np.random.default_rng(123)


def best_of(reps, f):
    """Smallest wall-clock time of `reps` calls (reduces noise from other processes)."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); times.append(time.perf_counter() - t0)
    return min(times)

os.makedirs(RESULTS, exist_ok=True)

# ---------- sanity check: small random images, random odd filters, separable and not ----------
errs = {"loop": 0.0, "vectorized": 0.0, "separable": 0.0}
counts = {"separable kernels": 0, "non-separable kernels": 0}
for _ in range(300):
    H, W = rng.integers(3, 21, size=2)
    kh, kw = 2 * rng.integers(1, 5, size=2) + 1                 # 3,5,7,9
    img = rng.standard_normal((H, W))
    if rng.random() < 0.5:
        k = rng.standard_normal((kh, 1)) @ rng.standard_normal((1, kw))
    else:
        k = rng.standard_normal((kh, kw))
    counts["separable kernels" if separable_factors(k) is not None else "non-separable kernels"] += 1
    ref = convolve2d(img, k, mode="same", boundary="fill", fillvalue=0)
    for name, f in [("loop", conv2d_loop), ("vectorized", conv2d_vectorized), ("separable", conv2d_separable)]:
        errs[name] = max(errs[name], np.abs(f(img, k) - ref).max())
print("SANITY CHECK vs scipy.signal.convolve2d(mode='same'), 300 random cases")
print("  kernels used:", counts)
for n, e in errs.items():
    print(f"  max |ours - scipy|  {n:11s}: {e:.2e}")
assert all(e < 1e-10 for e in errs.values())

# ---------- timing: 256x256 image, Gaussian filters 3x3 ... 31x31 ----------
img = rng.random((256, 256))
sizes = list(range(3, 32, 2))
rows = []
print("\nTIMING on a 256x256 image (seconds; vectorized/separable = best of several repeats)")
print(f"{'size':>7} {'loop':>10} {'vectorized':>12} {'separable':>11} {'loop/vec':>9} {'vec/sep':>8}")
for n in sizes:
    k = gaussian_kernel(n, n / 6.0)                               # sigma = n/6 -> kernel spans +-3 sigma
    t = time.perf_counter(); out_l = conv2d_loop(img, k); t_loop = time.perf_counter() - t
    reps = 20 if n < 15 else 8
    t_vec = best_of(reps, lambda: conv2d_vectorized(img, k))
    t_sep = best_of(reps, lambda: conv2d_separable(img, k))
    assert np.abs(out_l - conv2d_vectorized(img, k)).max() < 1e-10
    assert np.abs(out_l - conv2d_separable(img, k)).max() < 1e-10
    rows.append((n, t_loop, t_vec, t_sep))
    print(f"{n:>3}x{n:<3} {t_loop:10.4f} {t_vec:12.5f} {t_sep:11.5f} {t_loop/t_vec:9.0f} {t_vec/t_sep:8.1f}")

with open(f"{RESULTS}/benchmark_1_1.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["size", "loop_s", "vectorized_s", "separable_s"]); w.writerows(rows)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = np.array(rows)
plt.figure(figsize=(6.5, 4.2))
for j, lab in [(1, "loop"), (2, "vectorized"), (3, "separable")]:
    plt.plot(r[:, 0], r[:, j], "o-", label=lab)
plt.yscale("log"); plt.xlabel("Gaussian filter size (n x n)"); plt.ylabel("time (s), log scale")
plt.title("Convolution time, 256x256 image"); plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(f"{RESULTS}/benchmark_1_1.png", dpi=150)
print("\nsaved results/benchmark_1_1.csv and results/benchmark_1_1.png")
