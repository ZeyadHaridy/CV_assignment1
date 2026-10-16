# Computer Vision - Assignment 1: Fine Detail and the Big Picture

## Layout
```
cv_assignment1/
  code/        our implementation (conv.py, edges.py, freq.py, pyramids.py, blend.py, seam.py, ...), run_*.py scripts, tests/
  images/      the starter pack's images/ folder (NOT included here; copy it: part1_filters/, part2_frequencies/, part3_seams/)
  results/     figures (png), benchmark csv, and logs/ (one text log per script)
  requirements.txt
```
`code/utils.py` and `code/align_images.py` are the starter files, unchanged.

## Environment (Python 3.10+)
```
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # numpy, scipy, opencv-python, matplotlib   (Windows: .venv\Scripts\pip)
```
Tested with Python 3.12.3, numpy 2.5.4, scipy 1.18.1, opencv-python 5.0.0, matplotlib 3.11.2.

## Run everything (any working directory)
```
.venv/bin/python code/run_all.py                    # 84 automatic tests, then every experiment (~5 min, mostly the 1.1 benchmark)
.venv/bin/python code/run_all.py --skip-benchmark   # same without the slow 1.1 timing benchmark (~3 min)
.venv/bin/python code/run_all.py --tests-only       # about 15 s
```
Single pieces: `python code/tests/test_conv.py`, `python code/run_2_4_blending.py`, `python code/run_2_2_aliasing.py 2.5`, ...

## Which script produces what
| Script | Section | Output |
|---|---|---|
| run_1_1_benchmark.py | 1.1 | sanity check vs scipy (300 random cases), timing table + plot |
| run_1_2_1_3.py | 1.2, 1.3 | derivative / magnitude / edge maps, DoG filters, threshold sweep, frame-corner figure |
| run_2_1_hybrid.py, run_2_1_sigma_sweep.py | 2.1 | man+cat hybrid with Fourier views; sigma sweep and Fourier-cross check |
| run_2_2_aliasing.py | 2.2 | rings.png: sigma sweep, side-by-side result |
| run_2_3_pyramids.py | 2.3 | Gaussian / Laplacian pyramids, rebuild error, aliasing left by reduce() |
| run_2_4_blending.py | 2.4 | oraple, hand+eye (practice), depth sweeps, level-by-level figure |
| run_3_1_seams.py | 3.1 | energy map, removed seams, zero vs reflect padding |
| run_3_2_comparisons.py, run_3_2_failure_analysis.py | 3.2 | resize / crop / seam carving, both energies, failure case |
| run_bonus.py | Bonus | object removal + seam insertion, k seams from a copy vs repeated cheapest seam |

Photos of your own go through `process_pair` (run_2_1_hybrid.py), `blend_pair` (run_2_4_blending.py), `compare` (run_3_2_comparisons.py) and `object_removal_pipeline` (run_bonus.py).

## What is automatic and what needs your eyes
* **Automatic (pass/fail):** all `tests/test_*.py` (convolution vs scipy and by hand, gradients and DoG, filtering, aliasing mechanism on sine waves, pyramid reconstruction, blending properties, DP vs brute force, coordinate tracking, bonus insertion rules).
* **Visual judgement (not testable):** the edge thresholds (1.2/1.3), "no more false circles" in 2.2, the sigma and level choices in 2.1/2.4, whether a blend still shows a seam, and which seam-carving results look good or fail. The logs give numbers for every choice, but the final call is by eye.

## Known sources of variation
* Timings (1.1 benchmark, seam carving) depend on the machine; quote the table from your own run.
* `align_images` uses cv2.warpAffine: different OpenCV versions interpolate very slightly differently (we measured at most 0.009 in pixel value between OpenCV 4.13 and 5.0), so a few last digits of 2.1 numbers can change.
