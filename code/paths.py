"""Project folders, resolved relative to THIS file so every script works from any working directory.

    cv_assignment1/
        code/      (this file, the modules, run_*.py, tests/)
        images/    the starter pack's images/ folder (part1_filters/, part2_frequencies/, part3_seams/)
        results/   figures and logs written by the run_*.py scripts
"""
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
IMAGES = str(ROOT / "images")
RESULTS = str(ROOT / "results")
