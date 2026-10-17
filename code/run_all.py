"""Run the whole project: all automatic tests, then every experiment script.  Works from any working directory.

    python code/run_all.py                  # tests + all experiments (the 1.1 benchmark alone takes ~2-3 minutes)
    python code/run_all.py --skip-benchmark # everything except the slow 1.1 timing benchmark
    python code/run_all.py --tests-only

Logs are written to results/logs/<script>.txt, figures to results/.  The starter pack's images/ folder must sit next to code/.
"""
import subprocess, sys, time
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
LOGS = ROOT / "results" / "logs"
TESTS = ["test_conv", "test_edges", "test_freq", "test_aliasing", "test_pyramids", "test_blend", "test_seam", "test_bonus"]
SCRIPTS = [("run_1_2_1_3.py", []), ("run_2_1_hybrid.py", []), ("run_2_1_sigma_sweep.py", []), ("run_2_2_aliasing.py", ["2.5"]),
           ("run_2_3_pyramids.py", []), ("run_2_4_blending.py", []), ("run_3_1_seams.py", []), ("run_3_2_comparisons.py", []),
           ("run_3_2_failure_analysis.py", []), ("run_bonus.py", []), ("run_1_1_benchmark.py", [])]


def run(cmd, log=None):
    t = time.time()
    p = subprocess.run([sys.executable, "-u"] + cmd, cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if log:
        log.write_text(out)
    return p.returncode, time.time() - t, out


def main(argv):
    if not (ROOT / "images" / "part1_filters" / "facade.png").exists():
        sys.exit(f"Missing {ROOT / 'images'}: copy the starter pack's images/ folder (part1_filters, part2_frequencies, part3_seams) next to code/.")
    LOGS.mkdir(parents=True, exist_ok=True)
    failed = []
    print("AUTOMATIC TESTS")
    for t in TESTS:
        rc, secs, out = run([str(CODE / "tests" / f"{t}.py")])
        last = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        print(f"  {t:15s} {'OK  ' if rc == 0 else 'FAIL'} {secs:5.1f} s   {last}")
        if rc != 0:
            failed.append(t); print(out[-1500:])
    if "--tests-only" not in argv:
        print("\nEXPERIMENTS (figures -> results/, logs -> results/logs/)")
        for name, args in SCRIPTS:
            if name == "run_1_1_benchmark.py" and "--skip-benchmark" in argv:
                print(f"  {name:32s} skipped"); continue
            rc, secs, out = run([str(CODE / name)] + args, LOGS / (name[:-3] + ".txt"))
            print(f"  {name:32s} {'OK  ' if rc == 0 else 'FAIL'} {secs:6.1f} s")
            if rc != 0:
                failed.append(name); print(out[-1500:])
    print("\nALL PASSED" if not failed else f"\nFAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
