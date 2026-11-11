"""Run every correctness check locally, on the real GPU: build (which compiles every kernel), the GPU-vs-CPU check
scripts, the conformance sweep (failing on any mismatch or error), and optionally the acceptance table and benchmarks.

usage: uv run --extra dev --extra build tools/check.py [--quick] [--full]
  --quick  skip the build and use 5 samples per op in the sweep
  --full   also run tools/acceptance.py and tools/bench.py (compares with the last saved benchmark)
Exit status is non-zero if anything fails, so it can be used as a git pre-push hook."""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKS = ["native_final_check", "alexnet_ops_check", "activation_check", "conv_kinds_check", "eval_check",
          "grad_accum_check", "storage_check"]
quick, full = "--quick" in sys.argv, "--full" in sys.argv
failed = []


def run(name, cmd):
    t = time.time()
    print(f"== {name}", flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
    print(f"   {'ok' if r.returncode == 0 else 'FAILED'} ({time.time() - t:.0f}s)", flush=True)
    if r.returncode:
        failed.append(name)
    return r.returncode == 0


if not quick and not run("build", ["tools/build_native.py"]):
    sys.exit("build failed")
for c in CHECKS:
    run(c, [f"notes/scripts/{c}.py"])
out = "results/conformance.json"
if run("conformance", ["tools/conformance.py", "--grad", "--limit", "5" if quick else "25", "--out", out]):
    r = json.loads((ROOT / out).read_text())
    bad = {k: v["notes"] for k, v in r["ops"].items() if v["counts"].get("mismatch") or v["counts"].get("error")}
    print("   summary:", r["summary"])
    for k, v in bad.items():
        print(f"   {k}: {v}")
    if bad:
        failed.append("conformance mismatches")
if full:
    run("acceptance", ["tools/acceptance.py"])
    run("bench", ["tools/bench.py"])
print("\nall checks passed" if not failed else f"\nFAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
