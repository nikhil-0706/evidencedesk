import subprocess
import sys
import os

test_files = [
    "test_day08.py",
    "test_day08_fix.py",
    "test_day11.py",
    "test_day12.py",
    "test_day12_bugfix.py",
    "test_metadata.py",
    "test_rrf.py"
]

python_exe = sys.executable

print("Running all regression tests...")
all_passed = True

for tf in test_files:
    if os.path.exists("qdrant_db/.lock"):
        try:
            os.remove("qdrant_db/.lock")
        except Exception:
            pass
    print(f"\n--- Running {tf} ---")
    res = subprocess.run([python_exe, tf], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)
    if res.returncode != 0:
        print(f"FAILED: {tf} with exit code {res.returncode}")
        all_passed = False
    else:
        print(f"PASSED: {tf}")

if all_passed:
    print("\nALL REGRESSION TESTS PASSED SUCCESSFULLY!")
else:
    print("\nSOME REGRESSION TESTS FAILED!")
    sys.exit(1)
