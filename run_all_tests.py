"""
run_all_tests.py — sequential regression test runner with Qdrant lock management.

Each test file is a standalone Python script that imports evidencedesk.py,
which opens a Qdrant local database on import.  Because each subprocess holds
the Qdrant lock for the lifetime of the process, we must:

  1. Kill any lingering python processes that may still hold the lock.
  2. Delete the .lock file before each run.
  3. Wait a short period to ensure the OS fully releases the file handle.

This is the documented workaround for the known Qdrant local-mode lock
contention on Windows when running many short-lived test processes.
"""

import subprocess
import sys
import os
import time

TEST_FILES = [
    "test_day08.py",
    "test_day08_fix.py",
    "test_day11.py",
    "test_day12.py",
    "test_day12_bugfix.py",
    "test_metadata.py",
    "test_rrf.py",
]

PYTHON = sys.executable
LOCK_PATH = os.path.join("qdrant_db", ".lock")
WAIT_BETWEEN = 3  # seconds between test runs


def release_qdrant_lock():
    """Delete the Qdrant lock file with retries (do NOT kill the runner process)."""
    for attempt in range(5):
        if not os.path.exists(LOCK_PATH):
            return  # no lock file — good
        try:
            os.remove(LOCK_PATH)
            print(f"  [lock] removed {LOCK_PATH}")
            return
        except (PermissionError, OSError):
            wait = (attempt + 1) * 2
            print(f"  [lock] {LOCK_PATH} still locked, waiting {wait}s (attempt {attempt+1}/5)…")
            time.sleep(wait)
    print(f"  [lock] WARNING: could not remove {LOCK_PATH} after 5 attempts — proceeding anyway")


def main():
    print("Running all regression tests (with Qdrant lock management)…\n")
    all_passed = True

    for tf in TEST_FILES:
        release_qdrant_lock()
        time.sleep(WAIT_BETWEEN)

        print(f"--- Running {tf} ---")
        res = subprocess.run(
            [PYTHON, tf],
            capture_output=True,
            text=True,
        )
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            # Filter out noisy weight-loading progress bars
            stderr_lines = [
                line for line in res.stderr.splitlines()
                if "Loading weights" not in line and line.strip()
            ]
            if stderr_lines:
                print("STDERR:\n", "\n".join(stderr_lines))

        if res.returncode != 0:
            print(f"FAILED: {tf} (exit code {res.returncode})\n")
            all_passed = False
        else:
            print(f"PASSED: {tf}\n")

    if all_passed:
        print("ALL REGRESSION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME REGRESSION TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
