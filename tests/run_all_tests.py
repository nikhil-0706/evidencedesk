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
    os.path.join("tests", "regression", "test_day08.py"),
    os.path.join("tests", "regression", "test_day08_fix.py"),
    os.path.join("tests", "regression", "test_day11.py"),
    os.path.join("tests", "regression", "test_day12.py"),
    os.path.join("tests", "regression", "test_day12_bugfix.py"),
    os.path.join("tests", "unit", "test_metadata.py"),
    os.path.join("tests", "unit", "test_rrf.py"),
    os.path.join("tests", "regression", "test_day16.py"),
    os.path.join("tests", "regression", "test_day17.py"),
    os.path.join("tests", "regression", "test_day18.py"),
    os.path.join("tests", "regression", "test_day19.py"),
    os.path.join("tests", "regression", "test_day19_5.py"),
    os.path.join("tests", "regression", "test_day19_75.py"),
    os.path.join("tests", "regression", "test_day20.py"),
    os.path.join("tests", "regression", "test_day21_chunking_expansion.py"),
    os.path.join("tests", "regression", "test_day22_caiq.py"),
    os.path.join("tests", "regression", "test_caiq_grounded_qa.py"),
    os.path.join("tests", "regression", "test_day23_audit.py"),
    os.path.join("tests", "regression", "test_workspace_upload_isolation.py"),
    os.path.join("evaluations", "run_final_evaluation.py"),
]


venv_python = os.path.join(".venv", "Scripts", "python.exe")
PYTHON = venv_python if os.path.exists(venv_python) else sys.executable
LOCK_PATH = os.path.join("qdrant_db", ".lock")
WAIT_BETWEEN = 1  # seconds between test runs



import shutil

TEST_DB_DIR = "qdrant_test_db"

def reset_qdrant_db():
    """Delete the entire Qdrant database directory with retries for full test isolation."""
    # Safety check: ensure we NEVER delete anything other than the designated test db
    if TEST_DB_DIR != "qdrant_test_db":
        raise ValueError("Safety violation: attempted to reset non-test database")
        
    for attempt in range(5):
        if not os.path.exists(TEST_DB_DIR):
            return
        try:
            shutil.rmtree(TEST_DB_DIR)
            print(f"  [db] removed {TEST_DB_DIR} directory")
            return
        except (PermissionError, OSError):
            wait = (attempt + 1) * 2
            print(f"  [db] {TEST_DB_DIR} locked, waiting {wait}s (attempt {attempt+1}/5)…")
            time.sleep(wait)
    print(f"  [db] WARNING: could not remove {TEST_DB_DIR} after 5 attempts — proceeding anyway")


def main():
    print("Running all regression tests (with Qdrant lock management)…\n")
    all_passed = True

    for tf in TEST_FILES:
        reset_qdrant_db()
        time.sleep(WAIT_BETWEEN)

        env = os.environ.copy()
        env["QDRANT_PATH"] = TEST_DB_DIR
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        print(f"--- Running {tf} ---")
        res = subprocess.run(
            [PYTHON, tf],
            capture_output=True,
            text=True, encoding='utf-8',
            env=env
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
