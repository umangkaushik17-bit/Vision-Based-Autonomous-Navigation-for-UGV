"""Unit tests for auto_run.py stage dispatcher and dry-run execution."""

import subprocess
import sys
import pytest

def test_auto_run_dry_run_all_stages():
    cmd = [sys.executable, "gpu_package/auto_run.py", "--stage", "all", "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "STAGE 1: HARDWARE & ENVIRONMENT AUDIT" in res.stdout
    assert "STAGE 4: FULL GPU TRAVERSABILITY TRAINING" in res.stdout
    assert "STAGE 8: PACKAGING RESULTS FOR DOWNLOAD" in res.stdout
    assert "ALL GPU TRAINING & EVALUATION STAGES COMPLETED SUCCESSFULLY" in res.stdout

def test_auto_run_individual_stage():
    cmd = [sys.executable, "gpu_package/auto_run.py", "--stage", "smoke-test", "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "STAGE 3: FAST TRAINING SMOKE TEST" in res.stdout
