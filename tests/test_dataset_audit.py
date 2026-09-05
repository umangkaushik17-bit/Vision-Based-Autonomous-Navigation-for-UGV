"""Unit tests for dataset audit and statistical health analyzer."""

import pytest
from ai.datasets.audit import DatasetAuditor, AuditReport

def test_dataset_audit_metrics():
    manifest_path = "data/manifests/dataset_manifest.json"
    auditor = DatasetAuditor(manifest_path)
    report = auditor.audit(max_pixel_samples=10)

    assert isinstance(report, AuditReport)
    assert report.total_samples > 0
    assert report.zero_leakage_verified is True
    assert "train" in report.split_counts
    assert "val" in report.split_counts
    assert "test" in report.split_counts
    assert "UGV DATASET AUDIT REPORT" in report.summary_string()
