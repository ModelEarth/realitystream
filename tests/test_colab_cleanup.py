import scripts.colab_cleanup as cleanup


def test_collect_candidate_paths_detects_local_entries(tmp_path):
    params = {
        "features": {"path": "features/data.csv"},
        "targets": {"path": "gs://bucket/example"},  # remote path ignored
        "report": {"output_path": "reports/run"},
        "misc": [
            {"directory": "tmp/cache"},
            {"note": "not a path"},
        ],
    }

    result = cleanup.collect_candidate_paths(params, tmp_path)

    expected = {
        (tmp_path / "features" / "data.csv").resolve(),
        (tmp_path / "reports" / "run").resolve(),
        (tmp_path / "tmp" / "cache").resolve(),
    }
    assert result == expected


def test_cleanup_paths_respects_keep_and_clears_contents(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    (features / "example.csv").write_text("content", encoding="utf-8")

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "report.md").write_text("done", encoding="utf-8")

    keep_path = reports

    result = cleanup.cleanup_paths(
        {features, reports},
        dry_run=False,
        keep={keep_path},
        clear_contents_only=True,
    )

    assert (features / "example.csv").exists() is False
    assert features.exists()
    assert reports.exists()
    assert (reports / "report.md").exists()
    assert features.resolve() in result.removed
    assert reports.resolve() in result.skipped