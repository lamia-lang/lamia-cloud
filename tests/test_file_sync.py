import pytest
from pathlib import Path

from lamia_cloud.file_sync import build_file_sync_plan


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_directory_is_expanded(tmp_path):
    project = tmp_path / "repo"
    _write(project / "docs" / "a.txt", "a")
    _write(project / "docs" / "b.txt", "b")

    entries = build_file_sync_plan(["./docs"], project_root=project, local_home=tmp_path / "home")
    keys = sorted(e.bucket_key for e in entries)
    assert keys == ["docs/a.txt", "docs/b.txt"]


def test_single_file_uploaded(tmp_path):
    project = tmp_path / "repo"
    _write(project / "data.csv", "a,b")

    entries = build_file_sync_plan(["data.csv"], project_root=project, local_home=tmp_path / "home")
    assert [e.bucket_key for e in entries] == ["data.csv"]


def test_missing_path_raises(tmp_path):
    project = tmp_path / "repo"
    project.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="with files\\(\\) path not found"):
        build_file_sync_plan(["./missing"], project_root=project, local_home=tmp_path / "home")


def test_secret_file_raises(tmp_path):
    project = tmp_path / "repo"
    _write(project / ".env", "API_KEY=secret")

    with pytest.raises(ValueError, match="Potential secret file"):
        build_file_sync_plan([".env"], project_root=project, local_home=tmp_path / "home")


def test_multiple_paths_all_included(tmp_path):
    project = tmp_path / "repo"
    _write(project / "docs" / "a.txt", "a")
    _write(project / "data.csv", "x")

    entries = build_file_sync_plan(["./docs", "data.csv"], project_root=project, local_home=tmp_path / "home")
    keys = sorted(e.bucket_key for e in entries)
    assert keys == ["data.csv", "docs/a.txt"]


def test_mixed_dirs_and_files(tmp_path):
    project = tmp_path / "repo"
    _write(project / "assets" / "img.png", "png")
    _write(project / "assets" / "style.css", "css")
    _write(project / "config.json", "{}")

    entries = build_file_sync_plan(["./assets", "config.json"], project_root=project, local_home=tmp_path / "home")
    keys = sorted(e.bucket_key for e in entries)
    assert keys == ["assets/img.png", "assets/style.css", "config.json"]


def test_nested_directory_all_files_included(tmp_path):
    project = tmp_path / "repo"
    _write(project / "data" / "sub" / "deep.txt", "d")
    _write(project / "data" / "top.txt", "t")

    entries = build_file_sync_plan(["./data"], project_root=project, local_home=tmp_path / "home")
    keys = sorted(e.bucket_key for e in entries)
    assert keys == ["data/sub/deep.txt", "data/top.txt"]
