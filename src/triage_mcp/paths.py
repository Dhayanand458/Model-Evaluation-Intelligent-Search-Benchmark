"""Filesystem layout. Every module resolves paths through here, not by hand."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repository root: the directory holding pyproject.toml."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def results_dir() -> Path:
    return project_root() / "results"


def assets_dir() -> Path:
    """Stable plot destination. Overwritten each run; the run dir keeps copies."""
    return project_root() / "assets"


def raw_dir(repo: str) -> Path:
    """Directory holding one JSONL file per fetched API page, for `repo`."""
    return data_dir() / "raw" / repo.replace("/", "__")


def issues_parquet() -> Path:
    return data_dir() / "issues.parquet"
