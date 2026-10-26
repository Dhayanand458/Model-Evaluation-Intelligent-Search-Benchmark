"""Resume behaviour. An interrupted harvest must never restart from page 1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_mcp import ingest
from triage_mcp.ingest import (
    Checkpoint,
    clear_raw,
    count_issues_on_disk,
    load_checkpoint,
    next_page_to_fetch,
    read_raw,
    save_checkpoint,
    write_page,
)

REPO = "owner/name"


@pytest.fixture(autouse=True)
def raw_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the raw store at a temp dir so tests never touch data/."""
    root = tmp_path / "raw"
    root.mkdir()
    monkeypatch.setattr(ingest, "raw_dir", lambda repo: root)
    return root


def _page(numbers: list[int], pulls: set[int] | None = None) -> list[dict[str, object]]:
    pulls = pulls or set()
    items: list[dict[str, object]] = []
    for n in numbers:
        item: dict[str, object] = {
            "number": n,
            "title": f"issue {n}",
            "labels": [],
            "state": "open",
            "created_at": "2024-01-01T00:00:00Z",
            "comments": 0,
        }
        if n in pulls:
            item["pull_request"] = {"url": "…"}
        items.append(item)
    return items


def test_first_run_starts_at_page_one() -> None:
    assert next_page_to_fetch(REPO) == 1


def test_resume_continues_after_last_written_page() -> None:
    for page in range(1, 4):
        write_page(REPO, page, _page([page * 10]))

    assert next_page_to_fetch(REPO) == 4


def test_resume_refetches_a_gap_rather_than_skipping_it() -> None:
    """A page lost mid-run must be refetched, not stepped over."""
    for page in (1, 2, 4, 5):
        write_page(REPO, page, _page([page * 10]))

    assert next_page_to_fetch(REPO) == 3


def test_resume_survives_a_missing_checkpoint() -> None:
    """Page files alone are enough to resume."""
    write_page(REPO, 1, _page([1, 2, 3]))
    write_page(REPO, 2, _page([4, 5]))

    assert load_checkpoint(REPO) is None
    assert next_page_to_fetch(REPO) == 3
    assert count_issues_on_disk(REPO) == 5


def test_resume_survives_a_corrupt_checkpoint(raw_in_tmp: Path) -> None:
    write_page(REPO, 1, _page([1, 2]))
    (raw_in_tmp / "_checkpoint.json").write_text("{ not json", encoding="utf-8")

    assert load_checkpoint(REPO) is None  # reported, not fatal
    assert next_page_to_fetch(REPO) == 2  # progress preserved


def test_checkpoint_with_a_byte_order_mark_still_parses(raw_in_tmp: Path) -> None:
    """PowerShell's Out-File -Encoding utf8 prepends a BOM; tolerate it."""
    bom = chr(0xFEFF)
    save_checkpoint(Checkpoint(repo=REPO, last_page=7, issues_collected=42))
    path = raw_in_tmp / "_checkpoint.json"
    path.write_text(bom + path.read_text(encoding="utf-8"), encoding="utf-8")

    loaded = load_checkpoint(REPO)

    assert loaded is not None
    assert loaded.last_page == 7


def test_checkpoint_from_another_repo_is_ignored() -> None:
    save_checkpoint(Checkpoint(repo="someone/else", last_page=9))

    assert load_checkpoint(REPO) is None


def test_checkpoint_round_trips() -> None:
    save_checkpoint(Checkpoint(repo=REPO, last_page=5, issues_collected=300, exhausted=True))
    loaded = load_checkpoint(REPO)

    assert loaded is not None
    assert (loaded.last_page, loaded.issues_collected, loaded.exhausted) == (5, 300, True)


def test_raw_pages_are_written_verbatim(raw_in_tmp: Path) -> None:
    """The raw store keeps PRs, so the filter stays auditable after the fact."""
    write_page(REPO, 1, _page([1, 2, 3], pulls={2}))

    lines = (raw_in_tmp / "page-0001.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 3
    assert "pull_request" in json.loads(lines[1])


def test_only_issues_count_toward_progress() -> None:
    write_page(REPO, 1, _page([1, 2, 3, 4], pulls={2, 4}))

    assert count_issues_on_disk(REPO) == 2


def test_read_raw_streams_pages_in_order() -> None:
    write_page(REPO, 2, _page([30, 40]))
    write_page(REPO, 1, _page([10, 20]))
    write_page(REPO, 10, _page([100]))

    assert [item["number"] for item in read_raw(REPO)] == [10, 20, 30, 40, 100]


def test_fresh_clears_the_raw_store(raw_in_tmp: Path) -> None:
    for page in range(1, 4):
        write_page(REPO, page, _page([page]))
    save_checkpoint(Checkpoint(repo=REPO, last_page=3))

    removed = clear_raw(REPO)

    assert removed == 3
    assert next_page_to_fetch(REPO) == 1
    assert not (raw_in_tmp / "_checkpoint.json").exists()
