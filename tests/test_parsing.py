"""Parsing raw GitHub payloads into validated Issue records."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from triage_mcp.ingest import Issue, compute_stats, issues_to_frame, parse_issues


def _by_number(issues_page: list[dict[str, Any]], number: int) -> Issue:
    return next(issue for issue in parse_issues(issues_page) if issue.number == number)


def test_parses_core_fields(issues_page: list[dict[str, Any]]) -> None:
    issue = _by_number(issues_page, 201)

    assert issue.title == "Editor freezes when opening large minified JS"
    assert issue.body is not None and issue.body.startswith("Opening a 40MB")
    assert issue.state == "open"
    assert issue.comments == 14
    assert issue.created_at == datetime(2024, 3, 11, 9, 22, 41, tzinfo=UTC)
    assert issue.closed_at is None


def test_labels_reduce_to_names(issues_page: list[dict[str, Any]]) -> None:
    assert _by_number(issues_page, 201).labels == ["bug", "perf"]
    assert _by_number(issues_page, 205).labels == ["feature-request", "editor", "under-discussion"]


def test_labels_accept_plain_strings() -> None:
    """Parquet round-trips labels as strings; re-validating must still work."""
    issue = Issue.model_validate(
        {
            "number": 7,
            "title": "t",
            "labels": ["bug", "perf"],
            "state": "open",
            "created_at": "2024-01-01T00:00:00Z",
        }
    )

    assert issue.labels == ["bug", "perf"]


def test_labels_accept_numpy_arrays() -> None:
    """pd.read_parquet hands list columns back as ndarrays, not lists."""
    issue = Issue.model_validate(
        {
            "number": 8,
            "title": "t",
            "labels": np.array(["bug", "perf"], dtype=object),
            "state": "open",
            "created_at": "2024-01-01T00:00:00Z",
        }
    )

    assert issue.labels == ["bug", "perf"]


@pytest.mark.parametrize("value", ["bug", 42, {"name": "bug"}, None])
def test_malformed_labels_yield_no_labels(value: Any) -> None:
    """A bare string must not be iterated into ['b', 'u', 'g']."""
    issue = Issue.model_validate(
        {
            "number": 9,
            "title": "t",
            "labels": value,
            "state": "open",
            "created_at": "2024-01-01T00:00:00Z",
        }
    )

    assert issue.labels == []


def test_missing_body_becomes_none(issues_page: list[dict[str, Any]]) -> None:
    issue = _by_number(issues_page, 203)

    assert issue.body is None
    assert issue.labels == []


def test_closed_issue_carries_state_reason(issues_page: list[dict[str, Any]]) -> None:
    issue = _by_number(issues_page, 203)

    assert issue.state == "closed"
    assert issue.state_reason == "not_planned"
    assert issue.closed_at == datetime(2024, 2, 9, 12, 30, tzinfo=UTC)


def test_open_issue_has_no_state_reason(issues_page: list[dict[str, Any]]) -> None:
    assert _by_number(issues_page, 201).state_reason is None


def test_unknown_api_fields_are_ignored(issues_page: list[dict[str, Any]]) -> None:
    """The payload carries user/html_url/etc.; the model keeps only its schema."""
    issue = _by_number(issues_page, 201)

    assert set(issue.model_dump()) == {
        "number",
        "title",
        "body",
        "labels",
        "state",
        "state_reason",
        "created_at",
        "closed_at",
        "comments",
    }


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Issue.model_validate({"title": "no number", "state": "open"})


def test_frame_schema_and_ordering(issues_page: list[dict[str, Any]]) -> None:
    frame = issues_to_frame(parse_issues(issues_page), repo="microsoft/vscode")

    assert list(frame.columns)[:3] == ["repo", "number", "title"]
    assert frame["repo"].unique().tolist() == ["microsoft/vscode"]
    # Newest first.
    assert frame["number"].tolist() == [201, 203, 205]


def test_empty_input_yields_empty_frame() -> None:
    frame = issues_to_frame([], repo="owner/name")

    assert frame.empty
    assert "number" in frame.columns


def test_stats_are_counted_from_the_frame(issues_page: list[dict[str, Any]]) -> None:
    frame = issues_to_frame(parse_issues(issues_page), repo="microsoft/vscode")
    stats = compute_stats(frame, top=20)

    assert stats["total_issues"] == 3
    assert stats["labelled_issues"] == 2  # #203 has none
    assert stats["pct_with_label"] == pytest.approx(66.67)
    assert stats["distinct_labels"] == 5
    assert dict(stats["top_labels"]) == {
        "bug": 1,
        "perf": 1,
        "feature-request": 1,
        "editor": 1,
        "under-discussion": 1,
    }


def test_top_labels_is_truncated() -> None:
    issues = parse_issues(
        [
            {
                "number": n,
                "title": f"issue {n}",
                "labels": [{"name": f"label-{n}"}],
                "state": "open",
                "created_at": "2024-01-01T00:00:00Z",
            }
            for n in range(10)
        ]
    )
    stats = compute_stats(issues_to_frame(issues, repo="owner/name"), top=3)

    assert len(stats["top_labels"]) == 3
    assert stats["distinct_labels"] == 10


def test_parquet_round_trip(issues_page: list[dict[str, Any]], tmp_path: Path) -> None:
    frame = issues_to_frame(parse_issues(issues_page), repo="microsoft/vscode")
    path = tmp_path / "issues.parquet"
    frame.to_parquet(path, index=False)

    restored = pd.read_parquet(path)

    assert restored["number"].tolist() == [201, 203, 205]
    assert list(restored.loc[0, "labels"]) == ["bug", "perf"]
