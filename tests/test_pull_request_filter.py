"""The list-issues endpoint returns PRs too. Dropping them is load-bearing."""

from __future__ import annotations

from typing import Any

from triage_mcp.ingest import is_pull_request, parse_issues


def test_issue_is_not_a_pull_request() -> None:
    assert is_pull_request({"number": 1, "title": "a bug"}) is False


def test_pull_request_key_marks_a_pr() -> None:
    item = {"number": 2, "pull_request": {"url": "https://api.github.com/…/pulls/2"}}
    assert is_pull_request(item) is True


def test_null_pull_request_value_still_marks_a_pr() -> None:
    """Key presence is authoritative -- a real issue never carries the key."""
    assert is_pull_request({"number": 3, "pull_request": None}) is True
    assert is_pull_request({"number": 4, "pull_request": {}}) is True


def test_parse_drops_pull_requests(issues_page: list[dict[str, Any]]) -> None:
    numbers = [issue.number for issue in parse_issues(issues_page)]

    assert 202 not in numbers, "#202 is a PR and must not enter the corpus"
    assert 204 not in numbers, "#204 has pull_request: null and is still a PR"
    assert numbers == [201, 203, 205]


def test_parse_keeps_every_genuine_issue(issues_page: list[dict[str, Any]]) -> None:
    """Guard against an over-eager filter quietly discarding real issues."""
    non_pr_numbers = {item["number"] for item in issues_page if "pull_request" not in item}

    assert {issue.number for issue in parse_issues(issues_page)} == non_pr_numbers


def test_parse_dedupes_repeated_numbers(issues_page: list[dict[str, Any]]) -> None:
    """Paging a repo sorted by created date can serve the same issue twice."""
    issues = parse_issues(issues_page)

    assert len(issues) == 3
    assert len({issue.number for issue in issues}) == 3
