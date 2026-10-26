"""Harvest historical GitHub issues into a validated local dataset.

The GitHub "list repository issues" endpoint returns pull requests alongside
issues. Every item carrying a ``pull_request`` key is a PR and is dropped before
anything is counted or persisted -- see :func:`is_pull_request`.

Issue *comments* are deliberately not fetched. The list endpoint provides a
comment count only; bodies would cost one request per issue.

CLI::

    python -m triage_mcp.ingest --repo microsoft/vscode --max 3000
    python -m triage_mcp.ingest stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from triage_mcp.paths import issues_parquet, raw_dir

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "microsoft/vscode"
DEFAULT_MAX_ISSUES = 3000
PER_PAGE = 100
MAX_RETRIES = 6
REQUEST_TIMEOUT = 30.0

# GitHub serves at most this many items from a paginated list resource and
# answers 422 beyond it. With ~43% of vscode's items being PRs, the ceiling
# works out to roughly 5.7k issues per (repo, sort order).
PAGINATION_CEILING = 10_000

# Never block longer than this on a single rate-limit wait. A longer wait means
# the run should stop and be resumed later from its checkpoint.
MAX_SLEEP_SECONDS = 900.0

CHECKPOINT_NAME = "_checkpoint.json"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class Issue(BaseModel):
    """One GitHub issue. Never a pull request."""

    model_config = ConfigDict(extra="ignore")

    number: int
    title: str
    body: str | None = None
    labels: list[str] = Field(default_factory=list)
    state: str
    state_reason: str | None = None
    created_at: datetime
    closed_at: datetime | None = None
    comments: int = 0

    @field_validator("labels", mode="before")
    @classmethod
    def _label_names(cls, value: object) -> list[str]:
        """Reduce GitHub's label objects to their names.

        The API returns ``[{"name": "bug", ...}]``, but the same field can come
        back as plain strings from some endpoints and from our own round-tripped
        Parquet, so accept both. Anything that is not a sequence of those is
        treated as no labels -- notably a bare string, which would otherwise
        iterate character by character.
        """
        if not isinstance(value, (list, tuple, np.ndarray)):
            return []
        names: list[str] = []
        for label in value:
            if isinstance(label, str):
                names.append(label)
            elif isinstance(label, dict) and label.get("name") is not None:
                names.append(str(label["name"]))
        return names


class Checkpoint(BaseModel):
    """Resume state for an interrupted harvest, written after every page."""

    repo: str
    last_page: int = 0
    issues_collected: int = 0
    exhausted: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def is_pull_request(raw: dict[str, Any]) -> bool:
    """True when a list-issues item is really a pull request.

    GitHub tags PRs with a ``pull_request`` key. Presence of the key is
    authoritative -- treat it as a PR even if the value is null or empty, since
    a genuine issue never carries the key at all.
    """
    return "pull_request" in raw


def parse_issues(raw_items: Iterable[dict[str, Any]]) -> list[Issue]:
    """Validate raw API items into :class:`Issue` records.

    Pull requests are dropped. Duplicate issue numbers are dropped too: paging
    a repo sorted by creation date shifts under us whenever a new issue lands
    mid-run, so the same item can appear on two pages.
    """
    issues: list[Issue] = []
    seen: set[int] = set()
    for raw in raw_items:
        if is_pull_request(raw):
            continue
        number = raw.get("number")
        if number is None or number in seen:
            continue
        seen.add(number)
        issues.append(Issue.model_validate(raw))
    return issues


def issues_to_frame(issues: list[Issue], repo: str) -> pd.DataFrame:
    """Build the tabular dataset. Column order is the on-disk schema."""
    columns = [
        "repo",
        "number",
        "title",
        "body",
        "labels",
        "state",
        "state_reason",
        "created_at",
        "closed_at",
        "comments",
    ]
    rows = [{"repo": repo, **issue.model_dump()} for issue in issues]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        # Give an empty frame the same dtypes a populated one would have, so a
        # zero-result run still writes a readable Parquet file.
        return frame.astype(
            {"number": "int64", "comments": "int64", "repo": "object", "title": "object"}
        )
    return frame.sort_values("created_at", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def github_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "triage-mcp",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def retry_delay(response: httpx.Response, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None when the response is final.

    Handles the three ways GitHub asks a client to slow down: ``retry-after``
    (secondary limits), an exhausted ``x-ratelimit-remaining`` with a reset
    timestamp (primary limit), and plain 5xx flakiness.
    """
    if response.status_code < 400:
        return None

    retry_after = _as_float(response.headers.get("retry-after"))
    if retry_after is not None:
        return min(retry_after, MAX_SLEEP_SECONDS)

    if response.status_code in (403, 429) and response.headers.get("x-ratelimit-remaining") == "0":
        reset = _as_float(response.headers.get("x-ratelimit-reset"))
        if reset is not None:
            return min(max(reset - time.time(), 0.0) + 1.0, MAX_SLEEP_SECONDS)

    if response.status_code == 429 or response.status_code >= 500:
        return min(2.0**attempt, 60.0)

    return None


def throttle(response: httpx.Response) -> None:
    """Sleep out the window when a *successful* response reports zero quota left.

    Cheaper than taking the 403 on the next call and backing off from it.
    """
    if response.headers.get("x-ratelimit-remaining") != "0":
        return
    reset = _as_float(response.headers.get("x-ratelimit-reset"))
    if reset is None:
        return
    wait = min(max(reset - time.time(), 0.0) + 1.0, MAX_SLEEP_SECONDS)
    if wait > 0:
        print(f"[ingest] rate limit exhausted; sleeping {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)


def is_pagination_limit(response: httpx.Response) -> bool:
    """True when a 422 means "you have paged past the ceiling", not a bad request.

    Hitting the ceiling is the normal end of a deep harvest, so it has to stop
    the run cleanly rather than raise -- otherwise every page already fetched is
    stranded without a parsed dataset.
    """
    if response.status_code != 422:
        return False
    try:
        message = str(response.json().get("message", ""))
    except ValueError:
        message = response.text
    return "pagination" in message.lower()


def fetch_page(client: httpx.Client, repo: str, page: int) -> list[dict[str, Any]]:
    """Fetch one page of issues, retrying on rate limits and 5xx.

    Returns an empty list at the end of the repo *or* at GitHub's pagination
    ceiling; the caller treats both as exhaustion.
    """
    params = {
        "state": "all",
        "sort": "created",
        "direction": "desc",
        "per_page": PER_PAGE,
        "page": page,
    }
    for attempt in range(MAX_RETRIES):
        response = client.get(f"/repos/{repo}/issues", params=params)
        if response.status_code == 200:
            throttle(response)
            payload = response.json()
            return payload if isinstance(payload, list) else []

        if is_pagination_limit(response):
            print(
                f"[ingest] page {page}: reached GitHub's pagination ceiling "
                f"({PAGINATION_CEILING} items per list query); stopping here",
                file=sys.stderr,
            )
            return []

        delay = retry_delay(response, attempt)
        if delay is None:
            response.raise_for_status()
        print(
            f"[ingest] HTTP {response.status_code} on page {page}; "
            f"retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s",
            file=sys.stderr,
        )
        time.sleep(delay)

    raise RuntimeError(f"giving up on page {page} of {repo} after {MAX_RETRIES} retries")


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Search-window harvest
#
# The list endpoint stops at PAGINATION_CEILING items, which for a busy repo is
# only a couple of months. Search slices by creation date instead, so an
# arbitrarily long history can be walked one window at a time. It costs a
# tighter rate limit (30 req/min) and a 1000-result cap per query, so windows
# that overflow are split rather than silently truncated.
# --------------------------------------------------------------------------- #

SEARCH_ENDPOINT = "/search/issues"
SEARCH_RESULT_CAP = 1000
SEARCH_MIN_INTERVAL = 2.1  # seconds between calls, to stay under 30/min
DEFAULT_WINDOW_DAYS = 7


def window_path(repo: str, start: date, end: date) -> Path:
    return raw_dir(repo) / f"search-{start:%Y%m%d}-{end:%Y%m%d}.jsonl"


def date_windows(since: date, until: date, window_days: int) -> list[tuple[date, date]]:
    """Contiguous inclusive [start, end] windows covering the range."""
    windows: list[tuple[date, date]] = []
    cursor = since
    while cursor <= until:
        end = min(cursor + timedelta(days=window_days - 1), until)
        windows.append((cursor, end))
        cursor = end + timedelta(days=1)
    return windows


def search_page(
    client: httpx.Client, repo: str, start: date, end: date, page: int
) -> tuple[int, list[dict[str, Any]]]:
    """One page of a date-window search. Returns (total_count, items)."""
    params = {
        "q": f"repo:{repo} type:issue created:{start:%Y-%m-%d}..{end:%Y-%m-%d}",
        "sort": "created",
        "order": "asc",
        "per_page": PER_PAGE,
        "page": page,
    }
    for attempt in range(MAX_RETRIES):
        response = client.get(SEARCH_ENDPOINT, params=params)
        if response.status_code == 200:
            payload = response.json()
            return int(payload.get("total_count", 0)), list(payload.get("items", []))

        delay = retry_delay(response, attempt)
        if delay is None:
            response.raise_for_status()
        print(
            f"[ingest] HTTP {response.status_code} on {start}..{end} p{page}; "
            f"retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s",
            file=sys.stderr,
        )
        time.sleep(delay)

    raise RuntimeError(f"giving up on search window {start}..{end} after {MAX_RETRIES} retries")


def harvest_window(client: httpx.Client, repo: str, start: date, end: date, depth: int = 0) -> int:
    """Fetch every issue created in [start, end], splitting if over the cap.

    A window holding more than SEARCH_RESULT_CAP issues cannot be read out in
    full, so it is halved and retried. Splitting beats truncating: a silently
    clipped window would look like a quiet stretch in the corpus rather than an
    error.
    """
    path = window_path(repo, start, end)
    if path.exists():
        return sum(1 for _ in path.open(encoding="utf-8"))

    time.sleep(SEARCH_MIN_INTERVAL)
    total, items = search_page(client, repo, start, end, page=1)

    if total > SEARCH_RESULT_CAP and start < end:
        midpoint = start + timedelta(days=max((end - start).days // 2, 0))
        print(
            f"[ingest] {start}..{end} holds {total} issues (> {SEARCH_RESULT_CAP}); splitting",
            file=sys.stderr,
        )
        left = harvest_window(client, repo, start, midpoint, depth + 1)
        right = harvest_window(client, repo, midpoint + timedelta(days=1), end, depth + 1)
        return left + right

    collected = list(items)
    pages = min(-(-min(total, SEARCH_RESULT_CAP) // PER_PAGE), SEARCH_RESULT_CAP // PER_PAGE)
    for page in range(2, pages + 1):
        time.sleep(SEARCH_MIN_INTERVAL)
        _, more = search_page(client, repo, start, end, page)
        if not more:
            break
        collected.extend(more)

    with path.open("w", encoding="utf-8") as handle:
        for item in collected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    indent = "  " * depth
    print(f"[ingest] {indent}{start}..{end}: {len(collected)} issues", file=sys.stderr)
    return len(collected)


def harvest_by_search(
    repo: str,
    since: date,
    until: date,
    token: str | None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> int:
    """Walk [since, until] in date windows, skipping windows already on disk."""
    raw_dir(repo).mkdir(parents=True, exist_ok=True)
    windows = date_windows(since, until, window_days)
    done = sum(1 for start, end in windows if window_path(repo, start, end).exists())
    print(
        f"[ingest] {len(windows)} windows of {window_days}d over {since}..{until} "
        f"({done} already on disk)",
        file=sys.stderr,
    )

    total = 0
    with httpx.Client(
        base_url=GITHUB_API,
        headers=github_headers(token),
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:
        for index, (start, end) in enumerate(windows, start=1):
            total += harvest_window(client, repo, start, end)
            if index % 10 == 0:
                print(f"[ingest] {index}/{len(windows)} windows, {total} issues", file=sys.stderr)
    return total


def load_checkpoint(repo: str) -> Checkpoint | None:
    """Read the advisory checkpoint, or None if absent/unusable.

    Only the ``exhausted`` flag is actually trusted from here -- resume position
    comes from :func:`next_page_to_fetch`. A corrupt checkpoint is reported and
    ignored, never treated as "start over".
    """
    path = raw_dir(repo) / CHECKPOINT_NAME
    if not path.exists():
        return None
    try:
        # utf-8-sig tolerates a BOM, which some editors and PowerShell add.
        checkpoint = Checkpoint.model_validate_json(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        print(
            f"[ingest] unreadable checkpoint ({exc}); falling back to page files",
            file=sys.stderr,
        )
        return None
    # A checkpoint from a different repo must never resume this one.
    return checkpoint if checkpoint.repo == repo else None


def save_checkpoint(checkpoint: Checkpoint) -> None:
    path = raw_dir(checkpoint.repo) / CHECKPOINT_NAME
    checkpoint.updated_at = datetime.now(UTC)
    path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")


def page_path(repo: str, page: int) -> Path:
    return raw_dir(repo) / f"page-{page:04d}.jsonl"


def write_page(repo: str, page: int, items: list[dict[str, Any]]) -> Path:
    """Persist one page verbatim, pull requests included.

    The raw store stays faithful to what the API returned; filtering happens
    once, at parse time, so the drop is auditable after the fact.
    """
    path = page_path(repo, page)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def next_page_to_fetch(repo: str) -> int:
    """Lowest page number with no file on disk.

    Page files -- not the checkpoint JSON -- are the source of truth for resume:
    a missing, stale, or corrupt checkpoint must never trigger a full
    re-harvest. Scanning for the first *gap* rather than the highest page also
    means a page lost mid-run is refetched instead of skipped.
    """
    page = 1
    while page_path(repo, page).exists():
        page += 1
    return page


def clear_raw(repo: str) -> int:
    """Delete every harvested page and the checkpoint for `repo` (``--fresh``)."""
    removed = 0
    for path in raw_dir(repo).glob("page-*.jsonl"):
        path.unlink()
        removed += 1
    checkpoint = raw_dir(repo) / CHECKPOINT_NAME
    if checkpoint.exists():
        checkpoint.unlink()
    return removed


def read_raw(repo: str) -> Iterator[dict[str, Any]]:
    """Stream every raw item previously written for `repo`.

    Covers both harvest modes -- ``page-*.jsonl`` from the list endpoint and
    ``search-*.jsonl`` from date-window search. Overlap between them is expected
    and harmless: :func:`parse_issues` dedupes by issue number.
    """
    for path in sorted(raw_dir(repo).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def count_issues_on_disk(repo: str) -> int:
    """Number of non-PR items already harvested, used to resume accurately."""
    return sum(1 for raw in read_raw(repo) if not is_pull_request(raw))


# --------------------------------------------------------------------------- #
# Harvest
# --------------------------------------------------------------------------- #


def harvest(repo: str, max_issues: int, token: str | None, fresh: bool = False) -> Checkpoint:
    """Page through the issues endpoint until `max_issues` issues are stored.

    Only non-PR items count toward `max_issues`, so a PR-heavy repo simply costs
    more pages. Progress is checkpointed after every page; re-running resumes
    from the next unfetched page unless `fresh` is set.
    """
    out = raw_dir(repo)
    out.mkdir(parents=True, exist_ok=True)

    if fresh:
        removed = clear_raw(repo)
        print(f"[ingest] --fresh: discarded {removed} previously harvested pages", file=sys.stderr)

    # Reconstruct state from disk, then take only `exhausted` from the advisory
    # checkpoint. Counting is cheap at this corpus size and always correct, even
    # after a partially written final page.
    start_page = next_page_to_fetch(repo)
    stored = load_checkpoint(repo)
    checkpoint = Checkpoint(
        repo=repo,
        last_page=start_page - 1,
        issues_collected=count_issues_on_disk(repo),
        exhausted=bool(stored and stored.exhausted),
    )

    if start_page > 1:
        print(
            f"[ingest] resuming {repo} at page {start_page} "
            f"({checkpoint.issues_collected} issues on disk)",
            file=sys.stderr,
        )

    if checkpoint.issues_collected >= max_issues:
        print(f"[ingest] already have {checkpoint.issues_collected} issues; nothing to fetch")
        return checkpoint
    if checkpoint.exhausted:
        print(f"[ingest] repo exhausted at {checkpoint.issues_collected} issues")
        return checkpoint

    with httpx.Client(
        base_url=GITHUB_API,
        headers=github_headers(token),
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:
        page = start_page
        while checkpoint.issues_collected < max_issues:
            items = fetch_page(client, repo, page)
            if not items:
                checkpoint.exhausted = True
                save_checkpoint(checkpoint)
                break

            write_page(repo, page, items)
            pulls = sum(1 for item in items if is_pull_request(item))
            checkpoint.issues_collected += len(items) - pulls
            checkpoint.last_page = page
            save_checkpoint(checkpoint)

            print(
                f"[ingest] page {page}: {len(items) - pulls} issues, {pulls} PRs skipped "
                f"({checkpoint.issues_collected}/{max_issues})",
                file=sys.stderr,
            )

            if len(items) < PER_PAGE:
                checkpoint.exhausted = True
                save_checkpoint(checkpoint)
                break
            page += 1

    return checkpoint


def build_dataset(repo: str, max_issues: int | None = None) -> pd.DataFrame:
    """Parse the raw store into a validated Parquet dataset."""
    issues = parse_issues(read_raw(repo))
    if max_issues is not None:
        issues = issues[:max_issues]
    frame = issues_to_frame(issues, repo)
    path = issues_parquet()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def load_dataset() -> pd.DataFrame:
    path = issues_parquet()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `make ingest` first")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def label_counts(frame: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for labels in frame["labels"]:
        counts.update(labels)
    return counts


def compute_stats(frame: pd.DataFrame, top: int = 20) -> dict[str, Any]:
    """Corpus summary. Every number here is counted from the frame, not estimated."""
    total = int(len(frame))
    counts = label_counts(frame)
    labelled = int(sum(1 for labels in frame["labels"] if len(labels) > 0))
    return {
        "total_issues": total,
        "labelled_issues": labelled,
        "pct_with_label": round(100.0 * labelled / total, 2) if total else 0.0,
        "distinct_labels": len(counts),
        "top_labels": counts.most_common(top),
    }


def print_stats(stats: dict[str, Any]) -> None:
    total = stats["total_issues"]
    print(f"total issues:      {total}")
    print(f"distinct labels:   {stats['distinct_labels']}")
    print(f"with >=1 label:    {stats['labelled_issues']} ({stats['pct_with_label']}%)")
    if not stats["top_labels"]:
        return
    width = max(len(name) for name, _ in stats["top_labels"])
    print(f"\ntop {len(stats['top_labels'])} labels:")
    for rank, (name, count) in enumerate(stats["top_labels"], start=1):
        share = 100.0 * count / total if total else 0.0
        print(f"  {rank:>2}. {name:<{width}}  {count:>5}  ({share:.1f}%)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_SUBCOMMANDS = {"fetch", "stats"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m triage_mcp.ingest",
        description="Harvest GitHub issues into data/issues.parquet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="harvest issues (default when omitted)")
    fetch.add_argument("--repo", default=DEFAULT_REPO, help=f"owner/name (default: {DEFAULT_REPO})")
    fetch.add_argument(
        "--max",
        dest="max_issues",
        type=int,
        default=DEFAULT_MAX_ISSUES,
        help=f"issue ceiling, PRs excluded (default: {DEFAULT_MAX_ISSUES})",
    )
    fetch.add_argument(
        "--fresh",
        action="store_true",
        help="delete already-harvested pages for this repo and start from page 1",
    )
    fetch.add_argument(
        "--via",
        choices=("list", "search"),
        default="list",
        help=(
            "list: the issues endpoint, fast but capped at "
            f"{PAGINATION_CEILING} items; search: date-window search, slower "
            "but reaches arbitrarily far back (default: list)"
        ),
    )
    fetch.add_argument(
        "--since",
        type=date.fromisoformat,
        help="earliest creation date for --via search (YYYY-MM-DD)",
    )
    fetch.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"search window width in days (default: {DEFAULT_WINDOW_DAYS})",
    )

    stats = sub.add_parser("stats", help="summarise data/issues.parquet")
    stats.add_argument("--top", type=int, default=20, help="labels to list (default: 20)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # `--repo X --max N` with no subcommand means fetch.
    if not args or (args[0] not in _SUBCOMMANDS and args[0] not in {"-h", "--help"}):
        args.insert(0, "fetch")

    parsed = build_parser().parse_args(args)

    if parsed.command == "stats":
        print_stats(compute_stats(load_dataset(), top=parsed.top))
        return 0

    load_dotenv()
    token = os.getenv("GITHUB_TOKEN") or None
    if not token:
        print(
            "[ingest] warning: no GITHUB_TOKEN -- unauthenticated requests are capped at "
            "60/hour and will not reach 3000 issues. See .env.example.",
            file=sys.stderr,
        )

    if parsed.via == "search":
        if parsed.since is None:
            parser_error = "--via search requires --since YYYY-MM-DD"
            raise SystemExit(f"[ingest] {parser_error}")
        harvest_by_search(
            parsed.repo,
            parsed.since,
            datetime.now(UTC).date(),
            token,
            window_days=parsed.window_days,
        )
    else:
        harvest(parsed.repo, parsed.max_issues, token, fresh=parsed.fresh)

    # In search mode the date range is the limiter, not --max, which exists to
    # bound the list harvest. Capping the dataset there too would silently drop
    # the oldest windows the user just paid to fetch.
    cap = None if parsed.via == "search" else parsed.max_issues
    frame = build_dataset(parsed.repo, max_issues=cap)
    print(f"[ingest] wrote {len(frame)} issues to {issues_parquet()}")
    print_stats(compute_stats(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
