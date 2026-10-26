"""Backoff decisions, computed from response headers only -- no network."""

from __future__ import annotations

import time

import httpx

from triage_mcp.ingest import (
    MAX_SLEEP_SECONDS,
    github_headers,
    is_pagination_limit,
    retry_delay,
)


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status_code=status, headers=headers or {})


def test_success_needs_no_retry() -> None:
    assert retry_delay(_response(200), attempt=0) is None


def test_retry_after_header_wins() -> None:
    delay = retry_delay(_response(403, {"retry-after": "45"}), attempt=0)

    assert delay == 45.0


def test_exhausted_primary_limit_waits_for_reset() -> None:
    reset = time.time() + 120
    delay = retry_delay(
        _response(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(reset))}),
        attempt=0,
    )

    assert delay is not None
    assert 100 < delay < 140


def test_waits_are_capped() -> None:
    """A multi-hour reset should stop the run, not block on it."""
    reset = time.time() + 7200
    delay = retry_delay(
        _response(429, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(reset))}),
        attempt=0,
    )

    assert delay == MAX_SLEEP_SECONDS


def test_server_errors_back_off_exponentially() -> None:
    delays = [retry_delay(_response(502), attempt=n) for n in range(4)]

    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_client_errors_are_final() -> None:
    """404 and 401 are not retryable -- surface them instead of looping."""
    assert retry_delay(_response(404), attempt=0) is None
    assert retry_delay(_response(401), attempt=0) is None


def test_malformed_headers_do_not_crash() -> None:
    assert retry_delay(_response(403, {"retry-after": "soon"}), attempt=0) is None


def test_pagination_ceiling_is_recognised() -> None:
    """422 past the ceiling ends the harvest cleanly instead of crashing it."""
    body = (
        b'{"message":"In order to keep the API fast for everyone, pagination is limited '
        b'for this resource. Check the link header for the number of pages available."}'
    )
    response = httpx.Response(status_code=422, content=body)

    assert is_pagination_limit(response) is True


def test_other_422s_are_not_treated_as_the_ceiling() -> None:
    """A genuine validation error must still surface."""
    response = httpx.Response(status_code=422, content=b'{"message":"Validation Failed"}')

    assert is_pagination_limit(response) is False
    assert retry_delay(response, attempt=0) is None  # final -> raises upstream


def test_non_422_is_never_the_ceiling() -> None:
    assert is_pagination_limit(_response(403)) is False


def test_token_is_sent_as_bearer() -> None:
    assert github_headers("ghp_x")["Authorization"] == "Bearer ghp_x"
    assert "Authorization" not in github_headers(None)
