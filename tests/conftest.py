"""Shared fixtures.

Enforces the project convention that unit tests never touch the network: any
socket connection or httpx send raises instead of reaching GitHub.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test tries to reach the network.

    Integration tests opt out: asyncio's event loop builds its self-pipe with
    `socket.socketpair()`, which calls `connect`, so blocking it would break the
    stdio client rather than catch a real network call. Those tests instead pin
    the server subprocess offline with HF_HUB_OFFLINE=1.
    """
    if request.node.get_closest_marker("integration"):
        return

    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("unit tests must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(httpx.Client, "send", _blocked)


@pytest.fixture
def issues_page() -> list[dict[str, Any]]:
    """One page of the GitHub list-issues response.

    6 items: 2 pull requests and 4 issue items, one of which repeats #201 --
    so a correct parse yields 3 unique issues.
    """
    return json.loads((FIXTURES / "issues_page.json").read_text(encoding="utf-8"))
