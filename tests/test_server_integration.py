"""End-to-end MCP protocol tests: spawn the server over stdio and call its tools.

These drive the server as a real client does -- subprocess, stdio transport,
JSON-RPC -- so they cover the wiring that in-process calls would skip: schema
generation, argument coercion, structured-output serialisation.

Every tool is exercised once inside a single event loop by the module-scoped
`transcript` fixture, and the tests below assert against what came back. Sharing
one subprocess matters: each start-up loads 70MB of vectors and the encoder.
(An async fixture would not work here -- anyio's pytest plugin gives each test
its own event loop, and the stdio streams cannot outlive the loop that made
them.)

The artefacts in `data/` are required; without them the module skips, so a
fresh clone still gets a green `make test`.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from triage_mcp.evals import classifier_path
from triage_mcp.paths import issues_parquet
from triage_mcp.store import embeddings_path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            issues_parquet().exists() and embeddings_path().exists() and classifier_path().exists()
        ),
        reason="needs data/issues.parquet, embeddings.npy, classifier.joblib "
        "(run: make ingest && make embed && make eval)",
    ),
]

NEW_ISSUE = (
    "Terminal freezes when the agent runs a long build command. The integrated "
    "terminal stops accepting input until I kill the process."
)


def _structured(result: CallToolResult) -> dict[str, Any]:
    """The typed payload, falling back to the JSON text block."""
    if getattr(result, "structuredContent", None):
        return dict(result.structuredContent)
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return dict(json.loads(block.text))
    raise AssertionError("tool returned no structured content")


async def _exercise_every_tool() -> dict[str, Any]:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "triage_mcp.server"],
        # Pinned offline: corpus and model are local, so a cache miss should fail
        # loudly rather than quietly reach out to Hugging Face mid-test.
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    async with stdio_client(server) as (read, write), ClientSession(read, write) as client:
        init = await client.initialize()
        listed = await client.list_tools()

        search = _structured(
            await client.call_tool("search_similar_issues", {"query": "terminal hangs", "k": 5})
        )
        ranked = _structured(
            await client.call_tool("search_similar_issues", {"query": "python debugger", "k": 8})
        )
        empty_query = await client.call_tool("search_similar_issues", {"query": "   "})

        duplicates = _structured(
            await client.call_tool("find_duplicates", {"text": NEW_ISSUE, "k": 6, "threshold": 0.6})
        )
        loose = _structured(
            await client.call_tool(
                "find_duplicates", {"text": NEW_ISSUE, "k": 10, "threshold": 0.3}
            )
        )
        strict = _structured(
            await client.call_tool(
                "find_duplicates", {"text": NEW_ISSUE, "k": 10, "threshold": 0.9}
            )
        )

        classification = _structured(
            await client.call_tool("classify_component", {"text": NEW_ISSUE})
        )

        known_number = search["hits"][0]["number"]
        issue = _structured(await client.call_tool("get_issue", {"number": known_number}))
        missing = _structured(await client.call_tool("get_issue", {"number": -1}))

        report = _structured(await client.call_tool("triage", {"text": NEW_ISSUE}))
        info = _structured(await client.call_tool("corpus_info", {}))

        return {
            "server_name": init.serverInfo.name,
            "tools": {tool.name: tool for tool in listed.tools},
            "search": search,
            "ranked": ranked,
            "empty_query_is_error": bool(empty_query.isError),
            "duplicates": duplicates,
            "loose": loose,
            "strict": strict,
            "classification": classification,
            "known_number": known_number,
            "issue": issue,
            "missing": missing,
            "triage": report,
            "corpus_info": info,
        }


@pytest.fixture(scope="module")
def transcript() -> dict[str, Any]:
    """Spawn the server once and capture every tool's response."""
    return anyio.run(_exercise_every_tool)


# --------------------------------------------------------------------------- #
# Protocol surface
# --------------------------------------------------------------------------- #


def test_server_advertises_every_tool(transcript: dict[str, Any]) -> None:
    assert set(transcript["tools"]) == {
        "search_similar_issues",
        "find_duplicates",
        "classify_component",
        "get_issue",
        "triage",
        "corpus_info",
    }


def test_tools_declare_output_schemas(transcript: dict[str, Any]) -> None:
    """Typed results: the client sees a schema, not an opaque blob."""
    for name, tool in transcript["tools"].items():
        assert tool.outputSchema, f"{name} has no output schema"


def test_tools_are_documented_for_the_client_model(transcript: dict[str, Any]) -> None:
    for name, tool in transcript["tools"].items():
        assert tool.description and len(tool.description) > 40, f"{name} is underdocumented"


# --------------------------------------------------------------------------- #
# search_similar_issues
# --------------------------------------------------------------------------- #


def test_search_returns_scored_and_cited_hits(transcript: dict[str, Any]) -> None:
    payload = transcript["search"]

    assert len(payload["hits"]) == 5
    assert payload["cited_issue_numbers"] == [hit["number"] for hit in payload["hits"]]
    for hit in payload["hits"]:
        assert 0.0 <= hit["similarity"] <= 1.0
        assert hit["number"] > 0
        assert hit["url"].startswith("https://github.com/")


def test_search_results_are_ranked_by_similarity(transcript: dict[str, Any]) -> None:
    scores = [hit["similarity"] for hit in transcript["ranked"]["hits"]]

    assert scores == sorted(scores, reverse=True)


def test_search_rejects_an_empty_query(transcript: dict[str, Any]) -> None:
    assert transcript["empty_query_is_error"]


# --------------------------------------------------------------------------- #
# find_duplicates
# --------------------------------------------------------------------------- #


def test_duplicate_flags_follow_the_threshold(transcript: dict[str, Any]) -> None:
    payload = transcript["duplicates"]

    assert payload["threshold"] == 0.6
    for candidate in payload["candidates"]:
        assert candidate["above_threshold"] == (candidate["similarity"] >= 0.6)
    assert payload["n_above_threshold"] == sum(
        1 for candidate in payload["candidates"] if candidate["above_threshold"]
    )


def test_duplicate_threshold_is_disclosed_as_uncalibrated(transcript: dict[str, Any]) -> None:
    """The tool must not let a client read `above_threshold` as ground truth."""
    assert "HEURISTIC" in transcript["duplicates"]["caveat"]


def test_a_stricter_threshold_never_flags_more(transcript: dict[str, Any]) -> None:
    assert transcript["strict"]["n_above_threshold"] <= transcript["loose"]["n_above_threshold"]


# --------------------------------------------------------------------------- #
# classify_component
# --------------------------------------------------------------------------- #


def test_classify_reports_both_methods(transcript: dict[str, Any]) -> None:
    payload = transcript["classification"]

    assert payload["logreg"]["predicted_class"] in payload["known_classes"]
    assert payload["knn"]["predicted_class"] in payload["known_classes"]
    assert 0.0 <= payload["logreg"]["confidence"] <= 1.0
    assert 0.0 <= payload["knn"]["confidence"] <= 1.0


def test_agreement_flag_matches_the_two_predictions(transcript: dict[str, Any]) -> None:
    payload = transcript["classification"]

    assert payload["methods_agree"] == (
        payload["logreg"]["predicted_class"] == payload["knn"]["predicted_class"]
    )


def test_knn_exemplars_are_the_evidence_behind_the_vote(transcript: dict[str, Any]) -> None:
    """A vote with no citable neighbours is not a grounded answer."""
    payload = transcript["classification"]

    assert payload["exemplars"]
    assert payload["cited_issue_numbers"] == [ref["number"] for ref in payload["exemplars"]]
    assert payload["knn"]["confidence"] > 0


def test_classification_quotes_measured_scores_not_self_assessment(
    transcript: dict[str, Any],
) -> None:
    quality = transcript["classification"]["measured_quality"]

    assert 0.0 < quality["macro_f1"] < 1.0
    assert quality["source_run"], "the score must name the eval run it came from"


# --------------------------------------------------------------------------- #
# get_issue
# --------------------------------------------------------------------------- #


def test_get_issue_round_trips_a_known_number(transcript: dict[str, Any]) -> None:
    payload = transcript["issue"]

    assert payload["found"] is True
    assert payload["number"] == transcript["known_number"]
    assert payload["title"] == transcript["search"]["hits"][0]["title"]


def test_get_issue_reports_a_miss_without_erroring(transcript: dict[str, Any]) -> None:
    assert transcript["missing"]["found"] is False
    assert transcript["missing"]["number"] == -1


# --------------------------------------------------------------------------- #
# triage and corpus_info
# --------------------------------------------------------------------------- #


def test_triage_cites_the_union_of_both_halves(transcript: dict[str, Any]) -> None:
    payload = transcript["triage"]
    expected = sorted(
        {
            *payload["duplicates"]["cited_issue_numbers"],
            *payload["classification"]["cited_issue_numbers"],
        }
    )

    assert payload["cited_issue_numbers"] == expected


def test_triage_carries_its_caveats_forward(transcript: dict[str, Any]) -> None:
    assert transcript["triage"]["caveats"]


def test_corpus_info_states_coverage_and_provenance(transcript: dict[str, Any]) -> None:
    payload = transcript["corpus_info"]

    assert payload["corpus_size"] > 0
    assert payload["created_from"] < payload["created_to"]
    assert payload["embedding_model"]
    assert payload["classifier_source_run"]
