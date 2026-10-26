"""Spawn the MCP server over stdio and exercise every tool, printing one sample each.

This talks to the server the way a real client does -- subprocess, stdio
transport, JSON-RPC -- so it exercises the actual protocol surface rather than
calling the Python functions directly. If this passes, Claude Desktop and Claude
Code can drive the server too.

    uv run python scripts/smoke_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

NEW_ISSUE = (
    "Terminal freezes when the agent runs a long build command. "
    "The integrated terminal stops accepting input and the whole window becomes "
    "unresponsive until I kill the process."
)


def _structured(result: CallToolResult) -> dict[str, Any]:
    """Prefer the typed structuredContent; fall back to parsing the text block."""
    if getattr(result, "structuredContent", None):
        return dict(result.structuredContent)
    for block in result.content:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                return {"text": block.text}
    return {}


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _show(payload: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if key in payload:
            print(f"  {key}: {payload[key]}")


def _show_issues(refs: list[dict[str, Any]], limit: int = 3) -> None:
    for ref in refs[:limit]:
        flag = ""
        if "above_threshold" in ref:
            flag = "  [ABOVE]" if ref["above_threshold"] else "  [below]"
        print(f"    #{ref['number']:<7} sim={ref['similarity']:.4f}{flag}  {ref['title'][:58]}")
        if ref.get("labels"):
            print(f"             labels: {', '.join(ref['labels'][:6])}")


async def main() -> int:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "triage_mcp.server"],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )

    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        _rule("initialize")
        print(f"  server: {init.serverInfo.name} v{init.serverInfo.version}")

        tools = await session.list_tools()
        print(f"  tools:  {', '.join(tool.name for tool in tools.tools)}")

        _rule("corpus_info")
        info = _structured(await session.call_tool("corpus_info", {}))
        _show(
            info,
            [
                "corpus_size",
                "repos",
                "created_from",
                "created_to",
                "embedding_model",
                "embedding_backend",
                "classifier_classes",
                "classifier_source_run",
                "classifier_measured_macro_f1",
            ],
        )

        _rule("search_similar_issues(query='terminal hangs during build', k=5)")
        search = _structured(
            await session.call_tool(
                "search_similar_issues",
                {"query": "terminal hangs during build", "k": 5},
            )
        )
        print(f"  corpus_size: {search['corpus_size']}   returned: {search['k']}")
        _show_issues(search["hits"])
        print(f"  cited_issue_numbers: {search['cited_issue_numbers']}")

        _rule("find_duplicates(text=<new issue>, k=5, threshold=0.6)")
        duplicates = _structured(
            await session.call_tool(
                "find_duplicates", {"text": NEW_ISSUE, "k": 5, "threshold": 0.6}
            )
        )
        print(
            f"  threshold: {duplicates['threshold']}   "
            f"above: {duplicates['n_above_threshold']}/{len(duplicates['candidates'])}"
        )
        _show_issues(duplicates["candidates"])
        print(f"  caveat: {duplicates['caveat'][:100]}...")

        _rule("classify_component(text=<new issue>)")
        classification = _structured(
            await session.call_tool("classify_component", {"text": NEW_ISSUE})
        )
        logreg, knn = classification["logreg"], classification["knn"]
        print(
            f"  logreg: {logreg['predicted_class']!r} p={logreg['confidence']:.4f}"
            f"   (runner-up {logreg['runner_up']!r} p={logreg['runner_up_confidence']:.4f})"
        )
        print(f"  kNN:    {knn['predicted_class']!r} vote={knn['confidence']:.4f}")
        print(f"  methods_agree: {classification['methods_agree']}")
        print(f"  measured_quality: macro_f1={classification['measured_quality']['macro_f1']:.4f}")
        print("  exemplars behind the kNN vote:")
        _show_issues(classification["exemplars"])

        first_number = search["hits"][0]["number"]
        _rule(f"get_issue(number={first_number})")
        issue = _structured(await session.call_tool("get_issue", {"number": first_number}))
        _show(issue, ["found", "number", "title", "labels", "state", "state_reason", "created_at"])
        print(f"  url: {issue['url']}")

        _rule("get_issue(number=-1)  [absent issue]")
        missing = _structured(await session.call_tool("get_issue", {"number": -1}))
        print(f"  found: {missing['found']}   number: {missing['number']}")

        _rule("triage(text=<new issue>)")
        report = _structured(await session.call_tool("triage", {"text": NEW_ISSUE}))
        triage_logreg = report["classification"]["logreg"]["predicted_class"]
        triage_knn = report["classification"]["knn"]["predicted_class"]
        print(f"  text_preview: {report['text_preview'][:70]}...")
        print(f"  duplicates above threshold: {report['duplicates']['n_above_threshold']}")
        print(f"  predicted component (logreg): {triage_logreg!r}")
        print(f"  predicted component (kNN):    {triage_knn!r}")
        print(f"  cited_issue_numbers: {report['cited_issue_numbers']}")
        print(f"  caveats: {len(report['caveats'])} attached")

        _rule("all tools exercised OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
