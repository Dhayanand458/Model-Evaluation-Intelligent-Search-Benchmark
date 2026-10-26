# triage-mcp

## Purpose

An issue-triage MCP server. Given a repository's historical GitHub issues, it
helps answer triage questions — "has this been reported before?", "what labels
does this belong under?", "who else hit this?" — by exposing retrieval-backed
tools over MCP.

The pipeline is four stages, in order:

1. **ingest** — harvest historical issues from the GitHub REST API into a
   validated local dataset (`triage_mcp/ingest.py`).
2. **embed** — encode issues into vectors and persist a searchable index
   (`triage_mcp/store.py`).
3. **evaluate** — measure retrieval quality against a labelled set and write the
   numbers to disk (`triage_mcp/evals.py`).
4. **serve** — expose the triage tools to MCP clients (`triage_mcp/server.py`).

Local-first by design: no Docker, no database server. State lives in `data/`
(Parquet + JSONL) and `results/` (metrics), both git-ignored.

## Conventions

**Type hints everywhere.** Every function signature is annotated, including
tests. `ruff` enforces this (`ANN` rules) — run `make lint`.

**Pydantic for structured data.** Anything crossing a boundary — API responses,
records written to disk, tool inputs and outputs — is defined as a pydantic
model and validated. Do not pass bare dicts between modules.

**Unit tests never touch the network.** Tests run against checked-in fixtures in
`tests/fixtures/`. `tests/conftest.py` installs an autouse fixture that blocks
socket connections and `httpx` sends, so a test that reaches for the network
fails loudly instead of silently depending on GitHub being up and a token being
present.

**Metrics are computed, never asserted.** All metrics come from `evals.py` and
are written to `results/<timestamp>/metrics.json`. Never hardcode, estimate,
round from memory, or carry a number forward from a previous run — not in code,
not in docs, not in a chat reply. If a number has not been produced by an eval
run, the honest answer is that it is not known yet. When quoting a metric,
quote it from the `metrics.json` that produced it.

**Retrieval-backed answers cite their issues.** Any tool response or written
claim grounded in the corpus must carry the issue numbers that support it (e.g.
`#12345`). An answer that cannot name its supporting issues is not a grounded
answer, and should say so rather than assert.

## Layout

```
src/triage_mcp/
  ingest.py    GitHub harvest -> data/raw/*.jsonl -> data/issues.parquet
  store.py     all-MiniLM-L6-v2 embeddings, content-hash cache, exact-cosine top_k
  evals.py     time-split classification + retrieval proxy -> results/<ts>/
  server.py    FastMCP stdio server; lazy-loads corpus/vectors/classifier
  paths.py     the one place filesystem layout is defined
tests/         offline, fixture-based (+ one marked `integration` suite)
scripts/       smoke_mcp.py -- drives the server over stdio via the SDK client
docs/          mcp-setup.md -- client registration
data/          git-ignored corpus, embeddings.npy, classifier.joblib
results/       git-ignored metric runs (metrics.json, class_map.json, summary.md)
assets/        latest plots (overwritten each eval; run dir keeps copies)
```

The server does **no LLM calls and no network calls**. Tools return retrieved
rows, scores and model outputs; the client's model reasons over them. Every
result carries `cited_issue_numbers` — a tool that returned a conclusion without
the issue numbers behind it would be asking the client to trust something it
cannot check. Two disclosures ride along in `caveat` fields and must not be
dropped when narrating: the duplicate threshold is an **uncalibrated heuristic**
(no labelled duplicate pairs exist in this corpus), and `other` means "none of
the known components", not "unknown".

`data/classifier.joblib` is written by `make eval` and holds the estimators
*exactly as fitted during evaluation* — training split only, never refit on the
full corpus. That is deliberate: refitting on everything would probably predict
better but would make the macro-F1 in `metrics.json` a claim about a model
nobody measured. The bundle carries its own provenance, and
`classify_component` reports those measured scores rather than self-assessing.

Evaluation design, briefly: the task is component classification over the top-K
component labels (status/type/provenance labels are stoplisted in
`evals.NON_COMPONENT_LABELS` — vscode has both `*duplicate` and bare
`duplicate`, both excluded). The split is strictly temporal; classes with fewer
than 10 holdout examples are demoted to `other` *before* scoring, identically
for every method, and the demotions are recorded in `class_map.json`. Retrieval
quality is reported as label-match precision@5, which is a clearly-labelled
proxy — there is no duplicate-pair ground truth, so never quote it as a
duplicate-detection rate.

Resume state lives in `data/raw/<owner>__<repo>/`: one `page-NNNN.jsonl` per
fetched API page plus an advisory `_checkpoint.json`. **The page files are the
source of truth for resume**, not the checkpoint — `next_page_to_fetch()` scans
for the first gap, so a missing or corrupt checkpoint costs nothing. Do not
invert that: a checkpoint-driven resume silently re-harvests from page 1 when
the JSON fails to parse.

## Commands

```
make setup    # uv sync
make ingest   # REPO=owner/name MAX=3000
make stats    # corpus summary from data/issues.parquet
make embed    # build data/embeddings.npy (content-hashed; re-runs are free)
make eval     # results/<timestamp>/ + data/classifier.joblib
make serve    # run the MCP server on stdio
make smoke    # spawn the server over stdio and exercise every tool
make test
make lint
```

Client registration for Claude Desktop and Claude Code: [docs/mcp-setup.md](docs/mcp-setup.md).

Deeper history than the list endpoint allows:

```
uv run python -m triage_mcp.ingest --via search --since 2025-07-20
```

`make` is not installed on this machine. Until it is (`winget install
ezwinports.make`), run the target bodies directly — e.g. `uv run pytest`,
`uv run ruff check .`. The Makefile is the canonical list of what each
target does.

## Two constraints that shaped the design

**GitHub's list endpoint stops at 10,000 items.** `/repos/{o}/{r}/issues`
returns 422 past that, which for vscode (~43% PRs) is only ~5.7k issues, about
two months. `ingest.is_pagination_limit()` treats it as clean exhaustion. To go
further back use `--via search`, which slices by `created:` date windows and can
walk arbitrarily far; windows holding more than 1000 issues are split
recursively rather than silently truncated. Both harvest modes write into the
same `data/raw/<repo>/` and are deduped by issue number at parse time, so they
compose.

**Two interchangeable embedding backends.** `store.embed_texts` prefers
sentence-transformers and this is what built the committed index
(`embeddings.meta.json` and `metrics.json` both record `backend:
sentence-transformers`). During development torch's DLLs were transiently
blocked by Windows Smart App Control (`OSError: [WinError 4551]`), so
`embed_texts` falls back automatically to **onnxruntime** — the same
all-MiniLM-L6-v2, the same mean-pool then L2-normalise pipeline, no torch. The
fallback path and its pooling math are covered by unit tests, and the backend
that actually produced a vector is recorded alongside the vectors, because which
runtime made an embedding is part of reproducing a number. If Smart App Control
blocks torch again, do not "fix" it by disabling the policy — that is a one-way
switch on Windows and the user's decision; the ONNX fallback exists precisely so
the pipeline keeps working without it.

## Notes for future work

- Issue **comments are deliberately not fetched**. The list endpoint gives a
  comment *count* only. If comment bodies are ever needed, that is a per-issue
  request and a separate rate-limit budget — decide deliberately before adding.
- The list-issues endpoint returns pull requests too. `is_pull_request()` in
  `ingest.py` drops them. Any new GitHub reader must apply the same filter.
