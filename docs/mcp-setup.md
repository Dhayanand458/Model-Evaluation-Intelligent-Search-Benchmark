# Registering triage-mcp with an MCP client

The server speaks MCP over **stdio**: the client launches it as a subprocess and
talks JSON-RPC over stdin/stdout. It makes no network calls and no LLM calls —
tools return retrieved issues, similarity scores and model predictions, and the
client's model does the reasoning.

Verified against MCP Python SDK **1.28.1**, which exposes the high-level server
API as `FastMCP` (`mcp.server.fastmcp`). SDK 2.x renames this to `MCPServer`; if
you upgrade past 1.x, `server.py` needs updating — `mcp.server.MCPServer` does
not exist in 1.28.1.

## 1. Build the artefacts first

The server refuses to start serving without them, and they are git-ignored, so a
fresh clone must build them:

```bash
make setup                                    # uv sync
cp .env.example .env                          # add GITHUB_TOKEN
make ingest REPO=microsoft/vscode MAX=3000    # -> data/issues.parquet
make embed                                    # -> data/embeddings.npy
make eval                                     # -> data/classifier.joblib + results/<ts>/
```

`make eval` is what persists `data/classifier.joblib`. The classifier is saved
exactly as fitted during evaluation — on the training split only — so the scores
in `results/<ts>/metrics.json` describe the artefact the server actually loads.

Check it works before wiring up a client:

```bash
uv run python scripts/smoke_mcp.py
```

## 2. Claude Code

Claude Code stores MCP servers at three scopes: `local` (default, just you, this
project), `project` (checked in via `.mcp.json`, shared with the team), and
`user` (you, across all projects).

Everything after `--` is the command Claude Code runs, passed through untouched:

```bash
# From the project directory
claude mcp add --scope local triage-mcp -- uv run python -m triage_mcp.server
```

`uv run` resolves the project venv, so no absolute interpreter path is needed —
as long as the server is launched from the project directory. To make it work
from anywhere, pin the directory:

```bash
claude mcp add --scope user triage-mcp -- \
  uv run --directory "/absolute/path/to/triage-mcp" python -m triage_mcp.server
```

To share it with the team, use project scope and commit the generated
`.mcp.json` at the repo root:

```bash
claude mcp add --scope project triage-mcp -- \
  uv run --directory "${CLAUDE_PROJECT_DIR:-.}" python -m triage_mcp.server
```

```json
{
  "mcpServers": {
    "triage-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "${CLAUDE_PROJECT_DIR:-.}", "python", "-m", "triage_mcp.server"]
    }
  }
}
```

> In a `.mcp.json` entry, `${CLAUDE_PROJECT_DIR}` is expanded in the *server's*
> environment, so give it a default (`${CLAUDE_PROJECT_DIR:-.}`) or the
> substitution can come out empty.

Environment variables use `-e` / `--env`. Put at least one other flag between
`--env` and the server name — otherwise the CLI reads the name as another
`KEY=value` pair and rejects it:

```bash
claude mcp add --env HF_HUB_OFFLINE=1 --scope local triage-mcp -- \
  uv run python -m triage_mcp.server
```

Managing it:

```bash
claude mcp list              # all configured servers
claude mcp get triage-mcp    # one server's config
claude mcp remove triage-mcp
```

Inside a session, `/mcp` shows connection status and the tools each server
exposes. First start is slow (loading ~70MB of vectors plus the encoder); if it
times out, raise the startup budget with `MCP_TIMEOUT=30000 claude`.

## 3. Claude Desktop

Open **Settings → Developer → Edit Config**, which creates or opens:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Claude Desktop does not inherit your shell's `PATH`, so use absolute paths for
both the command and the project directory.

```json
{
  "mcpServers": {
    "triage-mcp": {
      "command": "/absolute/path/to/uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/triage-mcp",
        "python",
        "-m",
        "triage_mcp.server"
      ]
    }
  }
}
```

On Windows, escape the backslashes:

```json
{
  "mcpServers": {
    "triage-mcp": {
      "command": "C:\\Users\\you\\.local\\bin\\uv.exe",
      "args": [
        "run",
        "--directory",
        "C:\\Users\\you\\triage-mcp",
        "python",
        "-m",
        "triage_mcp.server"
      ]
    }
  }
}
```

Find the absolute path to `uv` with `which uv` (macOS/Linux) or
`(Get-Command uv).Source` (PowerShell).

Then **quit Claude Desktop completely and reopen it** — reloading the window is
not enough. The server appears under the connectors control at the bottom-left
of the message box: click it, hover **Connectors**, choose **Manage connectors**,
and select `triage-mcp` to see its tools.

## 4. The tools

| Tool | Returns |
| --- | --- |
| `corpus_info()` | Corpus size, date coverage, embedding model, classifier provenance and measured scores |
| `search_similar_issues(query, k=10)` | Ranked issues with number, title, labels, similarity, snippet |
| `find_duplicates(text, k=10, threshold=0.6)` | Candidates flagged above/below threshold, with evidence snippets |
| `classify_component(text)` | Predictions from logreg *and* kNN, plus the neighbour issues behind the vote |
| `get_issue(number)` | Stored metadata for one issue |
| `triage(text)` | Duplicates + classification + every cited issue number, in one object |

Two things to pass on rather than paper over when narrating results:

- **The duplicate threshold is uncalibrated.** This corpus has no labelled
  duplicate pairs, so `above_threshold` means only "cosine similarity exceeded
  the number you passed". Every response carries this in a `caveat` field.
- **`other` is not "unknown".** It is the majority class by a wide margin and
  means "none of the known components". `classify_component` reports the
  classifier's measured macro-F1 alongside each prediction — quote that instead
  of implying the prediction is dependable.

Call `corpus_info()` first when it matters: the corpus is a fixed local
snapshot, so "no duplicates found" means "none in the ingested window", not
"none exist".

## 5. Troubleshooting

**Server does not appear.** Run the launch command by hand from a terminal — it
should sit silently waiting for JSON-RPC on stdin. Anything printed to *stdout*
that is not a protocol frame corrupts the stream; this server logs to stderr
only, so a stray `print()` added later is the first thing to suspect.

**Claude Desktop logs**: `~/Library/Logs/Claude/mcp*.log` (macOS) or
`%APPDATA%\Claude\logs\mcp*.log` (Windows). `mcp-server-triage-mcp.log` holds
this server's stderr.

**`FileNotFoundError: ... classifier.joblib`** — run `make eval`.

**`embeddings ... do not match corpus`** — `data/` was rebuilt without
re-embedding. Run `make embed`; it is content-hashed, so it is a no-op when
already current.

**Slow first tool call.** Artefacts load lazily on first use, so the first call
pays for the corpus, the vectors and the encoder. Later calls reuse them.
