"""MCP server exposing the triage corpus as grounded, structured tools.

Runs over stdio for local clients (Claude Desktop, Claude Code). Built on
``FastMCP`` from the official MCP Python SDK -- the high-level server API in
SDK 1.x. (SDK 2.x renames this to ``MCPServer``; this project pins 1.28.1, where
``mcp.server.MCPServer`` does not exist.)

**No LLM calls happen here.** Every tool returns retrieved rows, similarity
scores and model outputs. The reasoning -- "is this actually a duplicate?",
"how should this be routed?" -- belongs to the client model, which is why every
result carries the issue numbers and scores it was derived from. A tool that
returned a conclusion without its supporting issue numbers would be asking the
client to trust an assertion it cannot check.

Artefacts (corpus, embeddings, classifier) load lazily on first use and are then
cached for the process, so start-up is instant and a client that never calls a
tool never pays to load 70MB of vectors.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from triage_mcp import evals, store
from triage_mcp.ingest import load_dataset

SNIPPET_CHARS = 280
DEFAULT_K = 10
DEFAULT_DUPLICATE_THRESHOLD = 0.6
MAX_K = 50

THRESHOLD_CAVEAT = (
    "The similarity threshold is a HEURISTIC, not a calibrated decision boundary. "
    "This corpus has no labelled duplicate pairs, so no threshold has been validated "
    "against ground truth. `above_threshold` means 'cosine similarity exceeded the "
    "value you passed', nothing more. Judge each candidate from its title and snippet."
)

KNN_SCOPE_NOTE = (
    "kNN neighbours are drawn from the whole corpus, whereas the evaluation "
    "restricted them to the training split to keep the holdout unseen. Serving has "
    "no such constraint (the query text is new), so retrieval quality here should be "
    "no worse than the measured figures."
)


# --------------------------------------------------------------------------- #
# Result models -- these define the tools' structured output schemas
# --------------------------------------------------------------------------- #


class IssueRef(BaseModel):
    """A retrieved issue with the score that surfaced it. The unit of citation."""

    number: int = Field(description="GitHub issue number, for citing this evidence")
    title: str
    labels: list[str]
    state: str
    created_at: str
    similarity: float = Field(description="Cosine similarity to the query, 0-1")
    snippet: str = Field(description="Opening of the issue body, for judging relevance")
    url: str


class SearchResult(BaseModel):
    query: str
    k: int
    corpus_size: int
    hits: list[IssueRef]
    cited_issue_numbers: list[int]


class DuplicateCandidate(IssueRef):
    above_threshold: bool = Field(description="similarity >= the threshold that was passed")


class DuplicateResult(BaseModel):
    threshold: float
    n_above_threshold: int
    candidates: list[DuplicateCandidate]
    cited_issue_numbers: list[int]
    caveat: str = THRESHOLD_CAVEAT


class MethodPrediction(BaseModel):
    method: str
    predicted_class: str
    confidence: float = Field(description="Probability (logreg) or vote fraction (kNN), 0-1")
    runner_up: str | None = None
    runner_up_confidence: float | None = None


class ClassificationResult(BaseModel):
    logreg: MethodPrediction
    knn: MethodPrediction
    methods_agree: bool
    exemplars: list[IssueRef] = Field(
        description="The kNN neighbours that produced the vote -- the evidence behind it"
    )
    cited_issue_numbers: list[int]
    known_classes: list[str]
    measured_quality: dict[str, Any] = Field(
        description="Scores from the eval run that produced this classifier, not self-reported"
    )
    caveat: str


class TriageReport(BaseModel):
    """Everything a client model needs to narrate a triage decision, with citations."""

    text_preview: str
    duplicates: DuplicateResult
    classification: ClassificationResult
    cited_issue_numbers: list[int]
    caveats: list[str]


class IssueRecord(BaseModel):
    found: bool
    number: int
    title: str | None = None
    body: str | None = None
    labels: list[str] | None = None
    state: str | None = None
    state_reason: str | None = None
    created_at: str | None = None
    closed_at: str | None = None
    comments: int | None = None
    repo: str | None = None
    url: str | None = None


class CorpusInfo(BaseModel):
    corpus_size: int
    repos: list[str]
    created_from: str
    created_to: str
    embedding_model: str
    embedding_backend: str
    classifier_classes: list[str]
    classifier_source_run: str
    classifier_measured_macro_f1: float
    classifier_measured_accuracy: float
    notes: list[str]


# --------------------------------------------------------------------------- #
# Engine: lazy-loaded artefacts + the retrieval/classification logic
# --------------------------------------------------------------------------- #


def _text(value: object) -> str:
    """Parquet hands back NaN for missing text, and NaN is truthy."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _snippet(body: object, limit: int = SNIPPET_CHARS) -> str:
    """Flatten an issue body to a single-line preview."""
    flat = " ".join(_text(body).split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


class TriageEngine:
    """Holds the corpus, vectors and classifier. Everything loads on first use."""

    def __init__(self) -> None:
        self._frame: pd.DataFrame | None = None
        self._matrix: np.ndarray | None = None
        self._classifier: tuple[Any, Any, evals.ClassifierBundle] | None = None

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = load_dataset()
        return self._frame

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            # Verifies row alignment against the corpus and refuses a stale index.
            self._matrix = store.load_embeddings(self.frame)
        return self._matrix

    @property
    def classifier(self) -> tuple[Any, Any, evals.ClassifierBundle]:
        if self._classifier is None:
            self._classifier = evals.load_classifier()
        return self._classifier

    def embed(self, text: str) -> np.ndarray:
        """Encode query text with the same backend that built the index.

        Mixing backends would put the query in a different embedding space than
        the corpus, making every similarity meaningless.
        """
        meta = store.load_meta()
        backend = meta.backend if meta and meta.backend != "unknown" else "auto"
        model = meta.model if meta else store.DEFAULT_MODEL
        vector, _ = store.embed_texts([text], model, show_progress=False, backend=backend)
        return vector

    def _issue_ref(self, row_index: int, similarity: float) -> IssueRef:
        row = self.frame.iloc[row_index]
        repo = _text(row["repo"])
        number = int(row["number"])
        return IssueRef(
            number=number,
            title=_text(row["title"]),
            labels=list(row["labels"]),
            state=_text(row["state"]),
            created_at=str(row["created_at"]),
            similarity=round(float(similarity), 4),
            snippet=_snippet(row["body"]),
            url=f"https://github.com/{repo}/issues/{number}",
        )

    def search(self, query: str, k: int) -> list[IssueRef]:
        if not query.strip():
            raise ValueError("query must not be empty")
        k = max(1, min(int(k), MAX_K))
        indices, scores = store.top_k(self.embed(query), self.matrix, k)
        return [
            self._issue_ref(int(index), float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
        ]

    def classify(self, text: str, neighbours: int = evals.KNN_NEIGHBOURS) -> ClassificationResult:
        if not text.strip():
            raise ValueError("text must not be empty")
        vectoriser, logreg, bundle = self.classifier

        probabilities = logreg.predict_proba(vectoriser.transform([text]))[0]
        order = np.argsort(-probabilities, kind="stable")
        classes = list(logreg.classes_)

        indices, scores = store.top_k(self.embed(text), self.matrix, neighbours)
        neighbour_rows = indices[0]
        neighbour_scores = scores[0]
        y_all = evals.assign_classes(self.frame, bundle.classes[:-1])
        neighbour_classes = y_all[neighbour_rows]
        voted = evals.majority_vote(neighbour_classes, neighbour_scores)
        vote_fraction = float(np.mean(neighbour_classes == voted))

        exemplars = [
            self._issue_ref(int(index), float(score))
            for index, score, klass in zip(
                neighbour_rows, neighbour_scores, neighbour_classes, strict=True
            )
            if klass == voted
        ]

        return ClassificationResult(
            logreg=MethodPrediction(
                method="TF-IDF + logistic regression",
                predicted_class=str(classes[order[0]]),
                confidence=round(float(probabilities[order[0]]), 4),
                runner_up=str(classes[order[1]]) if len(order) > 1 else None,
                runner_up_confidence=(
                    round(float(probabilities[order[1]]), 4) if len(order) > 1 else None
                ),
            ),
            knn=MethodPrediction(
                method=f"embedding kNN (k={neighbours})",
                predicted_class=voted,
                confidence=round(vote_fraction, 4),
            ),
            methods_agree=str(classes[order[0]]) == voted,
            exemplars=exemplars,
            cited_issue_numbers=[ref.number for ref in exemplars],
            known_classes=bundle.classes,
            measured_quality={
                "source_run": bundle.source_run,
                "macro_f1": bundle.measured_macro_f1,
                "accuracy": bundle.measured_accuracy,
                "trained_on_issues_up_to": str(bundle.train_last),
                "n_train": bundle.n_train,
                "note": bundle.note,
            },
            caveat=(
                f"{KNN_SCOPE_NOTE} Classes outside {bundle.classes} are reported as "
                f"'{bundle.other_class}', which covers most of the corpus -- a prediction "
                f"of '{bundle.other_class}' means 'none of the known components', not 'unknown'."
            ),
        )


engine = TriageEngine()

mcp = FastMCP(
    "triage-mcp",
    instructions=(
        "Grounded GitHub issue triage over a local corpus. Tools return retrieved "
        "issues with similarity scores and model predictions; you do the reasoning. "
        "Always cite the issue numbers a tool returned when reporting its findings, "
        "and pass on the caveat fields rather than stating conclusions more "
        "confidently than the data supports."
    ),
)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@mcp.tool()
def search_similar_issues(query: str, k: int = DEFAULT_K) -> SearchResult:
    """Find issues semantically similar to a free-text query.

    Ranked by exact cosine similarity over sentence-embeddings of title + body.
    Use this to answer "has anything like this been reported?" -- then cite the
    returned issue numbers.
    """
    hits = engine.search(query, k)
    return SearchResult(
        query=query,
        k=len(hits),
        corpus_size=int(len(engine.frame)),
        hits=hits,
        cited_issue_numbers=[hit.number for hit in hits],
    )


@mcp.tool()
def find_duplicates(
    text: str,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> DuplicateResult:
    """Retrieve likely-duplicate candidates for a new issue's text.

    Returns the k nearest issues, each flagged for whether it cleared the
    threshold. The threshold is an uncalibrated heuristic -- see the `caveat`
    field -- so treat flagged items as candidates to read, not as confirmed
    duplicates, and cite issue numbers when reporting them.
    """
    hits = engine.search(text, k)
    candidates = [
        DuplicateCandidate(**hit.model_dump(), above_threshold=hit.similarity >= threshold)
        for hit in hits
    ]
    return DuplicateResult(
        threshold=threshold,
        n_above_threshold=sum(1 for candidate in candidates if candidate.above_threshold),
        candidates=candidates,
        cited_issue_numbers=[candidate.number for candidate in candidates],
    )


@mcp.tool()
def classify_component(text: str) -> ClassificationResult:
    """Predict which component/area an issue belongs to, by two methods.

    Reports the logistic-regression prediction and the embedding-kNN vote
    separately, along with the neighbour issues behind the vote, so you can see
    whether the methods agree and what evidence supports them. `measured_quality`
    carries the scores from the evaluation run that produced this classifier --
    quote those rather than assuming the prediction is reliable.
    """
    return engine.classify(text)


@mcp.tool()
def get_issue(number: int) -> IssueRecord:
    """Look up one issue by number and return its stored metadata."""
    frame = engine.frame
    match = frame[frame["number"] == int(number)]
    if match.empty:
        return IssueRecord(found=False, number=int(number))
    row = match.iloc[0]
    repo = _text(row["repo"])
    return IssueRecord(
        found=True,
        number=int(row["number"]),
        title=_text(row["title"]),
        body=_text(row["body"]) or None,
        labels=list(row["labels"]),
        state=_text(row["state"]),
        state_reason=_text(row["state_reason"]) or None,
        created_at=str(row["created_at"]),
        closed_at=None if pd.isna(row["closed_at"]) else str(row["closed_at"]),
        comments=int(row["comments"]),
        repo=repo,
        url=f"https://github.com/{repo}/issues/{int(row['number'])}",
    )


@mcp.tool()
def triage(text: str) -> TriageReport:
    """Full triage pass over new issue text: duplicates plus component routing.

    One call combining `find_duplicates` and `classify_component`, with every
    supporting issue number collected into `cited_issue_numbers`. Narrate this
    for the user -- including the caveats -- rather than presenting it as a
    decision already made.
    """
    duplicates = find_duplicates(text)
    classification = engine.classify(text)
    cited = sorted({*duplicates.cited_issue_numbers, *classification.cited_issue_numbers})
    return TriageReport(
        text_preview=_snippet(text, 200),
        duplicates=duplicates,
        classification=classification,
        cited_issue_numbers=cited,
        caveats=[THRESHOLD_CAVEAT, classification.caveat],
    )


@mcp.tool()
def corpus_info() -> CorpusInfo:
    """Describe what this server is serving: corpus coverage, model, and measured quality.

    Call this before relying on the other tools -- it says what date range the
    corpus covers, so you can tell whether a "no duplicates found" answer means
    "none exist" or "none within the ingested window".
    """
    frame = engine.frame
    meta = store.load_meta()
    _, _, bundle = engine.classifier
    return CorpusInfo(
        corpus_size=int(len(frame)),
        repos=sorted(frame["repo"].unique().tolist()),
        created_from=str(frame["created_at"].min()),
        created_to=str(frame["created_at"].max()),
        embedding_model=meta.model if meta else store.DEFAULT_MODEL,
        embedding_backend=meta.backend if meta else "unknown",
        classifier_classes=bundle.classes,
        classifier_source_run=bundle.source_run,
        classifier_measured_macro_f1=bundle.measured_macro_f1,
        classifier_measured_accuracy=bundle.measured_accuracy,
        notes=[
            "Corpus is a fixed local snapshot; issues created after `created_to` are absent.",
            THRESHOLD_CAVEAT,
            KNN_SCOPE_NOTE,
        ],
    )


def main() -> int:
    # stdio is FastMCP's default transport; naming it makes the contract explicit,
    # since anything written to stdout other than protocol frames breaks the client.
    print("[server] triage-mcp starting on stdio", file=sys.stderr)
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
