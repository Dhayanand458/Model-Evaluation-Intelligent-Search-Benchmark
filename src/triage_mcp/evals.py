"""Retrieval and classification evaluation. The only source of project metrics.

Every metric this project reports is produced by a run of this module and
written to ``results/<timestamp>/metrics.json``. Metrics are never hardcoded,
never estimated, and never carried over from a previous run.

The evaluation is a component-classification task built from the issue labels,
scored on a strict time-based holdout, plus a clearly-labelled *proxy* for
duplicate-retrieval quality. Read :func:`build_class_map` before trusting any
number: the class set is a judgment call, and it is persisted alongside the
metrics so it can be audited.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from triage_mcp import store
from triage_mcp.ingest import load_dataset
from triage_mcp.paths import assets_dir, data_dir, results_dir

SEED = 20260720
HOLDOUT_FRACTION = 0.2
DEFAULT_K_CLASSES = 10
DEFAULT_MIN_HOLDOUT_SUPPORT = 10
KNN_NEIGHBOURS = 10
RETRIEVAL_K = 5
OTHER = "other"

PROXY_DISCLAIMER = (
    "label-match precision@5 is a PROXY for duplicate/similar-issue retrieval. "
    "It measures how often a retrieved neighbour shares the query issue's "
    "component class, NOT whether it is actually a duplicate. There is no "
    "duplicate-pair ground truth in this corpus. Two issues in the same "
    "component are usually not duplicates, so treat this as a relative signal "
    "for comparing retrieval methods, never as an absolute duplicate-detection rate."
)

# --------------------------------------------------------------------------- #
# Which labels can be a class
#
# GitHub labels mix several orthogonal axes. Only component/area labels make a
# meaningful classification target, so the others are excluded by name. This
# stoplist is a judgment call that shapes every downstream number, which is why
# it is grouped by rationale and written verbatim into results/class_map.json.
# --------------------------------------------------------------------------- #

NON_COMPONENT_LABELS: dict[str, frozenset[str]] = {
    # Where an issue sits in the triage workflow -- says nothing about subject.
    "status": frozenset(
        {
            "info-needed",
            "insiders-released",
            "verified",
            "verification-needed",
            "verification-found",
            "verification-steps-needed",
            "new release",
            "triage-needed",
            "recent-regression",
            "regression",
            "stale",
            "spam",
            "candidate",
            "invalid",
            "under-discussion",
            "confirmed",
            "unreleased",
            "follow-up",
            "author-verification-requested",
            "important",
            "upstream",
            "on-testplan",
            "testplan-item",
            "endgame-plan",
            "smoke-test-failure",
            "stable-anomaly",
            "help wanted",
            "good first issue",
            "accessibility-sla",
            "error-telemetry",
            "z-author-verified",
            "needs more info",
            # vscode uses both "*duplicate" and a bare "duplicate"; the asterisk
            # rule only catches the first, and a resolution is never a component.
            "duplicate",
            "confirmation-pending",
            "on-release-notes",
        }
    ),
    # What kind of work it is -- orthogonal to which component it touches.
    "type": frozenset(
        {
            "bug",
            "feature-request",
            "debt",
            "engineering",
            "polish",
            "papercut :drop_of_blood:",
            "freeze-slow-crash-leak",
            "modernization",
            "perf",
            "security",
        }
    ),
    # How the issue was produced or processed, not what it is about.
    "provenance": frozenset({"ai-generated", "ai-analyzed", "ai-translated", "ai-triaged"}),
}

# vscode marks resolutions with a leading asterisk: *duplicate, *question,
# *caused-by-extension, *as-designed, ... All are status, none are components.
RESOLUTION_PREFIX = "*"

_STOPPED: frozenset[str] = frozenset().union(*NON_COMPONENT_LABELS.values())


def is_component_label(label: str) -> bool:
    """True when a label names a component/area rather than status, type or provenance."""
    return not label.startswith(RESOLUTION_PREFIX) and label not in _STOPPED


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class DemotedClass(BaseModel):
    label: str
    train_count: int
    holdout_count: int
    reason: str


class ClassMap(BaseModel):
    """The label -> class mapping, persisted so the task is auditable."""

    classes: list[str]
    other_class: str = OTHER
    k_requested: int
    min_holdout_support: int
    ranking_rule: str
    tie_break_rule: str
    demoted: list[DemotedClass]
    stoplist: dict[str, list[str]]
    resolution_prefix: str = RESOLUTION_PREFIX
    train_counts: dict[str, int]
    holdout_counts: dict[str, int]
    other_composition: dict[str, int]


class SplitInfo(BaseModel):
    """A strict time split: every training issue predates every holdout issue."""

    holdout_fraction: float
    n_total: int
    n_train: int
    n_holdout: int
    boundary: datetime
    train_first: datetime
    train_last: datetime
    holdout_first: datetime
    holdout_last: datetime


class Latency(BaseModel):
    n_queries: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


class MethodResult(BaseModel):
    method: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    latency: Latency | None = None
    notes: str = ""


class ProxyResult(BaseModel):
    method: str
    k: int
    metric: str = "label_match_precision_at_k_PROXY"
    precision_all: float
    n_queries_all: int
    precision_excluding_other: float
    n_queries_excluding_other: int
    latency: Latency
    disclaimer: str = PROXY_DISCLAIMER


# --------------------------------------------------------------------------- #
# Split and task construction
# --------------------------------------------------------------------------- #


def time_split(
    frame: pd.DataFrame, holdout_fraction: float = HOLDOUT_FRACTION
) -> tuple[np.ndarray, np.ndarray, SplitInfo]:
    """Most recent `holdout_fraction` of issues by creation date becomes holdout.

    The boundary is a timestamp, not a row index: issues sharing the boundary
    instant all land in holdout. That keeps the separation strictly temporal --
    no training issue was created at or after any holdout issue -- at the cost of
    the split being approximately rather than exactly the requested fraction.
    """
    created = frame["created_at"]
    n_holdout_target = max(1, int(round(len(frame) * holdout_fraction)))
    ordered = created.sort_values(kind="stable")
    boundary = ordered.iloc[len(frame) - n_holdout_target]

    holdout_mask = (created >= boundary).to_numpy()
    train_mask = ~holdout_mask
    if not train_mask.any():
        raise ValueError("time split left no training rows")

    info = SplitInfo(
        holdout_fraction=holdout_fraction,
        n_total=len(frame),
        n_train=int(train_mask.sum()),
        n_holdout=int(holdout_mask.sum()),
        boundary=boundary,
        train_first=created[train_mask].min(),
        train_last=created[train_mask].max(),
        holdout_first=created[holdout_mask].min(),
        holdout_last=created[holdout_mask].max(),
    )
    return train_mask, holdout_mask, info


def assign_classes(frame: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Map each issue to one class, or `other`.

    An issue carrying several class labels is resolved deterministically: the
    class earliest in `classes` wins, and `classes` is ordered by training
    frequency, so "most frequent class wins".
    """
    rank = {name: index for index, name in enumerate(classes)}
    assigned: list[str] = []
    for labels in frame["labels"]:
        best: str | None = None
        best_rank = len(classes)
        for label in labels:
            position = rank.get(label)
            if position is not None and position < best_rank:
                best, best_rank = label, position
        assigned.append(best if best is not None else OTHER)
    return np.array(assigned, dtype=object)


def build_class_map(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    k: int = DEFAULT_K_CLASSES,
    min_holdout_support: int = DEFAULT_MIN_HOLDOUT_SUPPORT,
) -> ClassMap:
    """Choose the class set, then drop classes the holdout cannot score.

    Candidates are ranked by frequency in the **training** period only, so the
    ranking never peeks at holdout content. The support guard then uses holdout
    *label counts* -- not features, not text -- to demote classes with too few
    holdout examples to `other`. A class with two holdout issues produces an F1
    that swings wildly on a single prediction and would corrupt the macro
    average; demoting it is the difference between a noisy headline and an
    honest one. The guard is applied once, identically for every method.
    """
    train_counts: Counter[str] = Counter()
    for labels in frame["labels"][train_mask]:
        train_counts.update(label for label in labels if is_component_label(label))

    candidates = [name for name, _ in train_counts.most_common(k)]
    provisional = assign_classes(frame, candidates)
    holdout_counts = Counter(provisional[holdout_mask])

    kept: list[str] = []
    demoted: list[DemotedClass] = []
    for name in candidates:
        support = int(holdout_counts.get(name, 0))
        if support >= min_holdout_support:
            kept.append(name)
        else:
            demoted.append(
                DemotedClass(
                    label=name,
                    train_count=int(train_counts[name]),
                    holdout_count=support,
                    reason=f"holdout support {support} < {min_holdout_support}",
                )
            )

    final = assign_classes(frame, kept)
    final_train = Counter(final[train_mask])
    final_holdout = Counter(final[holdout_mask])

    class_set = set(kept)
    unlabelled = sum(1 for labels in frame["labels"] if len(labels) == 0)
    labelled_no_class = sum(
        1
        for labels in frame["labels"]
        if len(labels) > 0 and not any(label in class_set for label in labels)
    )

    return ClassMap(
        classes=kept,
        k_requested=k,
        min_holdout_support=min_holdout_support,
        ranking_rule="component labels ranked by frequency in the training split only",
        tie_break_rule="multi-label issues take the class with the highest training frequency",
        demoted=demoted,
        stoplist={name: sorted(labels) for name, labels in NON_COMPONENT_LABELS.items()},
        train_counts={name: int(final_train.get(name, 0)) for name in [*kept, OTHER]},
        holdout_counts={name: int(final_holdout.get(name, 0)) for name in [*kept, OTHER]},
        other_composition={
            "unlabelled_entirely": unlabelled,
            "labelled_but_no_class_label": labelled_no_class,
        },
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def latency_stats(samples_ms: list[float]) -> Latency:
    array = np.asarray(samples_ms, dtype=float)
    return Latency(
        n_queries=int(array.size),
        mean_ms=float(array.mean()),
        p50_ms=float(np.percentile(array, 50)),
        p95_ms=float(np.percentile(array, 95)),
    )


def score(
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    latency: Latency | None = None,
    notes: str = "",
) -> MethodResult:
    """Accuracy, macro-F1 and a per-class breakdown, all computed here."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in labels
        if name in report
    }
    return MethodResult(
        method=method,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        per_class=per_class,
        latency=latency,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #


def majority_baseline(y_train: np.ndarray, n_holdout: int) -> np.ndarray:
    """Always predict the most common training class."""
    winner = Counter(y_train).most_common(1)[0][0]
    return np.array([winner] * n_holdout, dtype=object)


def tfidf_logreg(
    texts_train: list[str],
    y_train: np.ndarray,
    texts_holdout: list[str],
) -> tuple[np.ndarray, Latency, TfidfVectorizer, LogisticRegression]:
    """TF-IDF + multinomial logistic regression, fit on the training split only.

    The vectoriser is fit on training text alone -- fitting it on the full
    corpus would leak holdout vocabulary and IDF weights into the features.
    `class_weight="balanced"` is used because `other` dominates; without it the
    model collapses to predicting `other` and macro-F1 is uninformative.
    """
    vectoriser = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
    )
    x_train = vectoriser.fit_transform(texts_train)
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )
    classifier.fit(x_train, y_train)

    # Timed one query at a time: that is how the model would actually be served,
    # and a batched predict would understate per-request cost.
    predictions: list[str] = []
    samples: list[float] = []
    for text in texts_holdout:
        started = time.perf_counter()
        vector = vectoriser.transform([text])
        predictions.append(str(classifier.predict(vector)[0]))
        samples.append((time.perf_counter() - started) * 1000.0)

    return np.array(predictions, dtype=object), latency_stats(samples), vectoriser, classifier


def majority_vote(neighbour_classes: np.ndarray, scores: np.ndarray) -> str:
    """Plain majority vote; ties break on summed similarity, then class name."""
    counts = Counter(neighbour_classes.tolist())
    best = max(counts.values())
    tied = sorted(name for name, count in counts.items() if count == best)
    if len(tied) == 1:
        return str(tied[0])
    similarity = {name: float(scores[neighbour_classes == name].sum()) for name in tied}
    return str(max(tied, key=lambda name: similarity[name]))


def embedding_knn(
    matrix: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    y_all: np.ndarray,
    neighbours: int = KNN_NEIGHBOURS,
) -> tuple[np.ndarray, Latency]:
    """Cosine top-k neighbours from the training split, majority vote.

    `candidate_mask=train_mask` is the leakage guard: holdout rows are removed
    from the candidate pool before any similarity is computed, so a holdout
    issue can never retrieve itself or a future sibling.
    """
    holdout_indices = np.flatnonzero(holdout_mask)
    predictions: list[str] = []
    samples: list[float] = []
    for row in holdout_indices:
        started = time.perf_counter()
        indices, scores = store.top_k(matrix[row], matrix, neighbours, candidate_mask=train_mask)
        predictions.append(majority_vote(y_all[indices[0]], scores[0]))
        samples.append((time.perf_counter() - started) * 1000.0)
    return np.array(predictions, dtype=object), latency_stats(samples)


# --------------------------------------------------------------------------- #
# Retrieval proxy
# --------------------------------------------------------------------------- #


def _proxy_from_neighbours(
    neighbour_classes: list[np.ndarray], query_classes: np.ndarray
) -> tuple[float, float, int, int]:
    """Mean fraction of retrieved neighbours sharing the query's class.

    Reported twice: over every holdout issue, and over only those with a real
    component class. The first number is inflated because `other` is the
    majority class and `other`-to-`other` matches are mostly noise agreeing with
    noise; the second is the one that says something about retrieval.
    """
    all_scores: list[float] = []
    real_scores: list[float] = []
    for neighbours, query_class in zip(neighbour_classes, query_classes, strict=True):
        share = float(np.mean(neighbours == query_class)) if neighbours.size else 0.0
        all_scores.append(share)
        if query_class != OTHER:
            real_scores.append(share)
    return (
        float(np.mean(all_scores)) if all_scores else 0.0,
        float(np.mean(real_scores)) if real_scores else 0.0,
        len(all_scores),
        len(real_scores),
    )


def embedding_proxy(
    matrix: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    y_all: np.ndarray,
    k: int = RETRIEVAL_K,
) -> ProxyResult:
    holdout_indices = np.flatnonzero(holdout_mask)
    neighbours: list[np.ndarray] = []
    samples: list[float] = []
    for row in holdout_indices:
        started = time.perf_counter()
        indices, _ = store.top_k(matrix[row], matrix, k, candidate_mask=train_mask)
        samples.append((time.perf_counter() - started) * 1000.0)
        neighbours.append(y_all[indices[0]])

    precision_all, precision_real, n_all, n_real = _proxy_from_neighbours(
        neighbours, y_all[holdout_mask]
    )
    return ProxyResult(
        method="embeddings (all-MiniLM-L6-v2, exact cosine)",
        k=k,
        precision_all=precision_all,
        n_queries_all=n_all,
        precision_excluding_other=precision_real,
        n_queries_excluding_other=n_real,
        latency=latency_stats(samples),
    )


def tfidf_proxy(
    vectoriser: TfidfVectorizer,
    texts_train: list[str],
    texts_holdout: list[str],
    y_train: np.ndarray,
    y_holdout: np.ndarray,
    k: int = RETRIEVAL_K,
) -> ProxyResult:
    """Same proxy over TF-IDF cosine, as the bag-of-words comparison point."""
    x_train = vectoriser.transform(texts_train)  # already fit on train only
    neighbours: list[np.ndarray] = []
    samples: list[float] = []
    for text in texts_holdout:
        started = time.perf_counter()
        query = vectoriser.transform([text])
        similarities = (query @ x_train.T).toarray().ravel()
        k_eff = min(k, similarities.size)
        top = np.argpartition(-similarities, k_eff - 1)[:k_eff]
        top = top[np.argsort(-similarities[top], kind="stable")]
        samples.append((time.perf_counter() - started) * 1000.0)
        neighbours.append(y_train[top])

    precision_all, precision_real, n_all, n_real = _proxy_from_neighbours(neighbours, y_holdout)
    return ProxyResult(
        method="TF-IDF cosine",
        k=k,
        precision_all=precision_all,
        n_queries_all=n_all,
        precision_excluding_other=precision_real,
        n_queries_excluding_other=n_real,
        latency=latency_stats(samples),
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_macro_f1(results: list[MethodResult], destinations: list[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [result.method for result in results]
    values = [result.macro_f1 for result in results]

    figure, axes = plt.subplots(figsize=(8, 4.5))
    bars = axes.bar(names, values, color=["#9aa0a6", "#4c8bf5", "#34a853"][: len(names)])
    axes.set_ylabel("macro-F1")
    axes.set_title("Component classification: macro-F1 by method")
    axes.set_ylim(0, max(values + [0.1]) * 1.25)
    for bar, value in zip(bars, values, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    axes.spines[["top", "right"]].set_visible(False)
    plt.setp(axes.get_xticklabels(), rotation=12, ha="right")
    figure.tight_layout()
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    method: str,
    destinations: list[Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    # Row-normalised: raw counts would be unreadable when `other` outnumbers
    # every real class by an order of magnitude.
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)

    figure, axes = plt.subplots(figsize=(7.5, 6.5))
    image = axes.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes.set_yticks(range(len(labels)), labels)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title(f"Confusion matrix (row-normalised)\n{method}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes.text(
                j,
                i,
                f"{normalised[i, j]:.2f}\n({matrix[i, j]})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalised[i, j] > 0.5 else "black",
            )
    figure.colorbar(image, ax=axes, fraction=0.046, label="fraction of true class")
    figure.tight_layout()
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


class ClassifierBundle(BaseModel):
    """Provenance for the served classifier. Travels with the pickled estimators.

    The server reports these numbers verbatim rather than describing its own
    accuracy, so a client model never has to guess how good the prediction is.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    classes: list[str]
    other_class: str
    fitted_at: datetime
    seed: int
    n_train: int
    train_last: datetime
    holdout_first: datetime
    measured_accuracy: float
    measured_macro_f1: float
    source_run: str
    note: str


def classifier_path() -> Path:
    return data_dir() / "classifier.joblib"


def persist_classifier(
    vectoriser: TfidfVectorizer,
    classifier: LogisticRegression,
    bundle: ClassifierBundle,
    run_dir: Path,
) -> Path:
    """Write the fitted estimators plus provenance for the server to load.

    Persisted as fit during evaluation -- on the training split only, never
    refit on the full corpus. That keeps the server's behaviour identical to the
    configuration whose macro-F1 is in metrics.json; refitting on everything
    would probably predict better but would make the reported number a claim
    about a model nobody measured.
    """
    payload = {
        "vectoriser": vectoriser,
        "classifier": classifier,
        "bundle": bundle.model_dump(mode="json"),
    }
    joblib.dump(payload, classifier_path())
    joblib.dump(payload, run_dir / "classifier.joblib")
    return classifier_path()


def load_classifier() -> tuple[TfidfVectorizer, LogisticRegression, ClassifierBundle]:
    """Load the persisted classifier, or explain how to produce it."""
    path = classifier_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `make eval` to fit and persist it")
    payload = joblib.load(path)
    return (
        payload["vectoriser"],
        payload["classifier"],
        ClassifierBundle.model_validate(payload["bundle"]),
    )


def new_run_dir(now: datetime | None = None) -> Path:
    """Create and return ``results/<utc-timestamp>/`` for this eval run."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir() / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    """Write computed metrics to ``<run_dir>/metrics.json``.

    Callers pass numbers they just computed. Anything else is a bug.
    """
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def summary_markdown(
    results: list[MethodResult],
    proxies: list[ProxyResult],
    class_map: ClassMap,
    split: SplitInfo,
    best: MethodResult,
) -> str:
    lines: list[str] = []
    lines.append("# triage-mcp evaluation")
    lines.append("")
    lines.append(
        f"Corpus {split.n_total} issues · train {split.n_train} "
        f"({split.train_first:%Y-%m-%d} → {split.train_last:%Y-%m-%d}) · "
        f"holdout {split.n_holdout} ({split.holdout_first:%Y-%m-%d} → "
        f"{split.holdout_last:%Y-%m-%d})"
    )
    lines.append("")
    lines.append("## Component classification")
    lines.append("")
    lines.append("| method | accuracy | macro-F1 | weighted-F1 | p50 latency | p95 latency |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for result in results:
        p50 = f"{result.latency.p50_ms:.2f} ms" if result.latency else "—"
        p95 = f"{result.latency.p95_ms:.2f} ms" if result.latency else "—"
        lines.append(
            f"| {result.method} | {result.accuracy:.4f} | **{result.macro_f1:.4f}** | "
            f"{result.weighted_f1:.4f} | {p50} | {p95} |"
        )
    lines.append("")
    lines.append(f"Best by macro-F1: **{best.method}**")
    lines.append("")
    lines.append(f"### Per-class ({best.method})")
    lines.append("")
    lines.append("| class | precision | recall | F1 | holdout support |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, scores in best.per_class.items():
        lines.append(
            f"| `{name}` | {scores['precision']:.3f} | {scores['recall']:.3f} | "
            f"{scores['f1']:.3f} | {int(scores['support'])} |"
        )
    lines.append("")
    lines.append("## Retrieval proxy — label-match precision@5")
    lines.append("")
    lines.append(f"> **PROXY METRIC.** {PROXY_DISCLAIMER}")
    lines.append("")
    lines.append("| method | P@5 (all holdout) | P@5 (excl. `other`) | p50 latency | p95 latency |")
    lines.append("|---|---:|---:|---:|---:|")
    for proxy in proxies:
        lines.append(
            f"| {proxy.method} | {proxy.precision_all:.4f} | "
            f"{proxy.precision_excluding_other:.4f} | {proxy.latency.p50_ms:.2f} ms | "
            f"{proxy.latency.p95_ms:.2f} ms |"
        )
    lines.append("")
    lines.append("## Task construction")
    lines.append("")
    lines.append(
        f"Classes ({len(class_map.classes)} of {class_map.k_requested} requested): "
        + ", ".join(f"`{name}`" for name in class_map.classes)
        + f", `{class_map.other_class}`"
    )
    lines.append("")
    if class_map.demoted:
        lines.append(
            f"Demoted to `{class_map.other_class}` for insufficient holdout support "
            f"(< {class_map.min_holdout_support}):"
        )
        lines.append("")
        lines.append("| label | train | holdout |")
        lines.append("|---|---:|---:|")
        for entry in class_map.demoted:
            lines.append(f"| `{entry.label}` | {entry.train_count} | {entry.holdout_count} |")
        lines.append("")
    other_total = sum(class_map.other_composition.values())
    lines.append(
        f"`other` composition: {class_map.other_composition['unlabelled_entirely']} unlabelled, "
        f"{class_map.other_composition['labelled_but_no_class_label']} labelled but with no class "
        f"label ({other_total} rows). Accuracy is inflated by this bucket; macro-F1 is the "
        "number to compare methods on."
    )
    lines.append("")
    lines.append("Full mapping and stoplist: `class_map.json`. All numbers: `metrics.json`.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m triage_mcp.evals",
        description="Evaluate component classification and retrieval on the issue corpus.",
    )
    parser.add_argument("--k-classes", type=int, default=DEFAULT_K_CLASSES)
    parser.add_argument("--min-holdout-support", type=int, default=DEFAULT_MIN_HOLDOUT_SUPPORT)
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--neighbours", type=int, default=KNN_NEIGHBOURS)
    parser.add_argument("--retrieval-k", type=int, default=RETRIEVAL_K)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)

    frame = load_dataset()
    store.build_index()  # content-hashed: a no-op when the index is current
    matrix = store.load_embeddings(frame)

    train_mask, holdout_mask, split = time_split(frame, args.holdout_fraction)
    class_map = build_class_map(
        frame, train_mask, holdout_mask, args.k_classes, args.min_holdout_support
    )
    if not class_map.classes:
        raise SystemExit(
            "[evals] no class survived the holdout-support guard; the corpus cannot "
            "support this task. Widen the corpus or lower --min-holdout-support."
        )

    y_all = assign_classes(frame, class_map.classes)
    y_train, y_holdout = y_all[train_mask], y_all[holdout_mask]
    labels = [*class_map.classes, OTHER]

    texts = store.corpus_texts(frame)
    texts_train = [texts[i] for i in np.flatnonzero(train_mask)]
    texts_holdout = [texts[i] for i in np.flatnonzero(holdout_mask)]

    # flush=True throughout: with stdout redirected to a file these lines sit in
    # the block buffer for the whole run, and anyone tailing the log reads a
    # 20-minute-old picture of which stage is live.
    print(
        f"[evals] {split.n_total} issues, {split.n_train} train / {split.n_holdout} holdout, "
        f"{len(class_map.classes)} classes + other",
        flush=True,
    )

    results: list[MethodResult] = []

    print("[evals] majority baseline", flush=True)
    results.append(
        score(
            "majority baseline",
            y_holdout,
            majority_baseline(y_train, len(y_holdout)),
            labels,
            notes="predicts the most frequent training class for every issue",
        )
    )

    print("[evals] TF-IDF + logistic regression", flush=True)
    tfidf_predictions, tfidf_latency, vectoriser, logreg = tfidf_logreg(
        texts_train, y_train, texts_holdout
    )
    results.append(
        score(
            "TF-IDF + logistic regression",
            y_holdout,
            tfidf_predictions,
            labels,
            tfidf_latency,
            notes="vectoriser fit on training split only; class_weight=balanced",
        )
    )

    print(f"[evals] embedding kNN (k={args.neighbours})", flush=True)
    knn_predictions, knn_latency = embedding_knn(
        matrix, train_mask, holdout_mask, y_all, args.neighbours
    )
    results.append(
        score(
            f"embedding kNN (k={args.neighbours})",
            y_holdout,
            knn_predictions,
            labels,
            knn_latency,
            notes="cosine neighbours restricted to the training split via candidate mask",
        )
    )

    best = max(results, key=lambda result: result.macro_f1)
    best_predictions = {
        "majority baseline": None,
        "TF-IDF + logistic regression": tfidf_predictions,
        f"embedding kNN (k={args.neighbours})": knn_predictions,
    }.get(best.method)
    if best_predictions is None:
        best_predictions = majority_baseline(y_train, len(y_holdout))

    print(f"[evals] retrieval proxy @{args.retrieval_k}", flush=True)
    proxies = [
        embedding_proxy(matrix, train_mask, holdout_mask, y_all, args.retrieval_k),
        tfidf_proxy(vectoriser, texts_train, texts_holdout, y_train, y_holdout, args.retrieval_k),
    ]

    run_dir = new_run_dir()
    plot_macro_f1(
        results, [assets_dir() / "macro_f1_by_method.png", run_dir / "macro_f1_by_method.png"]
    )
    plot_confusion(
        y_holdout,
        best_predictions,
        labels,
        best.method,
        [assets_dir() / "confusion_matrix.png", run_dir / "confusion_matrix.png"],
    )

    index_meta = store.load_meta()
    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "proxy_disclaimer": PROXY_DISCLAIMER,
        "corpus": {
            "rows": int(len(frame)),
            "repos": sorted(frame["repo"].unique().tolist()),
            "embedding_model": index_meta.model if index_meta else None,
            "embedding_backend": index_meta.backend if index_meta else None,
            "embedding_content_hash": index_meta.content_hash if index_meta else None,
        },
        "split": split.model_dump(mode="json"),
        "class_map": class_map.model_dump(mode="json"),
        "classification": [result.model_dump(mode="json") for result in results],
        "best_method": best.method,
        "retrieval_proxy": [proxy.model_dump(mode="json") for proxy in proxies],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "sentence-transformers": _package_version("sentence-transformers"),
        },
        "config": vars(args),
    }
    write_metrics(run_dir, metrics)

    tfidf_result = next(r for r in results if r.method.startswith("TF-IDF"))
    persist_classifier(
        vectoriser,
        logreg,
        ClassifierBundle(
            classes=[*class_map.classes, OTHER],
            other_class=OTHER,
            fitted_at=datetime.now(UTC),
            seed=args.seed,
            n_train=split.n_train,
            train_last=split.train_last,
            holdout_first=split.holdout_first,
            measured_accuracy=tfidf_result.accuracy,
            measured_macro_f1=tfidf_result.macro_f1,
            source_run=run_dir.name,
            note=(
                "Fit on the training split only, exactly as evaluated. The measured "
                "scores describe this artefact. Accuracy is inflated by the dominant "
                "`other` class; macro-F1 is the meaningful figure."
            ),
        ),
        run_dir,
    )

    (run_dir / "class_map.json").write_text(class_map.model_dump_json(indent=2), encoding="utf-8")
    summary = summary_markdown(results, proxies, class_map, split, best)
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(f"\n[evals] wrote {run_dir}", flush=True)
    print(summary)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv if argv is not None else sys.argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
