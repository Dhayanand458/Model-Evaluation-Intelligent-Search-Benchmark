"""Leakage guards. A holdout issue must never be visible to a method.

These are the tests that make the headline numbers trustworthy: if any of them
fails, every metric in results/ is inflated by having seen the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triage_mcp import store
from triage_mcp.evals import (
    OTHER,
    assign_classes,
    build_class_map,
    is_component_label,
    time_split,
)


def make_frame(n: int, labels: list[list[str]] | None = None) -> pd.DataFrame:
    """Synthetic corpus, one issue per day, oldest first."""
    created = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "repo": ["owner/name"] * n,
            "number": list(range(1, n + 1)),
            "title": [f"title {i}" for i in range(n)],
            "body": [f"body {i}" for i in range(n)],
            "labels": labels if labels is not None else [[] for _ in range(n)],
            "state": ["open"] * n,
            "state_reason": [None] * n,
            "created_at": created,
            "closed_at": [pd.NaT] * n,
            "comments": [0] * n,
        }
    )


# --------------------------------------------------------------------------- #
# Time split
# --------------------------------------------------------------------------- #


def test_split_sizes() -> None:
    train_mask, holdout_mask, info = time_split(make_frame(100), 0.2)

    assert info.n_train == 80
    assert info.n_holdout == 20
    assert train_mask.sum() == 80
    assert holdout_mask.sum() == 20


def test_split_masks_partition_the_corpus() -> None:
    train_mask, holdout_mask, _ = time_split(make_frame(50), 0.2)

    assert not (train_mask & holdout_mask).any(), "an issue is in both splits"
    assert (train_mask | holdout_mask).all(), "an issue is in neither split"


def test_every_training_issue_predates_every_holdout_issue() -> None:
    """The core temporal guarantee -- no training issue is from the future."""
    frame = make_frame(200)
    train_mask, holdout_mask, info = time_split(frame, 0.2)

    assert frame["created_at"][train_mask].max() < frame["created_at"][holdout_mask].min()
    assert info.train_last < info.holdout_first


def test_timestamp_ties_do_not_straddle_the_boundary() -> None:
    """Issues sharing the boundary instant all land holdout-side.

    Splitting on a row index instead would put some of them in train and leave
    a training issue created at the same moment as a holdout issue.
    """
    frame = make_frame(100)
    boundary_value = frame.loc[80, "created_at"]
    frame.loc[75:85, "created_at"] = boundary_value

    train_mask, holdout_mask, _ = time_split(frame, 0.2)

    assert frame["created_at"][train_mask].max() < frame["created_at"][holdout_mask].min()
    assert holdout_mask[75:86].all(), "tied rows must all be holdout"


def test_split_rejects_a_degenerate_corpus() -> None:
    with pytest.raises(ValueError, match="no training rows"):
        time_split(make_frame(3), 1.0)


# --------------------------------------------------------------------------- #
# Retrieval candidate masking -- the leakage guard that matters most
# --------------------------------------------------------------------------- #


def _adversarial_matrix() -> tuple[np.ndarray, np.ndarray]:
    """10 unit vectors where the two holdout rows are each other's nearest match.

    Constructed so that a broken mask is impossible to miss: query with row 8
    and the two best answers in the whole matrix are rows 8 and 9, both holdout.
    """
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(10, 8)).astype(np.float32)
    matrix[9] = matrix[8]
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    train_mask = np.array([True] * 8 + [False] * 2)
    return matrix, train_mask


def test_the_guard_test_is_actually_adversarial() -> None:
    """Without a mask the holdout rows win, so masking them out is real work."""
    matrix, _ = _adversarial_matrix()

    indices, _ = store.top_k(matrix[8], matrix, k=3)

    assert 8 in indices[0] and 9 in indices[0]


def test_candidate_mask_excludes_holdout_rows() -> None:
    matrix, train_mask = _adversarial_matrix()

    indices, _ = store.top_k(matrix[8], matrix, k=5, candidate_mask=train_mask)

    assert set(indices[0].tolist()) <= set(range(8))


def test_a_holdout_issue_cannot_retrieve_itself() -> None:
    matrix, train_mask = _adversarial_matrix()

    for row in (8, 9):
        indices, _ = store.top_k(matrix[row], matrix, k=8, candidate_mask=train_mask)
        assert row not in indices[0].tolist()


def test_every_holdout_row_is_masked_for_every_query() -> None:
    matrix, train_mask = _adversarial_matrix()
    holdout = set(np.flatnonzero(~train_mask).tolist())

    indices, _ = store.top_k(matrix, matrix, k=4, candidate_mask=train_mask)

    assert holdout.isdisjoint(indices.ravel().tolist())


def test_scores_are_descending() -> None:
    matrix, train_mask = _adversarial_matrix()

    _, scores = store.top_k(matrix, matrix, k=6, candidate_mask=train_mask)

    assert np.all(np.diff(scores, axis=1) <= 1e-6)


def test_k_larger_than_candidate_set_is_clamped() -> None:
    matrix, train_mask = _adversarial_matrix()

    indices, _ = store.top_k(matrix[0], matrix, k=999, candidate_mask=train_mask)

    assert indices.shape[1] == int(train_mask.sum())


def test_empty_candidate_set_is_an_error() -> None:
    matrix, _ = _adversarial_matrix()

    with pytest.raises(ValueError, match="candidate set is empty"):
        store.top_k(matrix[0], matrix, k=3, candidate_mask=np.zeros(10, dtype=bool))


def test_mismatched_mask_length_is_an_error() -> None:
    matrix, _ = _adversarial_matrix()

    with pytest.raises(ValueError, match="mask covers"):
        store.top_k(matrix[0], matrix, k=3, candidate_mask=np.ones(5, dtype=bool))


def test_search_is_deterministic() -> None:
    matrix, train_mask = _adversarial_matrix()

    first = store.top_k(matrix, matrix, k=5, candidate_mask=train_mask)
    second = store.top_k(matrix, matrix, k=5, candidate_mask=train_mask)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


# --------------------------------------------------------------------------- #
# Class construction
# --------------------------------------------------------------------------- #


def test_status_and_type_labels_are_not_components() -> None:
    assert is_component_label("terminal") is True
    assert is_component_label("info-needed") is False
    assert is_component_label("bug") is False
    assert is_component_label("*duplicate") is False
    assert is_component_label("ai-generated") is False


@pytest.mark.parametrize(
    "label", ["duplicate", "*duplicate", "confirmation-pending", "on-release-notes", "ai-triaged"]
)
def test_bare_status_labels_are_excluded_not_just_prefixed_ones(label: str) -> None:
    """vscode ships both `*duplicate` and a bare `duplicate`.

    The asterisk rule only catches the prefixed form, so the bare ones must be
    named explicitly -- otherwise a resolution label becomes a component class
    and the whole task is contaminated.
    """
    assert is_component_label(label) is False


def test_real_component_labels_survive_the_stoplist() -> None:
    """Guard the other direction: the stoplist must not eat genuine areas."""
    for label in ["chat", "chat-billing", "terminal", "scm", "git", "accessibility", "debug"]:
        assert is_component_label(label) is True, label


def test_classes_are_ranked_on_training_frequency_only() -> None:
    """Ranking must not see holdout counts.

    `editor` leads in the training period (15 vs 10) but trails over the corpus
    as a whole (17 vs 28), because `notebook` surges after the split. Ranking on
    corpus frequency would put `notebook` first; ranking on training frequency
    alone puts `editor` first. Both clear the support guard, so ordering is the
    only thing under test.
    """
    labels: list[list[str]] = [[] for _ in range(100)]
    for i in range(0, 15):
        labels[i] = ["editor"]
    for i in range(15, 25):
        labels[i] = ["notebook"]
    for i in range(80, 82):
        labels[i] = ["editor"]
    for i in range(82, 100):
        labels[i] = ["notebook"]
    frame = make_frame(100, labels)
    train_mask, holdout_mask, _ = time_split(frame, 0.2)

    class_map = build_class_map(frame, train_mask, holdout_mask, k=2, min_holdout_support=1)

    assert class_map.classes == ["editor", "notebook"], "ranking used holdout counts"
    assert class_map.train_counts["editor"] == 15
    assert class_map.holdout_counts["notebook"] == 18


def test_classes_without_holdout_support_are_demoted() -> None:
    n = 100
    labels = [["editor"] if i < 80 else ["notebook"] for i in range(n)]
    frame = make_frame(n, labels)
    train_mask, holdout_mask, _ = time_split(frame, 0.2)

    class_map = build_class_map(frame, train_mask, holdout_mask, k=4, min_holdout_support=10)

    assert "editor" not in class_map.classes, "editor has zero holdout examples"
    assert [entry.label for entry in class_map.demoted] == ["editor"]
    assert class_map.demoted[0].holdout_count == 0


def test_multi_label_issues_take_the_most_frequent_class() -> None:
    classes = ["editor", "terminal"]  # ranked: editor is more frequent
    frame = make_frame(2, [["terminal", "editor"], ["terminal"]])

    assigned = assign_classes(frame, classes)

    assert assigned.tolist() == ["editor", "terminal"]


def test_issues_with_no_class_label_become_other() -> None:
    frame = make_frame(2, [["info-needed"], []])

    assert assign_classes(frame, ["editor"]).tolist() == [OTHER, OTHER]


def test_class_assignment_is_order_independent() -> None:
    """Label order within an issue must not change its class."""
    forward = make_frame(1, [["terminal", "editor"]])
    backward = make_frame(1, [["editor", "terminal"]])

    assert assign_classes(forward, ["editor", "terminal"]) == assign_classes(
        backward, ["editor", "terminal"]
    )


def test_other_composition_is_reported() -> None:
    labels = [[], ["info-needed"], ["editor"], []]
    frame = make_frame(4, labels)
    train_mask, holdout_mask, _ = time_split(frame, 0.25)

    class_map = build_class_map(frame, train_mask, holdout_mask, k=3, min_holdout_support=0)

    assert class_map.other_composition["unlabelled_entirely"] == 2
    assert class_map.other_composition["labelled_but_no_class_label"] == 1
