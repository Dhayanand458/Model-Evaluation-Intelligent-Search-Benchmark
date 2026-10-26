"""Metric math, checked against cases worked out by hand.

Every expected value below is derived in the comments from the confusion counts,
not copied from a previous run of the code. If sklearn's conventions ever shift
under us (zero_division, label ordering, averaging), these fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage_mcp.evals import (
    OTHER,
    Latency,
    _proxy_from_neighbours,
    latency_stats,
    majority_baseline,
    majority_vote,
    score,
)

# Hand-worked case, 3 classes, 6 holdout issues:
#
#   idx  true   pred
#    0   a      a      correct
#    1   a      b
#    2   b      b      correct
#    3   b      other
#    4   other  other  correct
#    5   other  other  correct
#
# accuracy = 4/6 = 0.666667
#
# class a:     TP=1  FP=0  FN=1  -> P=1/1=1.0    R=1/2=0.5  F1=2(1·.5)/1.5 = 0.666667
# class b:     TP=1  FP=1  FN=1  -> P=1/2=0.5    R=1/2=0.5  F1=0.5
# class other: TP=2  FP=1  FN=0  -> P=2/3=.66667 R=2/2=1.0  F1=2(2/3)/(5/3) = 0.8
#
# macro-F1    = (0.666667 + 0.5 + 0.8) / 3           = 0.655556
# weighted-F1 = (2·0.666667 + 2·0.5 + 2·0.8) / 6     = 0.655556  (equal supports)
Y_TRUE = np.array(["a", "a", "b", "b", OTHER, OTHER], dtype=object)
Y_PRED = np.array(["a", "b", "b", OTHER, OTHER, OTHER], dtype=object)
LABELS = ["a", "b", OTHER]


def test_accuracy_matches_hand_count() -> None:
    result = score("hand", Y_TRUE, Y_PRED, LABELS)

    assert result.accuracy == pytest.approx(4 / 6)


def test_macro_f1_matches_hand_computation() -> None:
    result = score("hand", Y_TRUE, Y_PRED, LABELS)

    assert result.macro_f1 == pytest.approx((2 / 3 + 1 / 2 + 4 / 5) / 3)


def test_weighted_f1_matches_hand_computation() -> None:
    result = score("hand", Y_TRUE, Y_PRED, LABELS)

    assert result.weighted_f1 == pytest.approx((2 * (2 / 3) + 2 * 0.5 + 2 * 0.8) / 6)


def test_per_class_precision_recall_match_hand_computation() -> None:
    per_class = score("hand", Y_TRUE, Y_PRED, LABELS).per_class

    assert per_class["a"] == pytest.approx(
        {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    )
    assert per_class["b"] == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    )
    assert per_class[OTHER] == pytest.approx(
        {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "support": 2}
    )


def test_supports_sum_to_the_holdout_size() -> None:
    per_class = score("hand", Y_TRUE, Y_PRED, LABELS).per_class

    assert sum(scores["support"] for scores in per_class.values()) == len(Y_TRUE)


def test_a_class_never_predicted_scores_zero_not_nan() -> None:
    """zero_division=0: an unpredicted class must drag macro-F1 down, not blank it."""
    y_true = np.array(["a", "a", "b"], dtype=object)
    y_pred = np.array(["a", "a", "a"], dtype=object)

    result = score("hand", y_true, y_pred, ["a", "b"])

    # a: TP=2 FP=1 FN=0 -> P=2/3 R=1.0 F1=0.8 ;  b: TP=0 -> P=R=F1=0
    assert result.per_class["b"]["f1"] == 0.0
    assert result.macro_f1 == pytest.approx((0.8 + 0.0) / 2)


def test_perfect_prediction_scores_one() -> None:
    result = score("hand", Y_TRUE, Y_TRUE, LABELS)

    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0


# --------------------------------------------------------------------------- #
# Majority baseline
# --------------------------------------------------------------------------- #


def test_majority_baseline_predicts_the_commonest_training_class() -> None:
    y_train = np.array(["a", OTHER, OTHER, "b", OTHER], dtype=object)

    predictions = majority_baseline(y_train, 3)

    assert predictions.tolist() == [OTHER, OTHER, OTHER]


def test_majority_baseline_scores_as_hand_computed() -> None:
    """Predicting `other` everywhere: accuracy = 2/6, and only `other` gets any F1.

    other: TP=2 FP=4 FN=0 -> P=2/6=1/3  R=1.0  F1=2(1/3)/(4/3)=0.5
    a, b:  never predicted -> F1=0
    macro-F1 = 0.5/3 = 0.166667
    """
    y_train = np.array([OTHER] * 5, dtype=object)
    predictions = majority_baseline(y_train, len(Y_TRUE))

    result = score("majority", Y_TRUE, predictions, LABELS)

    assert result.accuracy == pytest.approx(2 / 6)
    assert result.macro_f1 == pytest.approx(0.5 / 3)


# --------------------------------------------------------------------------- #
# kNN vote
# --------------------------------------------------------------------------- #


def test_vote_picks_the_clear_majority() -> None:
    neighbours = np.array(["a", "b", "a", "a"], dtype=object)
    scores = np.array([0.9, 0.99, 0.8, 0.7])

    assert majority_vote(neighbours, scores) == "a"


def test_vote_breaks_ties_on_summed_similarity() -> None:
    """2-2 tie: `a` sums to 1.0, `b` to 0.7, so `a` wins despite `b` ranking first."""
    neighbours = np.array(["b", "a", "b", "a"], dtype=object)
    scores = np.array([0.5, 0.9, 0.2, 0.1])

    assert majority_vote(neighbours, scores) == "a"


def test_vote_breaks_exact_ties_alphabetically() -> None:
    """Identical counts and identical similarity sums must still be deterministic."""
    neighbours = np.array(["b", "a", "a", "b"], dtype=object)
    scores = np.array([0.5, 0.5, 0.5, 0.5])

    assert majority_vote(neighbours, scores) == "a"


# --------------------------------------------------------------------------- #
# Retrieval proxy
# --------------------------------------------------------------------------- #


def test_proxy_precision_matches_hand_computation() -> None:
    # query 0 (class a):     [a, a, b, other] -> 2/4 = 0.50
    # query 1 (class b):     [b, b, b, b]     -> 4/4 = 1.00
    # query 2 (class other): [a, b, other, o] -> 2/4 = 0.50
    #
    # all holdout        = (0.50 + 1.00 + 0.50) / 3 = 0.666667
    # excluding `other`  = (0.50 + 1.00) / 2        = 0.75
    neighbours = [
        np.array(["a", "a", "b", OTHER], dtype=object),
        np.array(["b", "b", "b", "b"], dtype=object),
        np.array(["a", "b", OTHER, OTHER], dtype=object),
    ]
    queries = np.array(["a", "b", OTHER], dtype=object)

    precision_all, precision_real, n_all, n_real = _proxy_from_neighbours(neighbours, queries)

    assert precision_all == pytest.approx(2 / 3)
    assert precision_real == pytest.approx(0.75)
    assert (n_all, n_real) == (3, 2)


def test_proxy_excluding_other_ignores_other_queries_entirely() -> None:
    """`other`-to-`other` agreement is noise matching noise; it must not count."""
    neighbours = [np.array([OTHER] * 4, dtype=object)] * 3
    queries = np.array([OTHER, OTHER, OTHER], dtype=object)

    precision_all, precision_real, _, n_real = _proxy_from_neighbours(neighbours, queries)

    assert precision_all == 1.0
    assert precision_real == 0.0
    assert n_real == 0


def test_proxy_is_zero_when_no_neighbour_shares_the_class() -> None:
    neighbours = [np.array(["b", "b"], dtype=object)]
    queries = np.array(["a"], dtype=object)

    precision_all, _, _, _ = _proxy_from_neighbours(neighbours, queries)

    assert precision_all == 0.0


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #


def test_latency_stats_match_hand_computation() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]

    stats = latency_stats(samples)

    assert isinstance(stats, Latency)
    assert stats.n_queries == 5
    assert stats.mean_ms == pytest.approx(3.0)
    assert stats.p50_ms == pytest.approx(3.0)
    # numpy linear interpolation: p95 of 1..5 = 1 + 0.95*(5-1) = 4.8
    assert stats.p95_ms == pytest.approx(4.8)
