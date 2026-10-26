"""Embedding-input construction, including the shapes Parquet actually returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triage_mcp.store import (
    MAX_BODY_CHARS,
    content_hash,
    corpus_texts,
    embed_texts,
    issue_text,
    l2_normalise,
    mean_pool,
)


def test_title_and_body_are_joined() -> None:
    assert issue_text("Crash on save", "Steps to repro") == "Crash on save\n\nSteps to repro"


def test_long_bodies_are_truncated_but_titles_are_not() -> None:
    title = "t" * 500
    text = issue_text(title, "b" * 5000, max_body_chars=100)

    assert text.startswith(title), "the title carries most of the signal; never truncate it"
    assert text.count("b") == 100


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, pd.NaT])
def test_missing_body_does_not_leak_a_non_string(missing: object) -> None:
    """Parquet hands back NaN/pd.NA for null text, and NaN is truthy."""
    assert issue_text("only a title", missing) == "only a title"


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_missing_title_is_tolerated(missing: object) -> None:
    assert issue_text(missing, "body text") == "body text"


def test_entirely_empty_issue_is_empty_string() -> None:
    assert issue_text(None, None) == ""


def test_corpus_texts_survives_a_parquet_round_trip(tmp_path: object) -> None:
    """The real failure mode: null bodies become floats after read_parquet."""
    frame = pd.DataFrame(
        {
            "title": ["has body", "no body"],
            "body": ["some text", None],
        }
    )
    path = tmp_path / "t.parquet"  # type: ignore[operator]
    frame.to_parquet(path, index=False)

    texts = corpus_texts(pd.read_parquet(path))

    assert texts == ["has body\n\nsome text", "no body"]


def test_corpus_texts_is_row_aligned() -> None:
    frame = pd.DataFrame({"title": ["a", "b", "c"], "body": ["1", "2", "3"]})

    assert corpus_texts(frame) == ["a\n\n1", "b\n\n2", "c\n\n3"]


# --------------------------------------------------------------------------- #
# Content hash -- the cache key that makes re-runs free
# --------------------------------------------------------------------------- #


def test_hash_is_stable_for_identical_input() -> None:
    texts = ["one", "two"]

    assert content_hash(texts, "m", 100) == content_hash(list(texts), "m", 100)


def test_hash_changes_when_the_corpus_changes() -> None:
    assert content_hash(["a"], "m", 100) != content_hash(["b"], "m", 100)


def test_hash_changes_when_the_model_changes() -> None:
    """Same text, different encoder, different vectors -- must not reuse the cache."""
    assert content_hash(["a"], "model-x", 100) != content_hash(["a"], "model-y", 100)


def test_hash_changes_when_truncation_changes() -> None:
    assert content_hash(["a"], "m", 100) != content_hash(["a"], "m", 200)


def test_hash_distinguishes_row_boundaries() -> None:
    """['ab'] and ['a','b'] must not collide -- concatenation alone would."""
    joined = content_hash(["ab"], "m", MAX_BODY_CHARS)
    split = content_hash(["a", "b"], "m", MAX_BODY_CHARS)

    assert joined != split


def test_hash_is_order_sensitive() -> None:
    """Row order is the alignment contract with the parquet, so it must matter."""
    assert content_hash(["a", "b"], "m", 100) != content_hash(["b", "a"], "m", 100)


def test_hash_handles_non_ascii() -> None:
    assert isinstance(content_hash(["日本語 🎉"], "m", 100), str)


# --------------------------------------------------------------------------- #
# Pooling -- what makes the ONNX backend equivalent to sentence-transformers
# --------------------------------------------------------------------------- #


def test_mean_pool_averages_over_real_tokens() -> None:
    # one sequence, 3 token positions, 2 dims; all positions real
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 1]], dtype=np.int64)

    assert mean_pool(hidden, mask) == pytest.approx(np.array([[3.0, 4.0]]))


def test_mean_pool_ignores_padding() -> None:
    """Padding must not drag the mean toward zero -- this is the whole point."""
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)

    # mean of the two real positions only: (1+3)/2, (2+4)/2
    assert mean_pool(hidden, mask) == pytest.approx(np.array([[2.0, 3.0]]))


def test_mean_pool_handles_an_all_padding_row_without_dividing_by_zero() -> None:
    hidden = np.ones((1, 2, 2), dtype=np.float32)
    mask = np.array([[0, 0]], dtype=np.int64)

    assert np.isfinite(mean_pool(hidden, mask)).all()


def test_mean_pool_is_row_independent() -> None:
    hidden = np.array([[[1.0, 1.0], [3.0, 3.0]], [[10.0, 10.0], [0.0, 0.0]]], dtype=np.float32)
    mask = np.array([[1, 1], [1, 0]], dtype=np.int64)

    assert mean_pool(hidden, mask) == pytest.approx(np.array([[2.0, 2.0], [10.0, 10.0]]))


def test_l2_normalise_gives_unit_rows() -> None:
    """Cosine == dot product downstream depends on this; assert the invariant."""
    normalised = l2_normalise(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    assert np.linalg.norm(normalised, axis=1) == pytest.approx([1.0, 1.0])
    assert normalised[0] == pytest.approx([0.6, 0.8])


def test_l2_normalise_survives_a_zero_row() -> None:
    normalised = l2_normalise(np.zeros((1, 3), dtype=np.float32))

    assert np.isfinite(normalised).all()


def test_embed_texts_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        embed_texts(["x"], backend="magic")
