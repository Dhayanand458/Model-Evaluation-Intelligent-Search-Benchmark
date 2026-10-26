"""Embedding index over the ingested issue corpus.

Row `i` of ``data/embeddings.npy`` corresponds to row `i` of
``data/issues.parquet``. That alignment is the whole contract -- everything
downstream indexes the two together -- so the index carries a content hash of
the exact texts it was built from and refuses to load against a corpus that has
moved underneath it.

Search is exact cosine over the full matrix. At a few tens of thousands of rows
a brute-force dot product is milliseconds, so there is no vector index to keep
in sync and no approximate-recall error term to reason about.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:  # import-time cost avoided; these are only needed for typing
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer

from triage_mcp.ingest import load_dataset
from triage_mcp.paths import data_dir

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
EMBEDDING_DIM = 384
ONNX_MODEL_FILE = "onnx/model.onnx"
MAX_SEQUENCE_TOKENS = 256  # all-MiniLM-L6-v2's trained context

# all-MiniLM-L6-v2 truncates at 256 word-pieces (~1000 chars), so the model
# would drop the tail regardless. Cutting earlier just saves tokenizer work;
# the title is kept whole because it carries most of the triage signal.
MAX_BODY_CHARS = 2000


class IndexMeta(BaseModel):
    """Provenance for the embedding matrix on disk."""

    model: str
    rows: int
    dim: int
    content_hash: str
    max_body_chars: int
    built_at: datetime
    backend: str = "unknown"


def embeddings_path() -> Path:
    return data_dir() / "embeddings.npy"


def meta_path() -> Path:
    return data_dir() / "embeddings.meta.json"


# --------------------------------------------------------------------------- #
# Text preparation
# --------------------------------------------------------------------------- #


def _as_text(value: object) -> str:
    """Coerce a possibly-missing cell to a string.

    Parquet returns missing text as NaN or pd.NA rather than None, and NaN is
    truthy -- so `value or ""` silently passes a float through to string code.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):  # arrays and other non-scalars
        pass
    return str(value)


def issue_text(title: object, body: object, max_body_chars: int = MAX_BODY_CHARS) -> str:
    """The string that gets embedded: full title, truncated body."""
    title_text = _as_text(title).strip()
    body_text = _as_text(body).strip()
    if len(body_text) > max_body_chars:
        body_text = body_text[:max_body_chars]
    return f"{title_text}\n\n{body_text}".strip()


def corpus_texts(frame: pd.DataFrame, max_body_chars: int = MAX_BODY_CHARS) -> list[str]:
    """Embedding inputs for every row, in parquet order."""
    return [
        issue_text(title, body, max_body_chars)
        for title, body in zip(frame["title"], frame["body"], strict=True)
    ]


def content_hash(texts: list[str], model_name: str, max_body_chars: int) -> str:
    """Fingerprint of exactly what was embedded, so stale indexes are detected.

    Covers the model and truncation settings too: changing either changes the
    vectors even when the corpus is untouched.
    """
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(str(max_body_chars).encode("utf-8"))
    digest.update(str(len(texts)).encode("utf-8"))
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def mean_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average token vectors over non-padding positions.

    This is the pooling all-MiniLM-L6-v2 was trained with, so reproducing it
    exactly is what makes the ONNX backend interchangeable with the
    sentence-transformers one. Padding positions must not dilute the mean,
    hence the mask rather than a plain `.mean(axis=1)`.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden_states * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so cosine similarity is a plain dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


@lru_cache(maxsize=2)
def _sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load and cache the encoder.

    Cached because the server embeds one query per tool call: reconstructing the
    model each time would add seconds of load to every request.
    """
    # Imported lazily: loading torch costs seconds, and every consumer that only
    # reads a prebuilt matrix should not pay it.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _onnx_encoder(model_name: str) -> tuple[ort.InferenceSession, Tokenizer, frozenset[str]]:
    """Load and cache the ONNX session plus tokenizer. See `_sentence_transformer`."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    session = ort.InferenceSession(
        hf_hub_download(model_name, ONNX_MODEL_FILE), providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_TOKENS)
    tokenizer.enable_padding()  # per batch, to the longest member
    return session, tokenizer, frozenset(spec.name for spec in session.get_inputs())


def _embed_sentence_transformers(
    texts: list[str], model_name: str, batch_size: int, show_progress: bool
) -> np.ndarray:
    model = _sentence_transformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _embed_onnx(
    texts: list[str], model_name: str, batch_size: int, show_progress: bool
) -> np.ndarray:
    """Run the same model through onnxruntime, with no torch in the picture."""
    session, tokenizer, input_names = _onnx_encoder(model_name)

    starts: Iterable[int] = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm

        starts = tqdm(list(starts), desc="embedding", unit="batch")

    chunks: list[np.ndarray] = []
    for start in starts:
        encoded = tokenizer.encode_batch(texts[start : start + batch_size])
        mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
        feed = {
            "input_ids": np.array([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": mask,
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array([item.type_ids for item in encoded], dtype=np.int64)
        hidden = session.run(None, feed)[0]
        chunks.append(mean_pool(hidden, mask))

    if not chunks:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return l2_normalise(np.vstack(chunks)).astype(np.float32)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
    backend: str = "auto",
) -> tuple[np.ndarray, str]:
    """Encode texts to L2-normalised float32 vectors. Returns (vectors, backend).

    Two interchangeable backends run the *same* model. sentence-transformers is
    preferred, but it imports torch, and torch's DLLs are unsigned -- on a
    machine with Windows Smart App Control enforced they are blocked outright
    and the import raises OSError. Rather than drop to a weaker embedding, fall
    back to onnxruntime, which loads fine under that policy and reproduces
    MiniLM's mean-pool-then-normalise pipeline exactly.

    The backend actually used is recorded in the index metadata and in
    metrics.json, because "which code produced these vectors" is part of
    reproducing a number.
    """
    if backend not in {"auto", "sentence-transformers", "onnx"}:
        raise ValueError(f"unknown backend {backend!r}")

    if backend in {"auto", "sentence-transformers"}:
        try:
            return _embed_sentence_transformers(texts, model_name, batch_size, show_progress), (
                "sentence-transformers"
            )
        except (ImportError, OSError) as exc:
            if backend != "auto":
                raise
            print(f"[store] sentence-transformers backend unavailable: {exc}", file=sys.stderr)
            print("[store] falling back to onnxruntime (same model, no torch)", file=sys.stderr)

    return _embed_onnx(texts, model_name, batch_size, show_progress), "onnx"


def load_meta() -> IndexMeta | None:
    path = meta_path()
    if not path.exists():
        return None
    try:
        return IndexMeta.model_validate_json(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError):
        return None


def build_index(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_body_chars: int = MAX_BODY_CHARS,
    force: bool = False,
    show_progress: bool = True,
) -> IndexMeta:
    """Encode data/issues.parquet and persist the matrix, skipping no-op rebuilds."""
    frame = load_dataset()
    texts = corpus_texts(frame, max_body_chars)
    digest = content_hash(texts, model_name, max_body_chars)

    existing = load_meta()
    is_current = (
        not force
        and existing is not None
        and existing.content_hash == digest
        and embeddings_path().exists()
    )
    if is_current and existing is not None:
        print(f"[store] index is current ({existing.rows} rows, {existing.model}); nothing to do")
        return existing

    print(f"[store] embedding {len(texts)} issues with {model_name}")
    vectors, backend = embed_texts(texts, model_name, batch_size, show_progress)

    meta = IndexMeta(
        model=model_name,
        rows=int(vectors.shape[0]),
        dim=int(vectors.shape[1]),
        content_hash=digest,
        max_body_chars=max_body_chars,
        built_at=datetime.now(UTC),
        backend=backend,
    )
    embeddings_path().parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path(), vectors)
    meta_path().write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    print(f"[store] wrote {meta.rows}x{meta.dim} to {embeddings_path()}")
    return meta


def load_embeddings(frame: pd.DataFrame | None = None) -> np.ndarray:
    """Load the matrix, verifying it still lines up with the corpus.

    A silent row-count mismatch would misattribute every neighbour, so this is a
    hard failure rather than a warning.
    """
    path = embeddings_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `make embed` first")
    vectors = np.load(path)

    meta = load_meta()
    if frame is None:
        frame = load_dataset()
    if len(frame) != vectors.shape[0]:
        raise ValueError(
            f"embeddings ({vectors.shape[0]} rows) do not match corpus ({len(frame)} rows); "
            "rebuild with `make embed`"
        )
    if meta is not None:
        digest = content_hash(
            corpus_texts(frame, meta.max_body_chars), meta.model, meta.max_body_chars
        )
        if digest != meta.content_hash:
            raise ValueError("corpus changed since the index was built; rebuild with `make embed`")
    return vectors


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def top_k(
    queries: np.ndarray,
    matrix: np.ndarray,
    k: int,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k, optionally restricted to a subset of rows.

    `candidate_mask` is a boolean array over `matrix` rows. Evals rely on it to
    keep holdout issues out of the candidate pool entirely -- masking after
    scoring would still let a holdout row occupy a top-k slot, so the restriction
    is applied to the matrix before any similarity is computed.

    Both inputs must be L2-normalised, which makes the dot product the cosine.
    Returns (indices into `matrix`, scores), each shaped (n_queries, k_effective),
    ordered most similar first.
    """
    if queries.ndim == 1:
        queries = queries[None, :]
    if k < 1:
        raise ValueError("k must be >= 1")

    if candidate_mask is None:
        n_candidates = matrix.shape[0]
    else:
        if candidate_mask.shape[0] != matrix.shape[0]:
            raise ValueError(
                f"mask covers {candidate_mask.shape[0]} rows but matrix has {matrix.shape[0]}"
            )
        n_candidates = int(candidate_mask.sum())
    if n_candidates == 0:
        raise ValueError("candidate set is empty")

    # Score every row, then sink the excluded ones to -inf. Slicing the matrix
    # to the candidates instead would copy tens of MB per call, which both slows
    # the eval loops and pollutes their per-query latency measurements.
    scores = queries @ matrix.T
    if candidate_mask is not None:
        scores[:, ~candidate_mask] = -np.inf

    # k never exceeds the candidate count, so a -inf (excluded) row can never
    # be picked: there are always k_eff finite scores that outrank it.
    k_eff = min(k, n_candidates)

    # argpartition for the cut, then a stable sort of just that slice, so
    # repeated runs agree.
    part = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1, kind="stable")
    picked = part[rows, order]
    return picked, scores[rows, picked]


def main() -> int:
    build_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
