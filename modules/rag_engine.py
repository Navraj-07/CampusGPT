"""Hybrid RAG Engine: Local SentenceTransformer + FAISS Dense Index + BM25 RRF."""

import json
import math
import re
import shutil
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from modules.document_processor import sanitize_text

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], SentenceTransformer] = {}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2", cache_folder: Optional[str] = None
) -> SentenceTransformer:
    """Thread-safe lazy loader for the local SentenceTransformer embedding model."""
    cache_key = (model_name, cache_folder or "")
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _MODEL_LOCK:
        if cache_key not in _MODEL_CACHE:
            kwargs: dict[str, Any] = {}
            if cache_folder:
                Path(cache_folder).mkdir(parents=True, exist_ok=True)
                kwargs["cache_folder"] = cache_folder
            try:
                _MODEL_CACHE[cache_key] = SentenceTransformer(
                    model_name, local_files_only=True, **kwargs
                )
            except Exception:
                _MODEL_CACHE[cache_key] = SentenceTransformer(model_name, **kwargs)
    return _MODEL_CACHE[cache_key]


def _tokenize_for_bm25(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _compute_bm25_scores(query: str, chunks: list[dict[str, Any]]) -> list[float]:
    """Compute Okapi BM25 scores for a query across a list of chunks."""
    q_terms = _tokenize_for_bm25(query)
    if not q_terms or not chunks:
        return [0.0] * len(chunks)

    tokenized_docs = [_tokenize_for_bm25(c.get("text", "")) for c in chunks]
    num_docs = len(tokenized_docs)
    avg_dl = sum(len(doc) for doc in tokenized_docs) / max(1, num_docs)

    doc_freqs: Counter[str] = Counter()
    term_freqs: list[Counter[str]] = []
    for tokens in tokenized_docs:
        tf = Counter(tokens)
        term_freqs.append(tf)
        for term in tf:
            doc_freqs[term] += 1

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for idx, tokens in enumerate(tokenized_docs):
        dl = len(tokens)
        tf_map = term_freqs[idx]
        score = 0.0
        for term in q_terms:
            df = doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            tf = tf_map.get(term, 0)
            denom = tf + k1 * (1.0 - b + b * (dl / max(1.0, avg_dl)))
            score += idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
        scores.append(score)
    return scores


class RAGEngine:
    """Manages per-document FAISS indices and Hybrid (FAISS + BM25) search."""

    def __init__(
        self,
        vector_store_folder: str,
        model_name: str = "all-MiniLM-L6-v2",
        model_cache_folder: Optional[str] = None,
    ) -> None:
        self.vector_store_folder = Path(vector_store_folder)
        self.vector_store_folder.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.model_cache_folder = model_cache_folder

    def _doc_dir(self, doc_id: int) -> Path:
        return self.vector_store_folder / str(doc_id)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode sanitized texts into L2-normalized float32 embeddings."""
        model = get_embedding_model(self.model_name, self.model_cache_folder)
        clean_texts = [sanitize_text(t) for t in texts]
        vectors = model.encode(
            clean_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors

    def build_index(
        self, doc_id: int, chunks: list[dict[str, Any]], doc_name: str = ""
    ) -> int:
        """Build and persist a FAISS IndexFlatIP + metadata sidecar for a document."""
        if not chunks:
            raise ValueError("Cannot build FAISS index for an empty chunk list.")

        doc_dir = self._doc_dir(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        texts = [sanitize_text(c["text"]) for c in chunks]
        embeddings = self._encode_texts(texts)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        stored_chunks: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            stored_chunks.append(
                {
                    "chunk_index": int(chunk.get("chunk_index", idx)),
                    "page": int(chunk.get("page", 1)),
                    "text": texts[idx],
                    "doc_id": int(doc_id),
                    "doc_name": doc_name,
                }
            )

        faiss.write_index(index, str(doc_dir / "index.faiss"))
        with open(doc_dir / "chunks.json", "w", encoding="utf-8") as fh:
            json.dump(stored_chunks, fh, ensure_ascii=False, indent=2)

        return len(stored_chunks)

    def load_index(self, doc_id: int) -> tuple[faiss.Index, list[dict[str, Any]]]:
        """Load a document's FAISS index and chunk metadata from disk."""
        doc_dir = self._doc_dir(doc_id)
        index_path = doc_dir / "index.faiss"
        chunks_path = doc_dir / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Vector index not found for document ID {doc_id}."
            )

        index = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as fh:
            chunks = json.load(fh)
        return index, chunks

    def get_document_chunks(
        self, doc_id: int, max_chunks: int = 30
    ) -> list[dict[str, Any]]:
        """Return evenly sampled representative chunks for full-document operations."""
        _, chunks = self.load_index(doc_id)
        if len(chunks) <= max_chunks:
            return chunks

        step = len(chunks) / max_chunks
        return [chunks[int(i * step)] for i in range(max_chunks)]

    def search(
        self, query: str, doc_ids: list[int], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Hybrid Search across one or more documents combining FAISS dense cosine similarity
        and BM25 keyword matching via weighted score + Reciprocal Rank Fusion (RRF).
        """
        clean_query = sanitize_text(query)
        if not clean_query or not doc_ids:
            return []

        query_vec = self._encode_texts([clean_query])
        all_candidates: list[dict[str, Any]] = []

        for doc_id in doc_ids:
            try:
                index, chunks = self.load_index(int(doc_id))
            except FileNotFoundError:
                continue

            if index.ntotal == 0 or not chunks:
                continue

            # Score all dense chunks when doc is small or top candidate_k when large
            candidate_k = min(max(top_k * 4, 20), index.ntotal)
            dense_scores, dense_indices = index.search(query_vec, candidate_k)

            dense_map: dict[int, float] = {}
            for sc, c_idx in zip(dense_scores[0], dense_indices[0]):
                if 0 <= c_idx < len(chunks):
                    dense_map[int(c_idx)] = max(0.0, float(sc))

            bm25_raw = _compute_bm25_scores(clean_query, chunks)
            max_bm25 = max(bm25_raw) if bm25_raw else 0.0

            # Include top BM25 matches even if dense score was outside candidate_k
            bm25_ranked = sorted(
                range(len(chunks)), key=lambda i: bm25_raw[i], reverse=True
            )[:candidate_k]

            candidate_indices = set(dense_map.keys()) | {
                i for i in bm25_ranked if bm25_raw[i] > 0
            }

            for c_idx in candidate_indices:
                chunk = dict(chunks[c_idx])
                d_score = dense_map.get(c_idx, 0.0)
                b_norm = (bm25_raw[c_idx] / max_bm25) if max_bm25 > 0 else 0.0
                chunk["dense_score"] = round(d_score, 4)
                chunk["bm25_score"] = round(b_norm, 4)
                all_candidates.append(chunk)

        if not all_candidates:
            return []

        # Compute RRF ranks across all documents
        dense_order = sorted(
            range(len(all_candidates)),
            key=lambda i: all_candidates[i]["dense_score"],
            reverse=True,
        )
        bm25_order = sorted(
            range(len(all_candidates)),
            key=lambda i: all_candidates[i]["bm25_score"],
            reverse=True,
        )

        dense_rank = {idx: rank + 1 for rank, idx in enumerate(dense_order)}
        bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25_order)}

        for idx, item in enumerate(all_candidates):
            rrf = (0.65 / (40 + dense_rank[idx])) + (0.35 / (40 + bm25_rank[idx]))
            blended = (0.7 * item["dense_score"]) + (0.3 * item["bm25_score"])
            item["rrf_score"] = round(rrf, 6)
            item["score"] = round(max(blended, item["dense_score"] * 0.85), 4)

        all_candidates.sort(
            key=lambda item: (item["rrf_score"], item["score"]), reverse=True
        )
        return all_candidates[:top_k]

    def delete_index(self, doc_id: int) -> None:
        """Delete a document's FAISS index directory if present."""
        doc_dir = self._doc_dir(doc_id)
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
