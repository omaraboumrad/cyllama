"""ChromaDB-backed vector store for disk-based similarity search.

Drop-in replacement for VectorStore that uses ChromaDB instead of SQLite.
ChromaDB stores data on disk and only loads what's needed for each query.

Requires: pip install chromadb

Example:
    >>> from cyllama.rag.chroma_store import ChromaVectorStore
    >>> store = ChromaVectorStore(dimension=384, db_path="./chroma_db")
    >>> store.add([[0.1, 0.2, ...]], ["Hello world"])
    >>> results = store.search([0.1, 0.2, ...], k=5)
    >>> store.close()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import SearchResult


class ChromaVectorStore:
    """ChromaDB-based vector store with disk persistence.

    Compatible with cyllama's VectorStore interface, so it can be used
    as a drop-in replacement in the RAG pipeline.
    """

    def __init__(
        self,
        dimension: int,
        db_path: str = "./chroma_db",
        collection_name: str = "embeddings",
        metric: str = "cosine",
    ):
        """Initialize ChromaDB vector store.

        Args:
            dimension: Embedding dimension (must match your embeddings)
            db_path: Directory for ChromaDB persistent storage
            collection_name: Name of the collection
            metric: Distance metric: "cosine", "l2", or "ip" (inner product)
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )

        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self.dimension = dimension
        self.db_path = db_path
        self._closed = False
        self._next_id = 0

        # Map metric names
        metric_map = {
            "cosine": "cosine",
            "l2": "l2",
            "dot": "ip",
            "ip": "ip",
        }
        chroma_metric = metric_map.get(metric)
        if chroma_metric is None:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: {set(metric_map.keys())}"
            )

        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": chroma_metric},
        )

        # Set next_id based on existing data
        count = self._collection.count()
        if count > 0:
            existing = self._collection.get()
            max_id = max(int(id_) for id_ in existing["ids"])
            self._next_id = max_id + 1

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add embeddings with associated texts and metadata.

        Args:
            embeddings: List of embedding vectors
            texts: List of text strings
            metadata: Optional list of metadata dicts

        Returns:
            List of generated IDs
        """
        self._check_closed()

        if len(embeddings) != len(texts):
            raise ValueError(
                f"embeddings and texts must have same length: "
                f"{len(embeddings)} vs {len(texts)}"
            )

        if metadata is None:
            metadata = [{}] * len(embeddings)
        elif len(metadata) != len(embeddings):
            raise ValueError(
                f"metadata must have same length as embeddings: "
                f"{len(metadata)} vs {len(embeddings)}"
            )

        ids = []
        str_ids = []
        # ChromaDB doesn't accept empty metadata dicts or non-string values,
        # so serialize metadata as JSON strings
        chroma_metadata = []
        for i in range(len(embeddings)):
            current_id = self._next_id
            self._next_id += 1
            ids.append(current_id)
            str_ids.append(str(current_id))
            chroma_metadata.append({"_meta": json.dumps(metadata[i])} if metadata[i] else {})

        self._collection.add(
            ids=str_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=chroma_metadata if any(m for m in chroma_metadata) else None,
        )

        return ids

    def add_one(
        self,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a single embedding."""
        ids = self.add([embedding], [text], [metadata] if metadata else None)
        return ids[0]

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Find k most similar embeddings.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of SearchResult ordered by similarity
        """
        self._check_closed()

        count = self._collection.count()
        if count == 0:
            return []

        # Don't request more results than exist
        k = min(k, count)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            id_ = results["ids"][0][i]
            text = results["documents"][0][i]
            distance = results["distances"][0][i]
            raw_meta = results["metadatas"][0][i] if results["metadatas"] else {}

            # Deserialize metadata
            meta = {}
            if raw_meta and "_meta" in raw_meta:
                meta = json.loads(raw_meta["_meta"])

            # Convert distance to similarity score
            # ChromaDB returns distances (lower = more similar)
            score = 1.0 - distance  # Works for cosine

            if threshold is not None and score < threshold:
                continue

            search_results.append(
                SearchResult(
                    id=id_,
                    text=text,
                    score=score,
                    metadata=meta,
                )
            )

        return search_results

    def get(self, id: str | int) -> SearchResult | None:
        """Get a single embedding by ID."""
        self._check_closed()
        result = self._collection.get(ids=[str(id)], include=["documents", "metadatas"])
        if not result["ids"]:
            return None

        raw_meta = result["metadatas"][0] if result["metadatas"] else {}
        meta = {}
        if raw_meta and "_meta" in raw_meta:
            meta = json.loads(raw_meta["_meta"])

        return SearchResult(
            id=str(id),
            text=result["documents"][0],
            score=1.0,
            metadata=meta,
        )

    def delete(self, ids: list[str | int]) -> int:
        """Delete embeddings by ID."""
        self._check_closed()
        if not ids:
            return 0
        str_ids = [str(id_) for id_ in ids]
        self._collection.delete(ids=str_ids)
        return len(str_ids)

    def clear(self) -> int:
        """Delete all embeddings."""
        self._check_closed()
        count = len(self)
        if count > 0:
            all_ids = self._collection.get()["ids"]
            self._collection.delete(ids=all_ids)
        self._next_id = 0
        return count

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("ChromaVectorStore is closed")

    def close(self) -> None:
        """Close the store."""
        if not self._closed:
            self._closed = True

    def __len__(self) -> int:
        self._check_closed()
        return self._collection.count()

    def __contains__(self, id: str | int) -> bool:
        self._check_closed()
        result = self._collection.get(ids=[str(id)])
        return len(result["ids"]) > 0

    def __enter__(self) -> "ChromaVectorStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {len(self)} vectors"
        return (
            f"ChromaVectorStore(dimension={self.dimension}, "
            f"db_path={self.db_path!r}, status={status})"
        )
