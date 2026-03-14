"""Tests for the ChromaDB vector store."""

import tempfile
import uuid

import pytest

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from cyllama.rag.types import SearchResult

pytestmark = pytest.mark.skipif(
    not HAS_CHROMADB,
    reason="chromadb not installed. Install with: pip install chromadb"
)


def _make_ephemeral_store(dimension=4):
    """Create a ChromaVectorStore backed by an EphemeralClient (no disk, no file descriptors)."""
    from cyllama.rag.chroma_store import ChromaVectorStore
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.dimension = dimension
    store.db_path = ":memory:"
    store._closed = False
    store._next_id = 0
    client = chromadb.EphemeralClient()
    store._client = client
    store._collection = client.get_or_create_collection(
        name=f"test_{uuid.uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )
    return store


@pytest.fixture
def store():
    """Create an ephemeral ChromaVectorStore for testing (no disk I/O)."""
    s = _make_ephemeral_store()
    yield s
    s.close()


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
    ]


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "First document",
        "Second document",
        "Third document",
        "Fourth document",
        "Fifth document",
    ]


class TestChromaStoreInit:
    """Test ChromaVectorStore initialization."""

    def test_init_default(self):
        from cyllama.rag.chroma_store import ChromaVectorStore
        with tempfile.TemporaryDirectory() as d:
            with ChromaVectorStore(dimension=384, db_path=d) as store:
                assert store.dimension == 384

    def test_init_invalid_dimension(self):
        from cyllama.rag.chroma_store import ChromaVectorStore
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="dimension must be positive"):
                ChromaVectorStore(dimension=0, db_path=d)
            with pytest.raises(ValueError, match="dimension must be positive"):
                ChromaVectorStore(dimension=-1, db_path=d)

    def test_init_invalid_metric(self):
        from cyllama.rag.chroma_store import ChromaVectorStore
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="Invalid metric"):
                ChromaVectorStore(dimension=4, db_path=d, metric="invalid")

    def test_init_valid_metrics(self):
        from cyllama.rag.chroma_store import ChromaVectorStore
        for metric in ["cosine", "l2", "dot", "ip"]:
            with tempfile.TemporaryDirectory() as d:
                with ChromaVectorStore(dimension=4, db_path=d, metric=metric) as s:
                    assert s.dimension == 4


class TestChromaStoreAdd:
    """Test adding embeddings."""

    def test_add_single(self, store):
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test text"])
        assert len(ids) == 1
        assert isinstance(ids[0], int)
        assert len(store) == 1

    def test_add_multiple(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        assert len(ids) == 5
        assert len(store) == 5

    def test_add_with_metadata(self, store):
        ids = store.add(
            [[1.0, 0.0, 0.0, 0.0]],
            ["test"],
            metadata=[{"source": "doc1", "page": 1}],
        )
        result = store.get(ids[0])
        assert result.metadata == {"source": "doc1", "page": 1}

    def test_add_one(self, store):
        id_ = store.add_one([1.0, 0.0, 0.0, 0.0], "single text")
        assert isinstance(id_, int)
        assert len(store) == 1

    def test_add_one_with_metadata(self, store):
        id_ = store.add_one(
            [1.0, 0.0, 0.0, 0.0],
            "text",
            metadata={"key": "value"},
        )
        result = store.get(id_)
        assert result.metadata == {"key": "value"}

    def test_add_mismatched_lengths(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["text1", "text2"])

    def test_add_metadata_wrong_length(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add(
                [[1.0, 0.0, 0.0, 0.0]],
                ["text"],
                metadata=[{}, {}],
            )


class TestChromaStoreSearch:
    """Test similarity search."""

    def test_search_basic(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].text == "First document"

    def test_search_returns_ordered(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_larger_than_count(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["only one"])
        results = store.search([1.0, 0.0, 0.0, 0.0], k=10)
        assert len(results) == 1

    def test_search_empty_store(self, store):
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert results == []

    def test_search_with_threshold(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5, threshold=0.9)
        assert len(results) <= 2

    def test_search_result_fields(self, store):
        store.add(
            [[1.0, 0.0, 0.0, 0.0]],
            ["test text"],
            metadata=[{"key": "value"}],
        )
        results = store.search([1.0, 0.0, 0.0, 0.0], k=1)
        result = results[0]
        assert result.id is not None
        assert result.text == "test text"
        assert isinstance(result.score, float)
        assert result.metadata == {"key": "value"}


class TestChromaStoreGet:
    """Test get operations."""

    def test_get_existing(self, store):
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        result = store.get(ids[0])
        assert result is not None
        assert result.text == "test"

    def test_get_nonexistent(self, store):
        result = store.get(999)
        assert result is None

    def test_get_with_string_id(self, store):
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        result = store.get(str(ids[0]))
        assert result is not None


class TestChromaStoreDelete:
    """Test delete operations."""

    def test_delete_single(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        assert len(store) == 5
        store.delete([ids[0]])
        assert len(store) == 4
        assert store.get(ids[0]) is None

    def test_delete_multiple(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        store.delete(ids[:3])
        assert len(store) == 2

    def test_delete_empty_list(self, store):
        deleted = store.delete([])
        assert deleted == 0

    def test_clear(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        assert len(store) == 5
        deleted = store.clear()
        assert deleted == 5
        assert len(store) == 0


class TestChromaStoreContains:
    """Test __contains__ method."""

    def test_contains_existing(self, store):
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        assert ids[0] in store

    def test_contains_nonexistent(self, store):
        assert 999 not in store

    def test_contains_string_id(self, store):
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        assert str(ids[0]) in store


class TestChromaStorePersistence:
    """Test persistence across sessions."""

    def test_data_persists(self):
        from cyllama.rag.chroma_store import ChromaVectorStore
        with tempfile.TemporaryDirectory() as db_path:
            with ChromaVectorStore(dimension=4, db_path=db_path) as store:
                store.add([[1.0, 0.0, 0.0, 0.0]], ["persistent text"])
                assert len(store) == 1

            with ChromaVectorStore(dimension=4, db_path=db_path) as store:
                assert len(store) == 1
                results = store.search([1.0, 0.0, 0.0, 0.0], k=1)
                assert results[0].text == "persistent text"


class TestChromaStoreContextManager:
    """Test context manager protocol."""

    def test_context_manager_closes(self):
        store = _make_ephemeral_store()
        with store:
            store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        with pytest.raises(RuntimeError, match="closed"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])

    def test_close_idempotent(self):
        store = _make_ephemeral_store()
        store.close()
        store.close()  # Should not raise


class TestChromaStoreRepr:
    """Test string representation."""

    def test_repr_open(self, store):
        repr_str = repr(store)
        assert "ChromaVectorStore" in repr_str
        assert "dimension=4" in repr_str

    def test_repr_closed(self):
        store = _make_ephemeral_store()
        store.close()
        repr_str = repr(store)
        assert "closed" in repr_str
