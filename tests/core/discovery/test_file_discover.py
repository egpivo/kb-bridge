"""
Tests for FileDiscover (DSPy-based file discovery)
"""

from kbbridge.core.discovery.file_discover import FileDiscover
from kbbridge.integrations.retriever_base import ChunkHit, FileHit


class DummyRetriever:
    def __init__(self):
        self._chunks = [
            ChunkHit("alpha", document_name="a.pdf", score=0.7),
            ChunkHit("beta", document_name="b.pdf", score=0.9),
        ]

    def call(self, *, query: str, method: str, top_k: int, **kw):
        return {}

    def normalize_chunks(self, resp):
        return self._chunks

    def group_files(self, chunks, agg: str = "max"):
        # Aggregate by max score
        scores = {}
        for c in chunks:
            scores[c.document_name] = max(scores.get(c.document_name, 0.0), c.score)
        return [FileHit(file_name=k, score=v) for k, v in scores.items()]

    def build_metadata_filter(self, *, document_name: str = ""):
        return None


def test_file_discover_basic():
    r = DummyRetriever()
    d = FileDiscover(retriever=r)
    files = d(
        query="test",
        search_method="semantic_search",
        top_k_recall=10,
        top_k_return=10,
        do_chunk_rerank=False,
        do_file_rerank=False,
    )
    assert isinstance(files, list)
    assert len(files) == 2
    names = [f.file_name for f in files]
    assert "a.pdf" in names and "b.pdf" in names


def test_file_discover_with_rerank():
    r = DummyRetriever()

    def file_rerank_fn(query, documents, all_docs, **kw):
        detailed = [
            {
                "document": {"document_name": all_docs[1]["document_name"]},
                "relevance_score": 0.9,
            },
            {
                "document": {"document_name": all_docs[0]["document_name"]},
                "relevance_score": 0.8,
            },
        ]
        return {"success": True, "detailed_results": detailed}

    d = FileDiscover(retriever=r, file_rerank_fn=file_rerank_fn)
    files = d(
        query="test",
        search_method="semantic_search",
        top_k_recall=10,
        top_k_return=10,
        do_chunk_rerank=False,
        do_file_rerank=True,
    )
    assert files[0].file_name == "b.pdf"
