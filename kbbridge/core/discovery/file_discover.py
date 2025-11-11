import logging
from typing import Dict, List, Tuple

import dspy

from kbbridge.core.utils.text_processing_utils import build_file_surrogate_text
from kbbridge.integrations.dify.constants import DifyRetrieverDefaults
from kbbridge.integrations.retriever_base import ChunkHit, FileHit, Retriever

logger = logging.getLogger(__name__)


class FileDiscover(dspy.Module):
    """
    File discovery using backend-agnostic retriever + optional rerankers.

    This module provides a DSPy-style module (FileDiscover) that:
    - retrieves chunks via the configured Retriever (Dify, etc.)
    - optionally re-ranks chunks
    - aggregates to files
    - optionally re-ranks files using a reranking service
    """

    def __init__(
        self,
        retriever: Retriever,
        chunk_rerank_fn=None,
        file_rerank_fn=None,
        top_chunks_per_file: int = 2,
        use_surrogates: bool = True,
    ):
        super().__init__()
        self.retriever = retriever
        self.chunk_rerank_fn = chunk_rerank_fn
        self.file_rerank_fn = file_rerank_fn
        self.top_chunks_per_file = top_chunks_per_file
        self.use_surrogates = use_surrogates

    def _retrieve(self, query: str, method: str, top_k: int, **kw) -> List[ChunkHit]:
        # Use DifyRetrieverDefaults for reranking parameters if not provided
        reranking_provider_name = kw.get(
            "reranking_provider_name",
            DifyRetrieverDefaults.RERANKING_PROVIDER_NAME.value,
        )
        reranking_model_name = kw.get(
            "reranking_model_name",
            DifyRetrieverDefaults.RERANKING_MODEL_NAME.value,
        )

        resp = self.retriever.call(
            query=query,
            method=method,
            top_k=top_k,
            does_rerank=False,
            score_threshold_enabled=False,
            metadata_filter=kw.get("metadata_filter"),
            reranking_provider_name=reranking_provider_name,
            reranking_model_name=reranking_model_name,
            weights=kw.get("weights"),
            score_threshold=kw.get("score_threshold"),
        )
        return self.retriever.normalize_chunks(resp)

    def _rerank_chunks(self, query: str, chunks: List[ChunkHit]) -> List[ChunkHit]:
        if not self.chunk_rerank_fn or not chunks:
            return chunks

        # Use positional indices as stable IDs for this pass
        texts = [c.content for c in chunks]
        meta = [{"index": i, "file_id": c.document_name} for i, c in enumerate(chunks)]

        rr = self.chunk_rerank_fn(query=query, documents=texts, all_docs=meta)
        if not rr or not rr.get("success") or not rr.get("detailed_results"):
            return chunks

        # Expect detailed_results entries to include an 'index' ordering
        order = [
            item.get("index") for item in rr["detailed_results"] if "index" in item
        ]
        seen = set(i for i in order if isinstance(i, int) and 0 <= i < len(chunks))
        reordered = [
            chunks[i] for i in order if isinstance(i, int) and 0 <= i < len(chunks)
        ]
        reordered.extend(chunks[i] for i in range(len(chunks)) if i not in seen)
        return reordered

    def _aggregate(self, chunks: List[ChunkHit]) -> List[FileHit]:
        return self.retriever.group_files(chunks, agg="max")

    def _build_file_docs(
        self, files: List[FileHit], chunks: List[ChunkHit], max_chars: int = 800
    ) -> Tuple[List[str], List[Dict]]:
        # Map file -> top chunks
        per_file: Dict[str, List[ChunkHit]] = {}
        for i, c in enumerate(chunks):
            fid = c.document_name
            lst = per_file.setdefault(fid, [])
            if len(lst) < self.top_chunks_per_file:
                lst.append(c)

        docs: List[str] = []
        meta: List[Dict] = []
        for f in files:
            title = getattr(f, "file_name", None) or getattr(f, "title", "")
            file_id = title
            topc = per_file.get(file_id, [])
            if self.use_surrogates:
                text = build_file_surrogate_text(title, topc, max_chars=max_chars)
            else:
                text = (f"title: {title}\n" + "\n".join(c.content for c in topc))[
                    :max_chars
                ]
            docs.append(text)
            meta.append({"document_name": title, "file_id": file_id})
        return docs, meta

    def _apply_file_rerank(self, files, chunks, query, **kw):
        if not self.file_rerank_fn:
            return sorted(
                files,
                key=lambda f: (f.score, getattr(f, "file_name", "")),
                reverse=True,
            )

        docs, meta = self._build_file_docs(files, chunks)
        rr = self.file_rerank_fn(
            query=query,
            documents=docs,
            all_docs=meta,
            rerank_url=kw.get("rerank_url"),
            model=kw.get("rerank_model"),
            relevance_score_threshold=kw.get("relevance_score_threshold", 0.0),
        )
        if not rr or not rr.get("success") or not rr.get("detailed_results"):
            return sorted(
                files,
                key=lambda f: (f.score, getattr(f, "file_name", "")),
                reverse=True,
            )

        # Sort by relevance_score (desc) with document_name tie-breaker (asc) for stability
        detailed = sorted(
            rr["detailed_results"],
            key=lambda x: (
                -x.get("relevance_score", 0.0),
                (x.get("document", {}) or {}).get("document_name", ""),
            ),
        )
        order = [d["document"]["document_name"] for d in detailed if d.get("document")]
        scores = {
            d["document"]["document_name"]: d.get("relevance_score", 0.0)
            for d in detailed
            if d.get("document")
        }
        by_name = {getattr(f, "file_name", ""): f for f in files}
        reranked: List[FileHit] = []
        for name in order:
            if name in by_name:
                f = by_name[name]
                try:
                    f.score = float(scores.get(name, f.score))
                except Exception as e:
                    logger.debug(f"Failed to convert score for file '{name}': {e}")
                reranked.append(f)
        seen = set(order)
        tail = [f for f in files if getattr(f, "file_name", "") not in seen]
        tail.sort(key=lambda f: (f.score, getattr(f, "file_name", "")), reverse=True)
        return reranked + tail

    def forward(
        self,
        query: str,
        search_method: str = "semantic_search",
        top_k_recall: int = 100,
        top_k_return: int = 10,
        do_chunk_rerank: bool = False,
        do_file_rerank: bool = True,
        **kw,
    ) -> List[FileHit]:
        chunks = self._retrieve(query, search_method, top_k_recall, **kw)
        if do_chunk_rerank:
            chunks = self._rerank_chunks(query, chunks)
        files = self._aggregate(chunks)
        if not files:
            return []
        if do_file_rerank and self.file_rerank_fn:
            files = self._apply_file_rerank(files, chunks, query, **kw)
        return sorted(
            files, key=lambda f: (f.score, getattr(f, "file_name", "")), reverse=True
        )[:top_k_return]
