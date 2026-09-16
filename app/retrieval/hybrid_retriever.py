import hashlib
import json
import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from app.retrieval.onnx_rerank import OnnxCrossEncoder
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.bm25_index import BM25Index
from app.models.state import Evidence, InvestigationDimension


INDEX_MANIFEST = "index_manifest.json"


def _policy_sha256(pdf_path: str) -> str:
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as policy_file:
        for block in iter(lambda: policy_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_status(pdf_path: str, persist_dir: str) -> tuple[bool, str]:
    """Return whether both retrieval stores match the supplied policy PDF."""
    if not os.path.isfile(pdf_path):
        return False, f"Policy PDF not found: {pdf_path}"
    manifest_path = os.path.join(persist_dir, INDEX_MANIFEST)
    bm25_path = os.path.join(persist_dir, "bm25_index.pkl")
    if not os.path.isfile(manifest_path) or not os.path.isfile(bm25_path):
        return False, "Index manifest or BM25 index is missing"
    try:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("policy_sha256") != _policy_sha256(pdf_path):
            return False, "Policy PDF changed since the index was built"
        chroma = ChromaStore(persist_dir=persist_dir)
        if chroma.count() != manifest.get("chunk_count", 0) or chroma.count() == 0:
            return False, "Chroma index is missing chunks or does not match the manifest"
        bm25 = BM25Index(index_path=bm25_path)
        if not bm25.load() or len(bm25.chunk_ids) != manifest.get("chunk_count", 0):
            return False, "BM25 index is empty or does not match the manifest"
    except Exception as exc:
        return False, f"Unable to validate retrieval index: {exc}"
    return True, "Policy index and both retrieval stores are ready"


class HybridRetriever:
    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_index: BM25Index,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        fusion_k: int = 60,
        rerank_top_k: int = 12,
        final_top_k: int = 8
    ):
        self.chroma = chroma_store
        self.bm25 = bm25_index
        self.reranker_model = reranker_model
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fusion_k = fusion_k
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k
        self._reranker = None

    @property
    def reranker(self) -> OnnxCrossEncoder:
        if self._reranker is None:
            self._reranker = OnnxCrossEncoder(model_name=self.reranker_model)
        return self._reranker

    def rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        ranked_dense = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}
        ranked_sparse = {r["chunk_id"]: i + 1 for i, r in enumerate(sparse_results)}

        all_chunks = {}
        for r in dense_results + sparse_results:
            cid = r["chunk_id"]
            if cid not in all_chunks:
                all_chunks[cid] = r

        for cid, chunk in all_chunks.items():
            dense_rank = ranked_dense.get(cid, len(dense_results) + 1)
            sparse_rank = ranked_sparse.get(cid, len(sparse_results) + 1)
            chunk["rrf_score"] = (
                1.0 / (dense_rank + self.fusion_k) +
                1.0 / (sparse_rank + self.fusion_k)
            )

        fused = sorted(all_chunks.values(), key=lambda x: x["rrf_score"], reverse=True)
        return fused[:self.rerank_top_k]

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        documents = [chunk["text"] for chunk in chunks]
        reranked_entries = self.reranker.predict([(query, doc) for doc in documents])

        for chunk, score in zip(chunks, reranked_entries):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:self.final_top_k]

    def retrieve_for_dimension(
        self,
        query: str,
        dimension: InvestigationDimension
    ) -> List[Evidence]:
        dense_results = self.chroma.query(query, n_results=self.dense_top_k)
        sparse_results = self.bm25.query(query, n_results=self.sparse_top_k)

        fused = self.rrf_fusion(dense_results, sparse_results)
        reranked = self.rerank(query, fused)

        evidence_list = []
        for chunk in reranked:
            meta = chunk.get("metadata", {})
            evidence_list.append(Evidence(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                page=meta.get("page", 0),
                section=meta.get("section", ""),
                subsection=meta.get("subsection"),
                clause_id=meta.get("clause_id"),
                dense_score=chunk.get("dense_score", 0.0),
                sparse_score=chunk.get("sparse_score", 0.0),
                rerank_score=chunk.get("rerank_score", 0.0),
                relevance=chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
                dimension=dimension
            ))

        return evidence_list

    def retrieve(
        self,
        queries: Dict[InvestigationDimension, str]
    ) -> Dict[InvestigationDimension, List[Evidence]]:
        results = {}
        for dimension, query in queries.items():
            results[dimension] = self.retrieve_for_dimension(query, dimension)
        return results


def build_indices(pdf_path: str, persist_dir: str = "./chroma_db") -> tuple:
    from app.ingestion.policy_ingest import ingest_policy

    chunks = ingest_policy(pdf_path)
    if not chunks:
        raise RuntimeError("Policy ingestion produced no chunks")

    parent_dir = os.path.dirname(os.path.abspath(persist_dir))
    os.makedirs(parent_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix="claim-index-", dir=parent_dir)
    chroma = ChromaStore(persist_dir=staging_dir)
    chroma_chunks = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "page": c.page,
            "section": c.section,
            "subsection": c.subsection,
            "clause_id": c.clause_id,
            "token_count": c.token_count
        }
        for c in chunks
    ]
    chroma.add_chunks(chroma_chunks)

    bm25 = BM25Index(index_path=os.path.join(staging_dir, "bm25_index.pkl"))
    bm25.build(chroma_chunks)

    manifest = {
        "policy_sha256": _policy_sha256(pdf_path),
        "chunk_count": len(chroma_chunks),
        "embedding_model": chroma.embedding_model_name,
    }
    with open(os.path.join(staging_dir, INDEX_MANIFEST), "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    chroma.close()

    backup_dir = f"{persist_dir}.backup"
    try:
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        if os.path.exists(persist_dir):
            os.replace(persist_dir, backup_dir)
        os.replace(staging_dir, persist_dir)
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
    except Exception:
        if not os.path.exists(persist_dir) and os.path.exists(backup_dir):
            os.replace(backup_dir, persist_dir)
        raise

    chroma = ChromaStore(persist_dir=persist_dir)
    bm25 = BM25Index(index_path=os.path.join(persist_dir, "bm25_index.pkl"))
    bm25.load()

    return chroma, bm25
