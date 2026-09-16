import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from fastembed import TextEmbedding
import os


class ChromaStore:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "policy_chunks",
        embedding_model: str = "BAAI/bge-small-en-v1.5"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._client = None
        self._collection = None
        self._embedder = None

    @property
    def client(self):
        if self._client is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @property
    def embedder(self) -> TextEmbedding:
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name=self.embedding_model_name)
        return self._embedder

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "page": c["page"] or 0,
                "section": c["section"] or "",
                "subsection": c.get("subsection") or "",
                "clause_id": c.get("clause_id") or "",
                "token_count": c.get("token_count") or 0
            }
            for c in chunks
        ]

        embeddings = [emb.tolist() for emb in self.embedder.embed(texts)]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def query(
        self,
        query_text: str,
        n_results: int = 20,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        query_embedding = list(self.embedder.embed([query_text]))[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "dense_score": 1.0 - results["distances"][0][i]
                })
        return chunks

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        results = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._collection = None
        self._client = None

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._collection = None