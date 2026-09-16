import pickle
import os
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class BM25Index:
    def __init__(self, index_path: str = "./chroma_db/bm25_index.pkl"):
        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]
        self.metadatas = [
            c.get("metadata") or {
                "page": c.get("page", 0),
                "section": c.get("section", ""),
                "subsection": c.get("subsection", ""),
                "clause_id": c.get("clause_id", ""),
                "token_count": c.get("token_count", 0)
            }
            for c in chunks
        ]

        tokenized_corpus = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.save()

    def _tokenize(self, text: str) -> List[str]:
        return word_tokenize(text.lower())

    def save(self) -> None:
        data = {
            "chunk_ids": self.chunk_ids,
            "texts": self.texts,
            "metadatas": self.metadatas
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        if not os.path.exists(self.index_path):
            return False
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.texts = data["texts"]
        self.metadatas = data["metadatas"]
        tokenized_corpus = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return True

    def query(self, query_text: str, n_results: int = 20) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "chunk_id": self.chunk_ids[idx],
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "sparse_score": float(scores[idx])
                })
        return results