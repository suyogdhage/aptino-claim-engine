import os
import tempfile
from typing import List, Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxCrossEncoder:
    """Memory-light cross-encoder built on ONNX Runtime (no PyTorch).

    Loads the tokenizer and an ONNX export of the given cross-encoder model
    from the Hugging Face Hub, and returns sigmoid-free logits per query/doc
    pair in the same shape/ordering as sentence-transformers' CrossEncoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._session = None

    def _load(self) -> None:
        from huggingface_hub import hf_hub_download

        cache_root = os.environ.get("HF_HOME") or os.path.join(
            tempfile.gettempdir(), "hf-hub"
        )
        tokenizer_path = hf_hub_download(
            self.model_name, "tokenizer.json", cache_dir=cache_root
        )
        onnx_path = hf_hub_download(
            self.model_name, "onnx/model.onnx", cache_dir=cache_root
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

    def predict(self, pairs: Sequence[Sequence[str]]) -> np.ndarray:
        if not pairs:
            return np.array([])

        if self._tokenizer is None or self._session is None:
            self._load()

        tokenizer = self._tokenizer
        tokenizer.enable_truncation(max_length=self.max_length)
        encoded = [
            tokenizer.encode(text_a, text_b)
            for text_a, text_b in pairs
        ]

        max_len = max(min(len(e.ids), self.max_length) for e in encoded)
        input_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encoded), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(encoded), max_len), dtype=np.int64)

        for i, e in enumerate(encoded):
            length = min(len(e.ids), max_len)
            ids = e.ids[:length]
            attn = e.attention_mask[:length] if e.attention_mask else [1] * length
            input_ids[i, :length] = ids
            attention_mask[i, :length] = attn
            token_type_ids[i, :length] = 0

        outputs = self._session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        logits = np.asarray(outputs[0])
        return logits.reshape(len(pairs))