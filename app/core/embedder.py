"""Prompt embedding pipeline with CPU thread-pool offloading and L2 normalization."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

# Prefer local cache without hanging on network checks
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]

logger = logging.getLogger("cachesense.embedder")


class PromptEmbedder:
    """
    Asynchronous text embedder wrapping sentence-transformers 'all-MiniLM-L6-v2'.
    Offloads CPU-bound encoding to a dedicated bounded ThreadPoolExecutor
    to prevent event loop stalling and normalizes output vectors to unit length.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_threads: int = 4):
        self.model_name = model_name
        self.max_threads = max_threads
        self._executor = ThreadPoolExecutor(
            max_workers=max_threads,
            thread_name_prefix="embedder-worker",
        )
        self._model: Any | None = None
        self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def load_model(self) -> None:
        """Load and warm up model synchronously (called during startup)."""
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not installed. Embedder running in fallback/mock mode.")
            return

        logger.info(f"Loading embedding model '{self.model_name}' on CPU...")
        self._model = SentenceTransformer(self.model_name, device="cpu")
        # Warm up
        self.warmup()
        logger.info(f"Model '{self.model_name}' successfully loaded and warmed up.")

    def warmup(self) -> None:
        """Run single inference to initialize torch tensors."""
        if self._model is not None:
            _ = self._model.encode(["warmup sentence"], normalize_embeddings=True)

    def _sync_encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            # Fallback deterministic pseudo-embedding for testing when model not present
            vecs = []
            for t in texts:
                np.random.seed(abs(hash(t)) % (2**32))
                v = np.random.randn(self._dimension).astype(np.float32)
                norm = np.linalg.norm(v)
                vecs.append(v / (norm if norm > 0 else 1.0))
            return np.array(vecs, dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    async def embed(self, text: str) -> np.ndarray:
        """
        Embed single prompt string into (1, 384) float32 L2-normalized vector.
        Non-blocking async call offloaded to thread pool.
        """
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(self._executor, self._sync_encode, [text])
        return vectors

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed batch of prompts into (N, 384) float32 L2-normalized vectors.
        Non-blocking async call offloaded to thread pool.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(self._executor, self._sync_encode, texts)
        return vectors

    def shutdown(self) -> None:
        """Shutdown thread pool executor."""
        self._executor.shutdown(wait=False)
