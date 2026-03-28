"""Indexer: Create TurboQuant-compressed embeddings from chat messages."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_turboquant.quantizer import TurboQuantizer

from .models import ChatMessage, ChatIndex


def build_index(
    messages: List[ChatMessage],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    bits: int = 3,
    qjl_dim: Optional[int] = None,
    seed: int = 42,
    batch_size: int = 64,
) -> ChatIndex:
    """Build a TurboQuant-compressed index from chat messages.

    Args:
        messages: List of ChatMessage objects.
        model_name: Sentence transformer model name.
                    Default: multilingual model for Korean+English.
        bits: Quantization bits for TurboQuant (2-4).
        qjl_dim: QJL projection dimension. None for auto.
        seed: Random seed for reproducibility.
        batch_size: Batch size for embedding computation.

    Returns:
        ChatIndex with compressed vectors.
    """
    if not messages:
        raise ValueError("No messages to index.")

    # Step 1: Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(model_name)

    # Step 2: Create embedding texts
    texts = [m.to_embedding_text() for m in messages]

    # Step 3: Compute embeddings
    print(f"Computing embeddings for {len(texts):,} messages...")
    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    embed_time = time.time() - start_time
    print(f"Embeddings computed in {embed_time:.1f}s")

    embedding_dim = embeddings.shape[1]
    raw_memory = embeddings.nbytes

    # Step 4: TurboQuant compression
    print("Compressing with TurboQuant...")
    compress_start = time.time()

    quantizer = TurboQuantizer(dim=embedding_dim, bits=bits, qjl_dim=qjl_dim, seed=seed)
    compressed = quantizer.quantize(embeddings)

    compress_time = time.time() - compress_start
    print(f"Compressed in {compress_time:.1f}s")

    compressed_memory = (
        compressed.indices.nbytes
        + compressed.qjl_bits.nbytes
        + compressed.gammas.nbytes
        + compressed.norms.nbytes
    )

    total_time = time.time() - start_time

    # Collect metadata
    rooms = sorted(set(m.room for m in messages if m.room))
    senders = sorted(set(m.sender for m in messages if m.sender))
    platform = messages[0].platform if messages else ""

    return ChatIndex(
        messages=messages,
        compressed=compressed,
        quantizer=quantizer,
        model_name=model_name,
        embedding_dim=embedding_dim,
        raw_memory_bytes=raw_memory,
        compressed_memory_bytes=compressed_memory,
        index_time=total_time,
        platform=platform,
        rooms=rooms,
        senders=senders,
    )
