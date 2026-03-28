"""Searcher: Semantic search over compressed chat index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import ChatIndex, ChatMessage


@dataclass
class SearchResult:
    """A single search result."""
    rank: int
    score: float
    message: ChatMessage


def search(
    query: str,
    index: ChatIndex,
    k: int = 5,
    model: Optional[SentenceTransformer] = None,
    sender: Optional[str] = None,
    room: Optional[str] = None,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
) -> List[SearchResult]:
    """Search for messages semantically similar to the query.

    Args:
        query: Natural language search query.
        index: The ChatIndex to search over.
        k: Number of results to return.
        model: Pre-loaded SentenceTransformer model.
        sender: Filter by sender name (case-insensitive partial match).
        room: Filter by room name (case-insensitive partial match).
        after: Only include messages after this datetime.
        before: Only include messages before this datetime.

    Returns:
        List of SearchResult objects, sorted by relevance.
    """
    if not index.messages:
        return []

    # Load model if not provided
    if model is None:
        model = SentenceTransformer(index.model_name)

    # Embed the query
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )
    query_vec = np.array(query_embedding[0], dtype=np.float32)

    # Use TurboQuant asymmetric scoring
    scores = index.quantizer.cosine_scores(query_vec, index.compressed)

    # Apply filters - set filtered-out scores to -inf
    for i, msg in enumerate(index.messages):
        if sender and sender.lower() not in msg.sender.lower():
            scores[i] = -np.inf
        if room and room.lower() not in msg.room.lower():
            scores[i] = -np.inf
        if after and msg.timestamp < after:
            scores[i] = -np.inf
        if before and msg.timestamp > before:
            scores[i] = -np.inf

    # Get top-k indices
    k = min(k, len(index.messages))
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        if scores[idx] == -np.inf:
            break
        results.append(SearchResult(
            rank=rank,
            score=float(scores[idx]),
            message=index.messages[idx],
        ))

    return results
