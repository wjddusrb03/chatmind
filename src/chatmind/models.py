"""Data models for chat messages and indices."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from langchain_turboquant.quantizer import TurboQuantizer, CompressedVectors


@dataclass
class ChatMessage:
    """A single chat message."""
    timestamp: datetime
    sender: str
    content: str
    room: str = ""
    platform: str = ""
    message_id: str = ""

    def to_embedding_text(self) -> str:
        """Create text for embedding. Combines sender + content."""
        return f"{self.sender}: {self.content}"

    def time_str(self) -> str:
        """Formatted timestamp string."""
        return self.timestamp.strftime("%Y-%m-%d %H:%M")


@dataclass
class ChatIndex:
    """Indexed and compressed chat message data."""
    messages: List[ChatMessage]
    compressed: CompressedVectors
    quantizer: TurboQuantizer
    model_name: str
    embedding_dim: int
    raw_memory_bytes: int
    compressed_memory_bytes: int
    index_time: float
    platform: str = ""
    rooms: List[str] = field(default_factory=list)
    senders: List[str] = field(default_factory=list)
