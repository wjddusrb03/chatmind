"""Tests for storage module."""

import os
import tempfile
from datetime import datetime

import pytest
import numpy as np

from chatmind.models import ChatMessage, ChatIndex
from chatmind.storage import save_index, load_index, index_exists, get_storage_path


def _make_dummy_index():
    """Create a minimal ChatIndex for testing."""
    from langchain_turboquant.quantizer import TurboQuantizer

    messages = [
        ChatMessage(timestamp=datetime(2024, 1, 1), sender="user1", content="hello"),
        ChatMessage(timestamp=datetime(2024, 1, 2), sender="user2", content="world"),
    ]

    dim = 384
    quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
    vecs = np.random.randn(2, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    compressed = quantizer.quantize(vecs)

    return ChatIndex(
        messages=messages,
        compressed=compressed,
        quantizer=quantizer,
        model_name="test-model",
        embedding_dim=dim,
        raw_memory_bytes=vecs.nbytes,
        compressed_memory_bytes=100,
        index_time=0.1,
        platform="discord",
        rooms=["general"],
        senders=["user1", "user2"],
    )


class TestStorage:

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = _make_dummy_index()
            path = save_index(idx, base_path=tmpdir)
            assert os.path.exists(path)

            loaded = load_index(base_path=tmpdir)
            assert len(loaded.messages) == 2
            assert loaded.model_name == "test-model"

    def test_index_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not index_exists(base_path=tmpdir)
            save_index(_make_dummy_index(), base_path=tmpdir)
            assert index_exists(base_path=tmpdir)

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_index(base_path=tmpdir)

    def test_get_storage_path(self):
        path = get_storage_path("/tmp/test")
        assert ".chatmind" in path
        assert "index.pkl" in path

    def test_save_custom_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = _make_dummy_index()
            custom = os.path.join(tmpdir, "custom.pkl")
            path = save_index(idx, path=custom)
            assert path == custom
            assert os.path.exists(custom)

    def test_roundtrip_preserves_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = _make_dummy_index()
            save_index(idx, base_path=tmpdir)
            loaded = load_index(base_path=tmpdir)

            assert loaded.platform == "discord"
            assert loaded.rooms == ["general"]
            assert loaded.senders == ["user1", "user2"]
            assert loaded.embedding_dim == 384
