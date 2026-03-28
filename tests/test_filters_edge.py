"""Tests for search filters and edge cases.

24 tests covering: sender filter, room filter, date filter, edge cases.
"""

import os
import json
import tempfile
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

from chatmind.models import ChatMessage, ChatIndex
from chatmind.searcher import search
from chatmind.parsers.discord import parse_discord_json
from chatmind.parsers.kakao import parse_kakao
from chatmind.display import display_search_results, display_stats, display_rooms, display_people
from langchain_turboquant.quantizer import TurboQuantizer

DIM = 384
SEED = 42


def _build_multiroom_index():
    """Build index with multiple rooms, senders, and dates."""
    rng = np.random.RandomState(500)
    messages = [
        ChatMessage(datetime(2024, 1, 10), "Alice", "Pizza is my favorite food", "food-chat", "discord"),
        ChatMessage(datetime(2024, 1, 11), "Bob", "I love hamburgers and fries", "food-chat", "discord"),
        ChatMessage(datetime(2024, 2, 10), "Alice", "Python programming is fun", "tech-chat", "discord"),
        ChatMessage(datetime(2024, 2, 11), "Charlie", "JavaScript framework comparison", "tech-chat", "discord"),
        ChatMessage(datetime(2024, 3, 10), "Bob", "Let's go hiking this weekend", "outdoor-chat", "discord"),
        ChatMessage(datetime(2024, 3, 11), "Charlie", "Mountain climbing is exciting", "outdoor-chat", "discord"),
        ChatMessage(datetime(2024, 4, 10), "Alice", "New cat video is hilarious", "random", "discord"),
        ChatMessage(datetime(2024, 4, 11), "Bob", "Puppies are the cutest animals", "random", "discord"),
    ]

    vectors = []
    for _ in messages:
        v = rng.randn(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        vectors.append(v)

    embeddings = np.array(vectors, dtype=np.float32)
    quantizer = TurboQuantizer(dim=DIM, bits=3, seed=SEED)
    compressed = quantizer.quantize(embeddings)

    return ChatIndex(
        messages=messages,
        compressed=compressed,
        quantizer=quantizer,
        model_name="test",
        embedding_dim=DIM,
        raw_memory_bytes=embeddings.nbytes,
        compressed_memory_bytes=100,
        index_time=0.1,
        platform="discord",
        rooms=["food-chat", "outdoor-chat", "random", "tech-chat"],
        senders=["Alice", "Bob", "Charlie"],
    )


def _any_mock():
    model = MagicMock()
    rng = np.random.RandomState(999)
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    model.encode = MagicMock(return_value=np.array([v]))
    return model


INDEX = _build_multiroom_index()


# ===== SENDER FILTER (4) =====
class TestSenderFilter:
    def test_filter_alice_only(self):
        r = search("anything", INDEX, k=10, model=_any_mock(), sender="Alice")
        for x in r:
            assert x.message.sender == "Alice"

    def test_filter_bob_only(self):
        r = search("anything", INDEX, k=10, model=_any_mock(), sender="Bob")
        for x in r:
            assert x.message.sender == "Bob"

    def test_filter_charlie_only(self):
        r = search("anything", INDEX, k=10, model=_any_mock(), sender="Charlie")
        for x in r:
            assert x.message.sender == "Charlie"

    def test_filter_case_insensitive(self):
        r = search("anything", INDEX, k=10, model=_any_mock(), sender="alice")
        for x in r:
            assert x.message.sender == "Alice"


# ===== ROOM FILTER (4) =====
class TestRoomFilter:
    def test_filter_food_chat(self):
        r = search("anything", INDEX, k=10, model=_any_mock(), room="food-chat")
        for x in r:
            assert x.message.room == "food-chat"

    def test_filter_tech_chat(self):
        r = search("anything", INDEX, k=10, model=_any_mock(), room="tech-chat")
        for x in r:
            assert x.message.room == "tech-chat"

    def test_filter_partial_match(self):
        r = search("anything", INDEX, k=10, model=_any_mock(), room="food")
        for x in r:
            assert "food" in x.message.room

    def test_filter_nonexistent_room(self):
        r = search("anything", INDEX, k=10, model=_any_mock(), room="nonexistent")
        assert len(r) == 0


# ===== DATE FILTER (6) =====
class TestDateFilter:
    def test_after_february(self):
        r = search("anything", INDEX, k=10, model=_any_mock(),
                    after=datetime(2024, 2, 1))
        for x in r:
            assert x.message.timestamp >= datetime(2024, 2, 1)

    def test_before_march(self):
        r = search("anything", INDEX, k=10, model=_any_mock(),
                    before=datetime(2024, 3, 1))
        for x in r:
            assert x.message.timestamp < datetime(2024, 3, 1)

    def test_date_range(self):
        r = search("anything", INDEX, k=10, model=_any_mock(),
                    after=datetime(2024, 2, 1), before=datetime(2024, 3, 1))
        for x in r:
            assert datetime(2024, 2, 1) <= x.message.timestamp < datetime(2024, 3, 1)

    def test_combined_sender_date(self):
        r = search("anything", INDEX, k=10, model=_any_mock(),
                    sender="Alice", after=datetime(2024, 2, 1))
        for x in r:
            assert x.message.sender == "Alice"
            assert x.message.timestamp >= datetime(2024, 2, 1)

    def test_combined_room_date(self):
        r = search("anything", INDEX, k=10, model=_any_mock(),
                    room="food", before=datetime(2024, 2, 1))
        for x in r:
            assert "food" in x.message.room
            assert x.message.timestamp < datetime(2024, 2, 1)

    def test_all_filters_combined(self):
        r = search("anything", INDEX, k=10, model=_any_mock(),
                    sender="Alice", room="tech",
                    after=datetime(2024, 1, 1), before=datetime(2024, 12, 31))
        for x in r:
            assert x.message.sender == "Alice"
            assert "tech" in x.message.room


# ===== EDGE CASES (6) =====
class TestEdgeCases:
    def test_k_larger_than_messages(self):
        r = search("anything", INDEX, k=100, model=_any_mock())
        assert len(r) <= len(INDEX.messages)

    def test_k_one(self):
        r = search("anything", INDEX, k=1, model=_any_mock())
        assert len(r) == 1

    def test_results_sorted_by_score(self):
        r = search("anything", INDEX, k=5, model=_any_mock())
        scores = [x.score for x in r]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_sequential(self):
        r = search("anything", INDEX, k=5, model=_any_mock())
        for i, x in enumerate(r):
            assert x.rank == i + 1

    def test_discord_empty_json(self):
        data = {"messages": [], "channel": {"name": "empty"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 0

    def test_kakao_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("")
            f.flush()
            msgs = parse_kakao(f.name)
        os.unlink(f.name)
        assert len(msgs) == 0


# ===== DISPLAY TESTS (4) =====
class TestDisplay:
    def test_display_results_no_crash(self, capsys):
        rng = np.random.RandomState(42)
        from chatmind.searcher import SearchResult
        results = [
            SearchResult(rank=1, score=0.95, message=INDEX.messages[0]),
            SearchResult(rank=2, score=0.50, message=INDEX.messages[1]),
            SearchResult(rank=3, score=0.20, message=INDEX.messages[2]),
        ]
        display_search_results(results, "test query")
        out = capsys.readouterr().out
        assert "[HIGH]" in out
        assert "[LOW]" in out

    def test_display_stats_no_crash(self, capsys):
        display_stats(INDEX)
        out = capsys.readouterr().out
        assert "ChatMind" in out
        assert "8" in out  # 8 messages

    def test_display_rooms_no_crash(self, capsys):
        display_rooms(INDEX)
        out = capsys.readouterr().out
        assert "food-chat" in out

    def test_display_people_no_crash(self, capsys):
        display_people(INDEX)
        out = capsys.readouterr().out
        assert "Alice" in out
        assert "Bob" in out
