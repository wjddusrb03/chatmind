"""Tests for semantic search module."""

import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

from chatmind.models import ChatMessage, ChatIndex
from chatmind.searcher import search, SearchResult
from langchain_turboquant.quantizer import TurboQuantizer


DIM = 384
SEED = 42
rng = np.random.RandomState(SEED)

# Topic vectors
TOPICS = {
    "food": None,
    "gaming": None,
    "travel": None,
    "study": None,
    "music": None,
}
for t in TOPICS:
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    TOPICS[t] = v


def _make_index_with_topics():
    """Create a test index with topic-grouped messages."""
    messages = []
    vectors = []
    noise_rng = np.random.RandomState(100)

    topic_msgs = {
        "food": [
            ("Park", "I found an amazing sushi restaurant near Gangnam"),
            ("Kim", "Try the pasta place, their carbonara is amazing"),
            ("Lee", "The new ramen shop has incredible tonkotsu broth"),
            ("Alex", "Best Korean BBQ place with wagyu beef"),
        ],
        "gaming": [
            ("Alex", "Anyone want to play Minecraft tonight?"),
            ("Lee", "Just got RTX 4090, ray tracing is incredible"),
            ("Kim", "The new Zelda game is absolutely fantastic"),
            ("Park", "Server lag is terrible, need more RAM"),
        ],
        "travel": [
            ("Alex", "Let's plan a trip to Jeju Island"),
            ("Park", "We can rent a car and visit Seongsan"),
            ("Lee", "Booking Airbnb near Hallasan mountain"),
            ("Kim", "Tokyo trip next summer sounds great"),
        ],
        "study": [
            ("Kim", "Help with Python programming assignment"),
            ("Lee", "Data structures and algorithms practice"),
            ("Alex", "Calculus homework is due tomorrow"),
            ("Park", "Machine learning exam next week"),
        ],
        "music": [
            ("Park", "BTS concert tickets are sold out"),
            ("Kim", "Learning acoustic guitar basics"),
            ("Lee", "New album release from Blackpink"),
            ("Alex", "Piano practice for 2 hours daily"),
        ],
    }

    for topic, msg_list in topic_msgs.items():
        base = TOPICS[topic]
        for sender, content in msg_list:
            messages.append(ChatMessage(
                timestamp=datetime(2024, 1, 15, 10, 0),
                sender=sender,
                content=content,
                room="general",
                platform="discord",
            ))
            noise = noise_rng.randn(DIM).astype(np.float32) * 0.15
            vec = base + noise
            vec /= np.linalg.norm(vec)
            vectors.append(vec)

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
        rooms=["general"],
        senders=["Alex", "Kim", "Lee", "Park"],
    )


def _mock_model(topic):
    """Create a mock model that returns topic-biased vectors."""
    model = MagicMock()
    query_rng = np.random.RandomState(hash(topic) % (2**31))
    noise = query_rng.randn(DIM).astype(np.float32) * 0.1
    vec = TOPICS[topic] + noise
    vec /= np.linalg.norm(vec)
    model.encode = MagicMock(return_value=np.array([vec]))
    return model


class TestSearch:

    def test_food_search(self):
        index = _make_index_with_topics()
        model = _mock_model("food")
        results = search("restaurant recommendation", index, k=5, model=model)
        assert len(results) > 0
        # Top result should be food-related
        food_msgs = ["sushi", "pasta", "ramen", "BBQ"]
        assert any(word in results[0].message.content for word in food_msgs)

    def test_gaming_search(self):
        index = _make_index_with_topics()
        model = _mock_model("gaming")
        results = search("playing games", index, k=5, model=model)
        assert len(results) > 0
        gaming_msgs = ["Minecraft", "RTX", "Zelda", "Server lag"]
        assert any(word in results[0].message.content for word in gaming_msgs)

    def test_travel_search(self):
        index = _make_index_with_topics()
        model = _mock_model("travel")
        results = search("vacation planning", index, k=5, model=model)
        assert len(results) > 0
        travel_msgs = ["Jeju", "Seongsan", "Hallasan", "Tokyo"]
        assert any(word in results[0].message.content for word in travel_msgs)

    def test_study_search(self):
        index = _make_index_with_topics()
        model = _mock_model("study")
        results = search("homework help", index, k=5, model=model)
        assert len(results) > 0
        study_msgs = ["Python", "algorithms", "Calculus", "Machine learning"]
        assert any(word in results[0].message.content for word in study_msgs)

    def test_music_search(self):
        index = _make_index_with_topics()
        model = _mock_model("music")
        results = search("concert tickets", index, k=5, model=model)
        assert len(results) > 0
        music_msgs = ["BTS", "guitar", "Blackpink", "Piano"]
        assert any(word in results[0].message.content for word in music_msgs)

    def test_filter_by_sender(self):
        index = _make_index_with_topics()
        model = _mock_model("food")
        results = search("food", index, k=20, model=model, sender="Park")
        for r in results:
            assert "Park" in r.message.sender

    def test_filter_by_room(self):
        index = _make_index_with_topics()
        model = _mock_model("food")
        results = search("food", index, k=5, model=model, room="general")
        for r in results:
            assert "general" in r.message.room

    def test_filter_by_date(self):
        index = _make_index_with_topics()
        model = _mock_model("food")
        results = search("food", index, k=5, model=model,
                         after=datetime(2024, 1, 14),
                         before=datetime(2024, 1, 16))
        assert len(results) > 0

    def test_empty_index(self):
        from langchain_turboquant.quantizer import TurboQuantizer
        quantizer = TurboQuantizer(dim=DIM, bits=3, seed=SEED)
        vecs = np.zeros((0, DIM), dtype=np.float32)
        idx = ChatIndex(
            messages=[], compressed=quantizer.quantize(np.ones((1, DIM), dtype=np.float32)),
            quantizer=quantizer, model_name="test", embedding_dim=DIM,
            raw_memory_bytes=0, compressed_memory_bytes=0, index_time=0,
        )
        idx.messages = []
        results = search("test", idx, model=_mock_model("food"))
        assert results == []

    def test_result_has_score(self):
        index = _make_index_with_topics()
        model = _mock_model("food")
        results = search("sushi", index, k=1, model=model)
        assert results[0].score > 0.2
        assert results[0].rank == 1
