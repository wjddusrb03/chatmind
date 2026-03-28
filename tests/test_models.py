"""Tests for data models."""

from datetime import datetime
from chatmind.models import ChatMessage, ChatIndex


class TestChatMessage:

    def test_create_message(self):
        msg = ChatMessage(
            timestamp=datetime(2024, 3, 15, 14, 30),
            sender="Alex",
            content="Hello world",
            room="general",
            platform="discord",
        )
        assert msg.sender == "Alex"
        assert msg.content == "Hello world"
        assert msg.room == "general"

    def test_to_embedding_text(self):
        msg = ChatMessage(
            timestamp=datetime(2024, 1, 1),
            sender="Kim",
            content="Let's go eat sushi",
        )
        text = msg.to_embedding_text()
        assert "Kim" in text
        assert "sushi" in text

    def test_time_str(self):
        msg = ChatMessage(
            timestamp=datetime(2024, 3, 15, 14, 30),
            sender="user",
            content="test",
        )
        assert msg.time_str() == "2024-03-15 14:30"

    def test_default_fields(self):
        msg = ChatMessage(
            timestamp=datetime(2024, 1, 1),
            sender="user",
            content="test",
        )
        assert msg.room == ""
        assert msg.platform == ""
        assert msg.message_id == ""
