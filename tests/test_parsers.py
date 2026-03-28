"""Tests for chat message parsers."""

import os
import json
import tempfile
from datetime import datetime

import pytest

from chatmind.parsers.discord import parse_discord_json, parse_discord
from chatmind.parsers.kakao import parse_kakao
from chatmind.parsers.auto import auto_parse, _detect_platform


SAMPLE_DIR = os.path.dirname(__file__)
SAMPLE_DISCORD = os.path.join(SAMPLE_DIR, "sample_discord.json")


class TestDiscordParser:
    """Tests for Discord JSON parser."""

    def test_parse_discord_json_basic(self):
        msgs = parse_discord_json(SAMPLE_DISCORD)
        assert len(msgs) == 30
        assert msgs[0].sender == "Alex"
        assert msgs[0].platform == "discord"

    def test_parse_discord_channel_name(self):
        msgs = parse_discord_json(SAMPLE_DISCORD)
        assert msgs[0].room == "general"

    def test_parse_discord_content(self):
        msgs = parse_discord_json(SAMPLE_DISCORD)
        assert "Minecraft" in msgs[0].content

    def test_parse_discord_timestamp(self):
        msgs = parse_discord_json(SAMPLE_DISCORD)
        assert msgs[0].timestamp.year == 2024
        assert msgs[0].timestamp.month == 1
        assert msgs[0].timestamp.day == 15

    def test_parse_discord_message_id(self):
        msgs = parse_discord_json(SAMPLE_DISCORD)
        assert msgs[0].message_id == "1001"

    def test_parse_discord_room_override(self):
        msgs = parse_discord_json(SAMPLE_DISCORD, room="custom-room")
        assert msgs[0].room == "custom-room"

    def test_parse_discord_data_package_format(self):
        """Test Discord Data Package list format."""
        data = [
            {"ID": "1", "Timestamp": "2024-03-15T14:30:00+00:00",
             "Contents": "Hello world", "Author": {"username": "testuser"}},
            {"ID": "2", "Timestamp": "2024-03-15T14:31:00+00:00",
             "Contents": "Hi there", "Author": {"username": "user2"}},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 2
        assert msgs[0].sender == "testuser"
        assert msgs[0].content == "Hello world"

    def test_parse_discord_skips_empty_content(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "", "author": {"name": "user"}},
            {"id": "2", "timestamp": "2024-01-01T00:01:00+00:00",
             "content": "hello", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 1

    def test_parse_discord_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"messages": [
                {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
                 "content": "msg1", "author": {"name": "user1"}},
            ], "channel": {"name": "ch1"}}
            fpath = os.path.join(tmpdir, "export.json")
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f)
            msgs = parse_discord(tmpdir)
            assert len(msgs) == 1


class TestKakaoParser:
    """Tests for KakaoTalk parser."""

    def test_parse_kakao_korean(self):
        content = """친구들 채팅방
--------------- 2024년 3월 15일 금요일 ---------------
[김철수] 오후 2:30 강남역 맛집 추천해줘
[이영희] 오후 2:31 스시오마카세 어때?
[김철수] 오후 2:32 좋아 예약해줘
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            msgs = parse_kakao(f.name)
        os.unlink(f.name)
        assert len(msgs) == 3
        assert msgs[0].sender == "김철수"
        assert "맛집" in msgs[0].content
        assert msgs[0].platform == "kakao"

    def test_parse_kakao_pm_time(self):
        content = """test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오후 1:00 hello
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            msgs = parse_kakao(f.name)
        os.unlink(f.name)
        assert msgs[0].timestamp.hour == 13

    def test_parse_kakao_am_time(self):
        content = """test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오전 9:30 good morning
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            msgs = parse_kakao(f.name)
        os.unlink(f.name)
        assert msgs[0].timestamp.hour == 9
        assert msgs[0].timestamp.minute == 30


class TestAutoDetect:
    """Tests for auto format detection."""

    def test_detect_discord_json(self):
        data = {"messages": [{"id": "1", "content": "hi", "author": {"name": "u"}, "timestamp": "2024-01-01T00:00:00"}], "channel": {"name": "t"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            result = _detect_platform(f.name)
        os.unlink(f.name)
        assert result == "discord"

    def test_detect_kakao_txt(self):
        content = """test
--------------- 2024년 3월 15일 금요일 ---------------
[김철수] 오후 2:30 안녕
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            result = _detect_platform(f.name)
        os.unlink(f.name)
        assert result == "kakao"

    def test_auto_parse_discord(self):
        msgs = auto_parse(SAMPLE_DISCORD)
        assert len(msgs) == 30
        assert msgs[0].platform == "discord"

    def test_auto_parse_with_platform_hint(self):
        msgs = auto_parse(SAMPLE_DISCORD, platform="discord")
        assert len(msgs) == 30
