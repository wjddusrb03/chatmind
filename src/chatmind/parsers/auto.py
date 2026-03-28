"""Auto-detect chat export format and parse."""

from __future__ import annotations

import os
from typing import List

from ..models import ChatMessage
from .discord import parse_discord
from .kakao import parse_kakao


def _detect_platform(filepath: str) -> str:
    """Detect the platform from file format.

    Args:
        filepath: Path to the chat export file/directory.

    Returns:
        Platform name: "discord", "kakao", or "unknown".
    """
    if os.path.isdir(filepath):
        # Check for Discord JSON/CSV files in directory
        for fname in os.listdir(filepath):
            if fname.endswith(".json"):
                return "discord"
        return "unknown"

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".json":
        # JSON = Discord (or Telegram/Slack in future)
        import json
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # DiscordChatExporter format
            if isinstance(data, dict) and "messages" in data:
                return "discord"
            # Discord data package or generic
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if "Contents" in first or "content" in first:
                    return "discord"
        except Exception:
            pass
        return "discord"  # Default JSON = Discord

    if ext == ".csv":
        return "discord"

    if ext == ".txt":
        # Check for KakaoTalk markers
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i > 20:
                        break
                    if "년" in line and "월" in line and "일" in line:
                        return "kakao"
                    if "[" in line and ("오전" in line or "오후" in line):
                        return "kakao"
        except Exception:
            pass
        return "kakao"  # Default txt = KakaoTalk

    return "unknown"


def auto_parse(filepath: str, platform: str = "", room: str = "") -> List[ChatMessage]:
    """Auto-detect format and parse chat export.

    Args:
        filepath: Path to the chat export file or directory.
        platform: Platform hint ("discord", "kakao"). Auto-detected if empty.
        room: Optional room/channel name override.

    Returns:
        List of ChatMessage objects.

    Raises:
        ValueError: If the platform cannot be detected.
    """
    if not platform:
        platform = _detect_platform(filepath)

    if platform == "discord":
        return parse_discord(filepath, room=room)
    elif platform == "kakao":
        return parse_kakao(filepath, room=room)
    else:
        raise ValueError(
            f"Cannot detect chat platform for '{filepath}'. "
            f"Use --platform discord or --platform kakao."
        )
