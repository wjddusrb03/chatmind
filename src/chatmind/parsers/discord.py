"""Discord chat export parser.

Supports two formats:
1. Discord Data Package (Settings > Privacy > Request All Data) - JSON
2. DiscordChatExporter (https://github.com/Tyrrrz/DiscordChatExporter) - JSON

Discord Data Package format (messages/c{channel_id}/messages.csv or .json):
  [{"ID": "...", "Timestamp": "2024-03-15T14:30:00+00:00",
    "Contents": "hello", "Author": {"username": "user123"}}]

DiscordChatExporter format:
  {"messages": [{"id": "...", "timestamp": "2024-03-15T14:30:00+00:00",
    "content": "hello", "author": {"name": "user123"}}]}
"""

from __future__ import annotations

import json
import csv
import os
from datetime import datetime
from typing import List, Optional

from ..models import ChatMessage


def _parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse various Discord timestamp formats.

    Returns None if parsing fails (caller should skip the message).
    """
    if not ts:
        return None
    # Strip timezone info for parsing
    clean = ts.split("+")[0].split("Z")[0]
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    return None


def parse_discord_json(filepath: str, room: str = "") -> List[ChatMessage]:
    """Parse a Discord JSON export file.

    Args:
        filepath: Path to the JSON file.
        room: Channel/room name override.

    Returns:
        List of ChatMessage objects.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = []

    # DiscordChatExporter format: {"messages": [...], "channel": {...}}
    if isinstance(data, dict) and "messages" in data:
        channel_name = room or data.get("channel", {}).get("name", "unknown")
        for msg in data["messages"]:
            content = msg.get("content", "").strip()
            if not content:
                continue

            author = msg.get("author", {})
            sender = author.get("name", "") or author.get("nickname", "") or author.get("username", "Unknown")

            ts = _parse_timestamp(msg.get("timestamp", ""))
            if ts is None:
                continue  # skip messages with unparseable timestamps

            messages.append(ChatMessage(
                timestamp=ts,
                sender=sender,
                content=content,
                room=channel_name,
                platform="discord",
                message_id=str(msg.get("id", "")),
            ))

    # Discord Data Package format: [{...}, {...}]
    elif isinstance(data, list):
        for msg in data:
            content = msg.get("Contents", "").strip() or msg.get("content", "").strip()
            if not content:
                continue

            author = msg.get("Author", msg.get("author", {}))
            if isinstance(author, dict):
                sender = author.get("username", "") or author.get("name", "Unknown")
            else:
                sender = str(author)

            ts = _parse_timestamp(msg.get("Timestamp", msg.get("timestamp", "")))
            if ts is None:
                continue

            messages.append(ChatMessage(
                timestamp=ts,
                sender=sender,
                content=content,
                room=room or "unknown",
                platform="discord",
                message_id=str(msg.get("ID", msg.get("id", ""))),
            ))

    return messages


def parse_discord_csv(filepath: str, room: str = "") -> List[ChatMessage]:
    """Parse a Discord CSV export (DiscordChatExporter CSV format).

    Expected columns: AuthorID, Author, Date, Content, Attachments, Reactions
    """
    messages = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get("Content", "").strip()
            if not content:
                continue

            sender = row.get("Author", "Unknown")

            ts = _parse_timestamp(row.get("Date", ""))
            if ts is None:
                continue

            messages.append(ChatMessage(
                timestamp=ts,
                sender=sender,
                content=content,
                room=room or "unknown",
                platform="discord",
                message_id=row.get("AuthorID", ""),
            ))

    return messages


def parse_discord(filepath: str, room: str = "") -> List[ChatMessage]:
    """Auto-detect and parse Discord export file (JSON or CSV).

    Args:
        filepath: Path to the export file or directory.
        room: Optional room/channel name override.

    Returns:
        List of ChatMessage objects.
    """
    # If directory, scan for all JSON/CSV files
    if os.path.isdir(filepath):
        all_messages = []
        for root, dirs, files in os.walk(filepath):
            for fname in files:
                fpath = os.path.join(root, fname)
                if fname.endswith(".json"):
                    all_messages.extend(parse_discord_json(fpath, room=room))
                elif fname.endswith(".csv"):
                    all_messages.extend(parse_discord_csv(fpath, room=room))
        # Sort by timestamp
        all_messages.sort(key=lambda m: m.timestamp)
        return all_messages

    # Single file
    if filepath.endswith(".csv"):
        return parse_discord_csv(filepath, room=room)
    else:
        return parse_discord_json(filepath, room=room)
