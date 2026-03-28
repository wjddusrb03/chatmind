"""KakaoTalk chat export parser.

KakaoTalk export format (Korean):
  --------------- 2024년 3월 15일 금요일 ---------------
  [김철수] 오후 2:30 강남역 근처 맛집 추천해줘
  [이영희] 오후 2:31 스시오마카세 어때?

KakaoTalk export format (English):
  --------------- Friday, March 15, 2024 ---------------
  [John] 2:30 PM Hey, any restaurant recommendations?
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

from ..models import ChatMessage


# Korean date header
DATE_PATTERN_KO = re.compile(
    r'-+\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*.+\s*-+'
)

# English date header
DATE_PATTERN_EN = re.compile(
    r'-+\s*\w+,\s*(\w+)\s+(\d{1,2}),\s*(\d{4})\s*-+'
)

# Korean message: [sender] 오전/오후 H:MM content
MSG_PATTERN_KO = re.compile(
    r'\[(.+?)\]\s*(오전|오후)\s*(\d{1,2}):(\d{2})\s+(.*)'
)

# English message: [sender] H:MM AM/PM content
MSG_PATTERN_EN = re.compile(
    r'\[(.+?)\]\s*(\d{1,2}):(\d{2})\s*(AM|PM)\s+(.*)'
)

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def parse_kakao(filepath: str, room: str = "") -> List[ChatMessage]:
    """Parse a KakaoTalk chat export file.

    Args:
        filepath: Path to the .txt export file.
        room: Optional room/chat name override.

    Returns:
        List of ChatMessage objects.
    """
    messages = []
    current_date = None

    # Auto-detect room name from first line if not provided
    if not room:
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            # KakaoTalk exports start with room name
            if "," not in first_line and "-" not in first_line:
                room = first_line

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Try Korean date header
            m = DATE_PATTERN_KO.match(line)
            if m:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                current_date = (year, month, day)
                continue

            # Try English date header
            m = DATE_PATTERN_EN.match(line)
            if m:
                month_name, day, year = m.group(1), int(m.group(2)), int(m.group(3))
                month = MONTH_MAP.get(month_name, 1)
                current_date = (year, month, day)
                continue

            if current_date is None:
                continue

            # Try Korean message
            m = MSG_PATTERN_KO.match(line)
            if m:
                sender = m.group(1)
                ampm = m.group(2)
                hour = int(m.group(3))
                minute = int(m.group(4))
                content = m.group(5)

                # Convert to 24h
                if ampm == "오후" and hour != 12:
                    hour += 12
                elif ampm == "오전" and hour == 12:
                    hour = 0

                try:
                    timestamp = datetime(
                        current_date[0], current_date[1], current_date[2],
                        hour, minute
                    )
                except ValueError:
                    continue  # skip invalid date/time

                if content.strip():
                    messages.append(ChatMessage(
                        timestamp=timestamp,
                        sender=sender,
                        content=content.strip(),
                        room=room,
                        platform="kakao",
                    ))
                continue

            # Try English message
            m = MSG_PATTERN_EN.match(line)
            if m:
                sender = m.group(1)
                hour = int(m.group(2))
                minute = int(m.group(3))
                ampm = m.group(4)
                content = m.group(5)

                if ampm == "PM" and hour != 12:
                    hour += 12
                elif ampm == "AM" and hour == 12:
                    hour = 0

                try:
                    timestamp = datetime(
                        current_date[0], current_date[1], current_date[2],
                        hour, minute
                    )
                except ValueError:
                    continue  # skip invalid date/time

                if content.strip():
                    messages.append(ChatMessage(
                        timestamp=timestamp,
                        sender=sender,
                        content=content.strip(),
                        room=room,
                        platform="kakao",
                    ))
                continue

    return messages
