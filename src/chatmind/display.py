"""Display: Plain text terminal output (Windows cp949 compatible)."""

from __future__ import annotations

from typing import List

from .models import ChatIndex
from .searcher import SearchResult


def display_search_results(results: List[SearchResult], query: str) -> None:
    """Display search results."""
    if not results:
        print(f"No results found for: '{query}'")
        return

    print()
    print(f"Search: {query}")
    print(f"Found {len(results)} results")
    print()

    for r in results:
        # Score indicator
        if r.score >= 0.8:
            indicator = "[HIGH]"
        elif r.score >= 0.6:
            indicator = "[MED] "
        else:
            indicator = "[LOW] "

        content_preview = r.message.content[:100]
        if len(r.message.content) > 100:
            content_preview += "..."

        print(f"  #{r.rank}  {indicator} {r.score:.2f}  {r.message.time_str()}")
        print(f"      {r.message.sender}: {content_preview}")
        if r.message.room:
            print(f"      [{r.message.room}]")
        print()


def display_stats(index: ChatIndex) -> None:
    """Display index statistics."""
    raw_mb = index.raw_memory_bytes / (1024 * 1024)
    comp_mb = index.compressed_memory_bytes / (1024 * 1024)
    ratio = index.raw_memory_bytes / max(index.compressed_memory_bytes, 1)
    savings_pct = (1 - index.compressed_memory_bytes / max(index.raw_memory_bytes, 1)) * 100

    print()
    print("=" * 50)
    print("  ChatMind Index Stats")
    print("=" * 50)
    print(f"  Messages indexed     : {len(index.messages):,}")
    print(f"  Platform             : {index.platform or 'mixed'}")
    print(f"  Rooms/Channels       : {len(index.rooms)}")
    print(f"  Unique senders       : {len(index.senders)}")
    print(f"  Embedding model      : {index.model_name}")
    print(f"  Embedding dimension  : {index.embedding_dim}")
    print(f"  ---")
    print(f"  Memory (uncompressed): {raw_mb:.2f} MB")
    print(f"  Memory (TurboQuant)  : {comp_mb:.2f} MB")
    print(f"  Compression ratio    : {ratio:.1f}x")
    print(f"  Memory savings       : {savings_pct:.0f}%")
    print(f"  ---")
    print(f"  Index time           : {index.index_time:.1f}s")
    print("=" * 50)
    print()


def display_index_summary(index: ChatIndex, path: str) -> None:
    """Display summary after indexing."""
    ratio = index.raw_memory_bytes / max(index.compressed_memory_bytes, 1)

    print()
    print("Indexing complete!")
    print(f"  > {len(index.messages):,} messages indexed")
    print(f"  > Platform: {index.platform or 'mixed'}")
    print(f"  > Rooms: {', '.join(index.rooms[:5]) if index.rooms else 'N/A'}")
    print(f"  > Senders: {len(index.senders)} people")
    print(
        f"  > Compressed: "
        f"{index.raw_memory_bytes / 1024:.1f} KB -> "
        f"{index.compressed_memory_bytes / 1024:.1f} KB "
        f"({ratio:.1f}x)"
    )
    print(f"  > Saved to {path}")
    print()


def display_rooms(index: ChatIndex) -> None:
    """Display list of rooms/channels."""
    if not index.rooms:
        print("No rooms found in index.")
        return

    # Count messages per room in single pass
    room_counts = {}
    for m in index.messages:
        room_counts[m.room] = room_counts.get(m.room, 0) + 1

    print()
    print(f"Rooms/Channels ({len(index.rooms)}):")
    for room in index.rooms:
        count = room_counts.get(room, 0)
        print(f"  - {room} ({count:,} messages)")
    print()


def display_people(index: ChatIndex) -> None:
    """Display list of participants."""
    if not index.senders:
        print("No senders found in index.")
        return

    # Count messages per sender
    sender_counts = {}
    for m in index.messages:
        sender_counts[m.sender] = sender_counts.get(m.sender, 0) + 1

    # Sort by message count
    sorted_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)

    print()
    print(f"Participants ({len(sorted_senders)}):")
    for sender, count in sorted_senders:
        print(f"  - {sender} ({count:,} messages)")
    print()
