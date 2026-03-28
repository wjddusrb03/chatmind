"""CLI: Click-based command line interface for ChatMind."""

from __future__ import annotations

import sys
import os

import click

from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="chatmind")
def main():
    """ChatMind - Semantic search for chat messages.

    Powered by TurboQuant vector compression (ICLR 2026).
    Supports Discord, KakaoTalk, and more.
    """
    pass


@main.command()
@click.argument("filepath")
@click.option("--platform", "-P", default="", help="Platform: discord, kakao. Auto-detected if omitted.")
@click.option("--room", "-r", default="", help="Room/channel name override.")
@click.option("--bits", default=3, type=int, help="Quantization bits (2-4). Default: 3.")
@click.option("--model", "-m", default="paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model.")
@click.option("--output", "-o", default=".", help="Output directory for index. Default: current dir.")
def index(filepath, platform, room, bits, model, output):
    """Index chat messages from an export file.

    FILEPATH can be a .json, .csv, or .txt file, or a directory.

    Examples:
        chatmind index discord_export.json
        chatmind index kakao_chat.txt --platform kakao
        chatmind index ./exports/ --platform discord
    """
    from .parsers import auto_parse
    from .indexer import build_index
    from .storage import save_index
    from .display import display_index_summary

    # Validate file exists
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found.")
        sys.exit(1)

    # Parse messages
    print(f"Parsing: {os.path.abspath(filepath)}")
    try:
        messages = auto_parse(filepath, platform=platform, room=room)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not messages:
        print("No messages found. Check the file format.")
        sys.exit(0)

    print(f"Found {len(messages):,} messages")

    # Build index
    chat_index = build_index(messages, model_name=model, bits=bits)

    # Save
    save_path = save_index(chat_index, base_path=output)

    # Display summary
    display_index_summary(chat_index, save_path)


@main.command()
@click.argument("query")
@click.option("--path", "-p", default=".", help="Path to index directory.")
@click.option("-k", default=5, type=int, help="Number of results. Default: 5.")
@click.option("--from", "sender", default=None, help="Filter by sender name.")
@click.option("--room", "-r", default=None, help="Filter by room/channel.")
@click.option("--after", default=None, help="Messages after date (YYYY-MM-DD).")
@click.option("--before", default=None, help="Messages before date (YYYY-MM-DD).")
def search(query, path, k, sender, room, after, before):
    """Search messages by natural language query.

    Auto-indexes if a chat export file exists but no index.

    Examples:
        chatmind search "restaurant recommendation"
        chatmind search "meeting schedule" --from John
        chatmind search "travel plan" --room general --after 2024-01-01
    """
    from .storage import load_index, index_exists
    from .searcher import search as do_search
    from .display import display_search_results
    from datetime import datetime

    # Parse date filters
    after_dt = None
    before_dt = None
    if after:
        try:
            after_dt = datetime.strptime(after, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{after}'. Use YYYY-MM-DD.")
            sys.exit(1)
    if before:
        try:
            before_dt = datetime.strptime(before, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{before}'. Use YYYY-MM-DD.")
            sys.exit(1)

    # Load index
    if not index_exists(base_path=path):
        print(f"No index found. Run 'chatmind index <file>' first to index your chat export.")
        sys.exit(1)

    chat_index = load_index(base_path=path)

    # Search
    results = do_search(
        query, chat_index, k=k,
        sender=sender, room=room,
        after=after_dt, before=before_dt,
    )
    display_search_results(results, query)


@main.command()
@click.option("--path", "-p", default=".", help="Path to index directory.")
def stats(path):
    """Show index statistics."""
    from .storage import load_index
    from .display import display_stats

    try:
        chat_index = load_index(base_path=path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    display_stats(chat_index)


@main.command()
@click.option("--path", "-p", default=".", help="Path to index directory.")
def rooms(path):
    """List all rooms/channels in the index."""
    from .storage import load_index
    from .display import display_rooms

    try:
        chat_index = load_index(base_path=path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    display_rooms(chat_index)


@main.command()
@click.option("--path", "-p", default=".", help="Path to index directory.")
def people(path):
    """List all participants in the index."""
    from .storage import load_index
    from .display import display_people

    try:
        chat_index = load_index(base_path=path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    display_people(chat_index)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
