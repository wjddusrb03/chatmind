"""Storage: Save and load ChatIndex to/from disk."""

from __future__ import annotations

import os
import pickle
from typing import Optional

from .models import ChatIndex

DEFAULT_DIR = ".chatmind"
DEFAULT_FILE = "index.pkl"


def get_storage_path(base_path: str = ".", filename: Optional[str] = None) -> str:
    """Get the storage path for a chat index.

    Args:
        base_path: Base directory for storage.
        filename: Custom filename. Defaults to 'index.pkl'.

    Returns:
        Full path to the index file.
    """
    base = os.path.join(os.path.abspath(base_path), DEFAULT_DIR)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename or DEFAULT_FILE)


def save_index(index: ChatIndex, path: Optional[str] = None, base_path: str = ".") -> str:
    """Save a ChatIndex to disk.

    Args:
        index: The ChatIndex to save.
        path: Custom save path.
        base_path: Base directory (used if path is None).

    Returns:
        The path where the index was saved.
    """
    if path is None:
        path = get_storage_path(base_path)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(path)
    print(f"Index saved to {path} ({file_size / 1024:.1f} KB)")

    return path


def load_index(path: Optional[str] = None, base_path: str = ".") -> ChatIndex:
    """Load a ChatIndex from disk.

    Args:
        path: Custom load path.
        base_path: Base directory (used if path is None).

    Returns:
        The loaded ChatIndex.

    Raises:
        FileNotFoundError: If the index file doesn't exist.
    """
    if path is None:
        path = get_storage_path(base_path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No index found at {path}. Run 'chatmind index <file>' first."
        )

    with open(path, "rb") as f:
        index = pickle.load(f)

    if not isinstance(index, ChatIndex):
        raise ValueError(f"Invalid index file: {path}")

    print(f"Index loaded: {len(index.messages):,} messages")
    return index


def index_exists(base_path: str = ".") -> bool:
    """Check if an index exists at the given path."""
    path = get_storage_path(base_path)
    return os.path.exists(path)
