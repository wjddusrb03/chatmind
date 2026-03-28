# ChatMind

**Semantic search for chat messages** powered by TurboQuant vector compression (ICLR 2026).

Search your Discord, KakaoTalk, and other chat exports by **meaning**, not just keywords.

> *"What was that restaurant my friend recommended?"*
> → ChatMind finds it even if you don't remember the exact words.

## Why ChatMind?

| Feature | Discord Search | Ctrl+F | **ChatMind** |
|---------|---------------|--------|-------------|
| Semantic search | No | No | **Yes** |
| Synonym recognition | No | No | **Yes** |
| Similarity score | No | No | **Yes** (0.0~1.0) |
| Offline | No | Yes | **Yes** |
| Cross-language | No | No | **Yes** |
| Filter by sender/date | Partial | No | **Yes** |

## Installation

```bash
pip install chatmind
```

Or from source:

```bash
git clone https://github.com/wjddusrb03/chatmind.git
cd chatmind
pip install -e ".[dev]"
```

## Quick Start

### 1. Export your Discord chat

Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) to export as JSON.

### 2. Index & Search

```bash
# Index the exported file
chatmind index discord_export.json

# Search by meaning
chatmind search "restaurant recommendation"
chatmind search "travel plans" --from Alex
chatmind search "homework help" --after 2024-01-01
chatmind search "game night" -k 10
```

### Example Output

```
Search: restaurant recommendation
Found 5 results

  #1  [HIGH]  0.89  2024-01-15 10:02
      Park: I found an amazing sushi restaurant near Gangnam station
      [general]

  #2  [HIGH]  0.84  2024-01-15 14:32
      Park: Try the pasta place on 5th street, their carbonara is amazing
      [general]

  #3  [MED]   0.71  2024-01-16 12:02
      Alex: Their tonkotsu ramen is incredible, thick and creamy broth
      [general]
```

## Supported Platforms

| Platform | Format | How to Export |
|----------|--------|---------------|
| **Discord** | `.json`, `.csv` | [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) |
| **KakaoTalk** | `.txt` | Chat Room > Menu > Export Chat |
| More coming soon! | | |

## CLI Commands

| Command | Description |
|---------|-------------|
| `chatmind index <file>` | Index chat messages |
| `chatmind search "query"` | Semantic search |
| `chatmind stats` | Show index statistics |
| `chatmind rooms` | List rooms/channels |
| `chatmind people` | List participants |

### Search Filters

```bash
chatmind search "query" --from "Alex"         # Filter by sender
chatmind search "query" --room "general"       # Filter by room
chatmind search "query" --after 2024-01-01     # After date
chatmind search "query" --before 2024-12-31    # Before date
chatmind search "query" -k 20                  # Top 20 results
```

## How It Works

```
Chat Export → Parse → Sentence Embeddings → TurboQuant Compression → Semantic Search
                      (multilingual model)   (2x memory savings)     (asymmetric scoring)
```

1. **Parse**: Auto-detect and parse Discord/KakaoTalk export files
2. **Embed**: Convert messages to 384D vectors using multilingual model
3. **Compress**: TurboQuant (Google, ICLR 2026) for ~2x memory savings
4. **Search**: Asymmetric scoring without decompression

## Multilingual Support

Uses `paraphrase-multilingual-MiniLM-L12-v2` model supporting 50+ languages:

- English, Korean, Japanese, Chinese, Spanish, French, German...
- Cross-language search: Search in Korean, find English results!

## Requirements

- Python 3.9+
- CPU only (no GPU needed)
- ~420 MB for embedding model (first-time download)

## Related Projects

- [CommitMind](https://github.com/wjddusrb03/commitmind) - Semantic search for Git commits
- [langchain-turboquant](https://github.com/wjddusrb03/langchain-turboquant) - TurboQuant for LangChain

## Feedback

Found a bug or have an idea? Please open an [Issue](https://github.com/wjddusrb03/chatmind/issues)!
Korean and English are both welcome.

## License

MIT License
