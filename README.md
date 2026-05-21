# cowork-semantic-search

[![GitHub stars](https://img.shields.io/github/stars/ZhuBit/cowork-semantic-search?style=social)](https://github.com/ZhuBit/cowork-semantic-search/stargazers)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple)](https://modelcontextprotocol.io)

> If you find this useful, consider giving it a ⭐ — it helps others discover the project.

**Local semantic search for your documents. No API keys. No cloud. Works with any MCP client.**

![demo](assets/image.png)

---

## Why

AI coding tools are powerful, but they have blind spots when it comes to your local files:

- **Frozen knowledge** -- training data has a cutoff. Your latest reports, notes, and contracts don't exist in the model's world.
- **Context window limits** -- you can't paste 500 documents into a prompt.
- **No cross-file search** -- your AI tool can read one file at a time, but can't search across your entire document library for the relevant pieces.

This plugin bridges that gap. It indexes your local documents into a small, fast vector database. When you ask a question, it retrieves only the relevant pieces -- so your AI tool can answer with your actual data.

```
Your documents --> chunked --> embedded --> local vector DB
                                                 |
         Your question --> embedded --> similarity search --> relevant chunks --> AI answers
```

## Features

- **Fully offline** -- one-time model download (~120MB), then no network calls. No data leaves your machine.
- **Incremental indexing** -- SHA-256 content hashing. Only changed files get reprocessed. Re-indexing 1000 files where 3 changed takes seconds.
- **Multilingual** -- handles 50+ languages natively. Search in one language, find results in another.
- **Hybrid search** -- combines semantic similarity with full-text keyword search via Reciprocal Rank Fusion. Catches what pure vector search misses.
- **Multiple formats** -- txt, md, pdf, docx, pptx, csv out of the box.
- **Any MCP client** -- works with Claude Code, Cursor, Windsurf, Cline, and any other MCP-compatible tool.
- **Zero infrastructure** -- LanceDB stores everything as local files. No server, no Docker, no database to manage.

## Supported Formats

| Format | Extension | Details |
|--------|-----------|---------|
| Plain text | `.txt` | UTF-8 with fallback |
| Markdown | `.md` | Raw text preserved |
| PDF | `.pdf` | Page-level extraction with metadata |
| Word | `.docx` | Full paragraph extraction |
| PowerPoint | `.pptx` | Slide-level extraction with metadata |
| CSV | `.csv` | Row-based text extraction |
| Outlook archive | `.pst` | One part per mail message, with attachment text |

## Quick Start

### 1. Install

```bash
git clone https://github.com/ZhuBit/cowork-semantic-search.git
cd cowork-semantic-search
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

> **PST support compiles a C extension.** The `pst` extra — included in
> `all` — installs `libpff-python`, which has no Apple-Silicon wheel and
> builds from source, so the Xcode command-line tools
> (`xcode-select --install`) must be present. To skip it, install
> `pip install -e ".[pdf,docx,pptx]"` instead.

### 2. Configure your MCP client

Add the server to your MCP client's config. Replace paths with your own.

<details>
<summary><strong>Claude Code</strong> -- <code>.mcp.json</code> in your project root</summary>

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "server.main"],
      "cwd": "/absolute/path/to/cowork-semantic-search",
      "env": {
        "PYTHONPATH": "/absolute/path/to/cowork-semantic-search"
      }
    }
  }
}
```
</details>

<details>
<summary><strong>Cursor</strong> -- <code>.cursor/mcp.json</code> in your project root or <code>~/.cursor/mcp.json</code> globally</summary>

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "server.main"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/cowork-semantic-search"
      }
    }
  }
}
```
</details>

<details>
<summary><strong>Windsurf</strong> -- <code>~/.codeium/windsurf/mcp_config.json</code></summary>

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "server.main"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/cowork-semantic-search"
      }
    }
  }
}
```
</details>

<details>
<summary><strong>Cline</strong> -- MCP Servers settings in the Cline VS Code extension</summary>

Open Cline > MCP Servers icon > Configure > Advanced MCP Settings, then add:

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "server.main"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/cowork-semantic-search"
      }
    }
  }
}
```
</details>

### 3. Restart your MCP client and go

> "Index all documents in ~/Documents/projects"

> "Search for 'quarterly revenue report'"

First run downloads the embedding model (~120MB), then everything runs offline.

## Example: Search Your Obsidian Vault

If you keep notes in Obsidian (or any folder of markdown files), this plugin turns your AI tool into a search engine for your knowledge base.

```
You: "Index my vault at ~/Documents/ObsidianVault"
AI:  Indexed 847 files -> 3,291 chunks in 42s

You: "What did I write about API rate limiting?"
AI:  Found 6 relevant chunks across 3 files:
       - notes/backend/rate-limiting-strategies.md
       - projects/acme-api/design-decisions.md
       - daily/2025-11-03.md
       ...

You: "Find anything about the client meeting last November, use hybrid search"
AI:  Found 4 results using hybrid search (vector + keyword):
       - meetings/2025-11-12-acme-kickoff.md
       - daily/2025-11-12.md
       ...
```

Works the same with PDFs, Word docs, PowerPoints, and CSVs -- just point it at a folder.

## Tools

| Tool | Description |
|------|-------------|
| `index_folder` | Index or re-index all documents in a folder. Incremental -- skips unchanged files. |
| `semantic_search` | Search indexed documents using natural language. Supports `vector` and `hybrid` modes. |
| `get_index_status` | Show total chunks, file count, indexed files, index size on disk, and background-job history. |
| `reindex_file` | Force re-index a single file, bypassing the hash cache. |

## How It Works

1. **Parse** -- extract text from each document, preserving structure (pages, slides)
2. **Chunk** -- split into ~400 character overlapping pieces for precise retrieval
3. **Embed** -- convert each chunk into a 384-dimensional vector using `paraphrase-multilingual-MiniLM-L12-v2`
4. **Store** -- save chunks + vectors in a LanceDB database (a local file, no server needed)
5. **Search** -- embed your query, find nearest chunks by cosine similarity, optionally combine with full-text keyword search via RRF

## Advanced Usage

<details>
<summary><strong>Portable index (external drive)</strong></summary>

By default the index lives in `./lancedb` next to the server. To make the index **portable** -- so it survives an external drive being re-plugged at a different mount point, or moved between Macs -- set the `LANCEDB_PATH` environment variable to a directory **on the same drive as your documents**:

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "server.main"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/cowork-semantic-search",
        "LANCEDB_PATH": "/Volumes/MyDrive/.semantic-index"
      }
    }
  }
}
```

The index stores document paths **relative to `LANCEDB_PATH`**, so as long as the index directory and the indexed folders stay on the same volume, you can unplug the drive, re-plug it (even at a different mount point), or move it to another Mac -- incremental indexing and search keep working without re-indexing.

Notes:
- The index directory and **all** indexed folders must be on the same volume.
- Mount points differ per machine, so set `LANCEDB_PATH` to wherever the drive mounts on each Mac (e.g. `/Volumes/MyDrive` vs `/Volumes/MyDrive-1`).
- Use an **absolute** path -- the `./lancedb` default is relative to the working directory and is not portable.

</details>

<details>
<summary><strong>Indexing large folders & disk usage</strong></summary>

Indexing scales to large corpora (tens of GB, hundreds of thousands of files). A few things worth knowing:

- **Disk capacity.** The index stores each chunk's text plus a 384-dimensional vector, alongside full-text and ANN indexes. Expect the index directory to be **roughly the size of -- or larger than -- the source corpus**. Make sure the volume holding `LANCEDB_PATH` has headroom. Compaction runs automatically after every indexing run, so old versions and fragments don't pile up. `get_index_status` reports the current index size on disk (`db_size`).

- **Large files.** Files above a size cap (default **100 MB**) are skipped rather than indexed, so a single huge file cannot exhaust memory -- every parser loads the whole file. Skipped files are reported in the `index_folder` result under `oversized_files`. Change the cap with the `MAX_FILE_SIZE_MB` environment variable (set it to `0` to disable the cap):

  ```json
  "env": { "MAX_FILE_SIZE_MB": "250" }
  ```

- **Interrupted runs.** Indexing runs in the background and can take a long time on a large corpus. Job records are saved next to the index, so if the server is restarted mid-run, `get_index_status` reports that job as `interrupted`. There is no separate resume step -- just run `index_folder` again. Unchanged files are detected by a fast modification-time check and skipped, so re-running after an interruption is cheap.

</details>

<details>
<summary><strong>Use as a Python library</strong></summary>

```python
from server.indexer import index_folder
from server.search import semantic_search

# For a portable index, set the LANCEDB_PATH env var to an absolute path
# on the same drive as your documents (see "Portable index" above).

# Index a folder
result = index_folder("/path/to/docs")
print(f"{result['files_indexed']} files -> {result['total_chunks']} chunks")

# Search
results = semantic_search("project deadline", mode="hybrid")
for r in results["results"]:
    print(f"  {r['file_name']}: {r['text'][:100]}...")
```
</details>

## Architecture

```
server/
  main.py       # MCP server + tool definitions
  parsers.py    # Per-format text extraction
  chunker.py    # Text splitting with metadata
  indexer.py    # Discovery, hashing, embedding pipeline
  store.py      # LanceDB vector store + FTS + hybrid search
  search.py     # Query embedding + search orchestration
```

| Component | Choice | Why |
|-----------|--------|-----|
| MCP framework | FastMCP | Clean tool definitions, async support |
| Embeddings | sentence-transformers | Offline, multilingual, fast |
| Vector DB | LanceDB | Serverless, embedded, FTS built-in |
| Chunking | langchain-text-splitters | Battle-tested recursive splitting |
| PDF | PyMuPDF | Fast, accurate extraction |
| DOCX | python-docx | Lightweight, no system deps |
| PPTX | python-pptx | Slide-level extraction |
| PST | libpff (pypff) | Streams Outlook archives without reading them whole |

## Development

```bash
source .venv/bin/activate
pytest tests/ -v
```

131 tests covering parsers, chunking, indexing, search, path portability, background indexing jobs, and MCP tool integration.

Contributions welcome -- open an issue or submit a PR.

## Roadmap

- ONNX runtime for faster embeddings (drop PyTorch dependency)
- Configurable chunk size and overlap via tool params
- Multi-folder named indexes
- Metadata filtering (date ranges, tags, custom fields)
- Watch mode (auto-reindex on file changes)

## Support

If this is useful to you, consider giving it a ⭐ — it helps others find the project.

## License

AGPL-3.0 -- free to use, modify, and self-host. If you offer this as a network service, you must share your source code. See [LICENSE](LICENSE) for details.
