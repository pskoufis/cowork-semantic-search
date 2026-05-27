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
- **Multiple formats** -- txt, md, pdf, docx, pptx, pst, mbox, msg out of the box. (Spreadsheet support — csv, xlsx, xlsm, xls — is wired but temporarily disabled.)
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
| Outlook archive | `.pst` | One part per mail message, with attachment text |
| Unix mailbox | `.mbox` | Indexer auto-unpacks each `.mbox` into a sibling `<stem>_unpacked/` tree of per-thread directories with one `.txt` per message + materialized attachments; the unpacked files (and attachments via their native parsers) are what gets indexed |
| Outlook message | `.msg` | Indexer auto-unpacks each `.msg` into a sibling `<stem>.txt` (headers + body) and places attachments in a shared sibling `attachments/<stem>__<name>` layout; the unpacked files (and attachments via their native parsers) are what gets indexed |

> **Mbox detail:** when the indexer encounters `archive.mbox`, it first writes (or refreshes, on mtime change) `archive_unpacked/` next to it — one folder per thread (`thread-NNNN_<subject-slug>/`), one `msg-NNNN.txt` per message, attachments under `attachments/msg-NNNN/`, orphan messages under `_unthreaded/`. The `.mbox` itself is not indexed; the resulting `.txt` files and attachments are picked up by the normal walk via their native extensions. To unpack an mbox manually outside an indexing run, use `python -m mbox_handling.unpack <mbox>` (writes to `<stem>_unpacked/` by default, or pass `--output-dir <dir>` to choose a path).

> **Msg detail:** when the indexer encounters `foo.msg`, it writes (or refreshes, on mtime change) a sibling `foo.txt` containing the headers + plain-text body, and extracts each attachment to `attachments/foo__<original-name>` next to the `.msg`. Multiple `.msg` files in the same directory share one `attachments/` folder; the `<stem>__` prefix prevents collisions. Nested `.msg` attachments (forwarded emails) are recursively unpacked. The `.msg` itself is not indexed; the resulting `.txt` and attachments are picked up by the normal walk. Requires the `msg` extra: `pip install 'cowork-semantic-search[msg]'`. To unpack a single `.msg` manually outside an indexing run, use `python -m msg_handling.unpack <msg-path>`.

> **Spreadsheets (csv, xlsx, xlsm, xls) are temporarily disabled.** The description-based path (preview → LLM-written description → embedded chunk) is implemented in `server/spreadsheets.py` and the MCP queue tools (`list_pending_descriptions` / `submit_description` / `dismiss_pending_description`) are still available, but `index_folder` does not currently discover these files. Description chunks already in the index from prior runs are preserved. Re-enable by restoring `.csv`/`.xlsx`/`.xlsm`/`.xls` to `SUPPORTED_EXTENSIONS` in `server/parsers.py` and removing the subtraction in `server/indexer.py:discover_files`.

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
| `index_folder` | Index or re-index all documents in a folder. Incremental -- skips unchanged files. Honours a `.semanticignore` at the folder root and an optional `exclude` parameter (see [Excluding files](#excluding-files-and-folders) below). |
| `semantic_search` | Search indexed documents using natural language. Supports `vector` and `hybrid` modes. |
| `get_index_status` | Show total chunks, file count, indexed files, index size on disk, and background-job history. Pass `folder_path` to also surface the active `.semanticignore` for that folder. |
| `reindex_file` | Force re-index a single file, bypassing the hash cache. Bypasses exclusion rules -- this is an explicit per-file act. |

## How It Works

1. **Parse** -- extract text from each document, preserving structure (pages, slides)
2. **Chunk** -- split into ~400 character overlapping pieces for precise retrieval
3. **Embed** -- convert each chunk into a 256-dimensional vector using `Qwen/Qwen3-Embedding-0.6B` with Matryoshka truncation (the model is trained with MRL, so the head 256 dims are first-class — not a naive slice). First indexing run downloads ~1.2 GB of model weights from Hugging Face.
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
- Mount points differ per machine, so set `LANCEDB_PATH` to wherever the drive mounts on each Mac (e.g. `/Volumes/MyDrive` vs `/Volumes/MyDrive-1`).
- Use an **absolute** path -- the `./lancedb` default is relative to the working directory and is not portable.

**Cross-volume layouts (index and corpus on different volumes).** If `LANCEDB_PATH` is on a different volume than the indexed folder, the library falls back to storing **absolute** paths instead of relative ones. The index works, but it is no longer portable across remounts -- if the corpus drive mounts at a different path later (e.g. `/Volumes/MyDrive-1` instead of `/Volumes/MyDrive`), the stored paths become stale and you'll need to re-run `index_folder` against the new mount point (orphan cleanup will replace the stale rows). The most common reason to want this layout: the document drive is **exFAT** (cross-platform with Windows), which LanceDB cannot host because it lacks atomic rename -- so the index has to live on the internal APFS disk while the documents stay on the external drive.

</details>

<details id="excluding-files-and-folders">
<summary><strong>Excluding files and folders</strong></summary>

`index_folder` walks the target folder recursively and indexes anything matching the supported file types. For a typical project or `~/Documents` tree, that usually means you want to skip a few things — vendored libraries, build artifacts, caches, personal subfolders. Two ways to do that, combined:

- **`.semanticignore`** at the folder's root. Same syntax as `.gitignore` (via [`pathspec`](https://pypi.org/project/pathspec/)), durable and version-controllable. Example:

  ```gitignore
  # Build artifacts
  build/
  dist/

  # Vendored deps
  node_modules/
  vendor/

  # Logs and caches
  *.log
  .cache/

  # …but keep this one curated log
  !keep.log
  ```

- **`exclude` parameter** on the `index_folder` tool call. Same syntax, ad-hoc per call. Combined with whatever `.semanticignore` declares (union semantics; negation in the param can re-include a path the file excludes).

```
You: "Index ~/Documents/projects, but skip node_modules and any *.log files"
```

…becomes `index_folder(folder_path="~/Documents/projects", exclude=["node_modules/**", "*.log"])`.

**Re-runs converge.** Adding a rule to `.semanticignore` and re-running `index_folder` *prunes* the now-excluded chunks from the index — the result reports `files_excluded_pruned: N`. Removing a rule and re-running re-indexes the files normally on their next change.

**No default ignore list.** The indexer ships with no opinionated defaults — whatever you don't exclude gets indexed. The only hard-coded rule is that the active LanceDB directory cannot be indexed (so the index can't end up indexing itself when it lives under the walked folder); user negation cannot cancel this.

**Bad rules fail fast.** A syntactically invalid pattern is reported synchronously as `status: "rejected", reason: "invalid_exclusion_pattern"` and no indexing job is started.

**Inspecting active rules.** Call `get_index_status(folder_path="…")` to surface the active `.semanticignore` patterns for that folder. (Ad-hoc `exclude` patterns aren't persisted, so they don't appear here.)

</details>

<details>
<summary><strong>Indexing large folders & disk usage</strong></summary>

Indexing scales to large corpora (tens of GB, hundreds of thousands of files). A few things worth knowing:

- **Disk capacity.** The index stores each chunk's text plus a 256-dimensional vector, alongside full-text and ANN indexes. Expect the index directory to be **roughly the size of -- or larger than -- the source corpus**. Make sure the volume holding `LANCEDB_PATH` has headroom. Compaction runs automatically after every indexing run, so old versions and fragments don't pile up. `get_index_status` reports the current index size on disk (`db_size`).

- **Large files.** Files above a size cap (default **1024 MB**) are skipped rather than indexed, so a single huge file cannot exhaust memory -- every parser loads the whole file. Skipped files are reported in the `index_folder` result under `oversized_files`. Change the cap with the `MAX_FILE_SIZE_MB` environment variable (set it to `0` to disable the cap):

  ```json
  "env": { "MAX_FILE_SIZE_MB": "2048" }
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
  main.py        # MCP server + tool definitions
  parsers.py     # Per-format text extraction
  chunker.py     # Text splitting with metadata
  indexer.py     # Discovery, hashing, embedding pipeline
  store.py       # LanceDB vector store + FTS + hybrid search
  search.py      # Query embedding + search orchestration
  exclusions.py  # .semanticignore + exclude param (gitignore syntax)
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
| MSG | extract-msg | Outlook compound-document email format; pre-indexer unpacker spawns sibling .txt + attachments/ |

## Development

```bash
source .venv/bin/activate
pytest tests/ -v
```

180 tests covering parsers, chunking, indexing, search, path portability, exclusion rules, background indexing jobs, and MCP tool integration.

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
