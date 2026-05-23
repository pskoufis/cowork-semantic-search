---
name: Cowork RAG Plugin
description: Local semantic search Claude Code plugin — tech stack, architecture decisions, implementation phases
type: project
---

Claude Code plugin for local semantic search over document folders. Fully offline after initial model download. Repo: github.com/ZhuBit/cowork-semantic-search (private GitHub, Omar's personal account).

**Phase 1 MVP (completed 2026-03-24):** Plugin scaffolding, FastMCP server with `index_folder` + `semantic_search` tools, .txt/.md parsing, RecursiveCharacterTextSplitter chunking, paraphrase-multilingual-MiniLM-L12-v2 embeddings (lazy-loaded), LanceDB vector store, SHA-256 incremental indexing.

**Embedding model swap (v0.2.0, 2026-05-23):** Replaced paraphrase-multilingual-MiniLM-L12-v2 (384-dim) with `Qwen/Qwen3-Embedding-0.6B` truncated to 256-dim via Matryoshka. Uses `prompt_name="query"` for query encoding, `padding_side="left"` (last-token pooling on a decoder). Schema vector field + IVF_PQ `num_sub_vectors` derive from `EMBEDDING_DIM` in `server/store.py`. Breaking change for existing indexes — dim guard in `VectorStore._get_table` tells users to delete `LANCEDB_PATH` and re-index. Requires `transformers>=4.51.0`. Why: better retrieval quality + 33% smaller vectors on disk; user works mostly on Mac so the larger model's MPS slowdown is acceptable. How to apply: tests stub `get_model`; never depend on the model running in CI.

**Phase 2 (completed 2026-03-24):** PDF/DOCX/PPTX/CSV parsers (pymupdf, python-docx, python-pptx, stdlib csv), hybrid search (FTS + vector via manual RRF — LanceDB native hybrid requires registered embedding fn), `get_index_status` and `reindex_file` MCP tools, `mode` param on `semantic_search` (vector|hybrid). 56 passing tests. README, LICENSE, .mcp.json for local dev configured.

**Phase 3 (next):**
- ONNX runtime for faster embeddings (no PyTorch dependency)
- Performance: async indexing, progress callbacks, batch size tuning
- Configurable chunk size / overlap via tool params or env vars
- Marketplace publishing (Claude Code plugin registry)
- License decision pending: considering AGPL-3.0 + dual licensing for open-source with commercial upside

**Phase 4 (future):**
- Multi-folder index management (named indexes)
- Metadata filtering (date ranges, tags, custom fields)
- Summarization tool (retrieve + summarize in one call)
- Watch mode (auto-reindex on file changes)
- Web UI for index management

**Why:** Omar wants a local, offline, open-source RAG solution that works with German + English documents via Claude Code plugin system. Considering open-sourcing with a license that protects commercial upside if adoption grows.

**How to apply:** Python 3.13 required (system python is 3.9, use /opt/homebrew/bin/python3.13). pyproject.toml has `[tool.setuptools.packages.find] include = ["server*"]` to avoid flat-layout discovery issues from claude code dirs (hooks/, memory/, etc.). Git profile: ZhuBit (private GitHub).
