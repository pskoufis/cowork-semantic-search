# mbox exploration parser

**Status:** draft — pending implementation plan
**Date:** 2026-05-24
**Scope:** standalone exploration script, **not** a `server/parsers.py` plugin

## Problem

The repo already indexes `.pst` Outlook archives via `server/parsers.py:_extract_pst`. A sample `.mbox` file has been placed in `mbox_handling/samples/sample.mbox` to explore whether mbox should be a supported format too. Before committing to integration, we want a runnable script that parses the sample and produces output we can eyeball — headers, bodies, attachments — so we can validate the parsing approach against real mbox quirks before promoting it into the indexing pipeline.

## Goals

- One runnable Python script that parses an mbox file using only the standard library.
- Produces a `messages.jsonl` (one JSON object per message) **and** recreates every attachment as a real file on disk under an `attachments/msg-NNNN/` tree.
- Body extraction matches `server/parsers.py:_pst_body` precedence: text/plain → HTML-stripped text → empty.
- Robust per-message error isolation: a single broken message does not abort the run.

## Non-goals (v1)

- Integration into `server/parsers.py` or `SUPPORTED_EXTENSIONS`. That is a separate, later cycle once the script's output looks right.
- Unit tests in `tests/`. Validation is manual against `sample.mbox` for v1.
- Directory walks across multiple mboxes.
- Streaming/memory-bounded reads (sample is 5 KB; `mailbox.mbox` already lazy-loads message offsets).
- Deduplicating attachments by content hash.
- Resolving `cid:` references in HTML bodies to attachment paths.
- Recursing into `message/rfc822` nested message parts (placeholder `.eml` only).

## Approach

Use the Python standard library's `mailbox.mbox` to iterate messages (it handles the `From ` separator quirks including the in-body `>From ` escaping) and `email.message.EmailMessage` with `walk()` for the MIME tree. Body decoding via `get_content()` with a `get_payload(decode=True).decode("utf-8", errors="replace")` fallback for unknown/malformed charsets. HTML stripping via the same `html.parser.HTMLParser` subclass pattern used in `server/parsers.py:_strip_html`. Zero new dependencies.

Considered and rejected:
- **Manual `From ` splitting + `email.parser.BytesParser`** — reimplements what `mailbox.mbox` already gets right.
- **Third-party libraries (`mail-parser`, `flanker`)** — adds a dependency for an exploration script; risks the Apple-Silicon-wheel pain that bit `libpff` for PST. No upside here.

## File layout

```
mbox_handling/
  __init__.py                       # empty, enables python -m invocation
  parse_mbox.py                     # the script
  samples/
    sample.mbox                     # already present
  output/                           # gitignored
    <mbox-stem>/
      messages.jsonl
      attachments/
        msg-0001/<filename>
        msg-0002/<filename>
```

`mbox_handling/output/` should be added to `.gitignore`.

## CLI

```
python -m mbox_handling.parse_mbox [MBOX_PATH] [--output-dir DIR]
```

- `MBOX_PATH` positional, defaults to `mbox_handling/samples/sample.mbox`.
- `--output-dir` defaults to `mbox_handling/output/<mbox-stem>/`.
- Exit code 0 on success (including zero-message mbox); 1 if the mbox file is missing or unreadable.

## Components

All module-private except `main`. Order roughly top-down:

| Function | Purpose |
|---|---|
| `main(argv)` | Parse argv, open mbox, drive the loop, write JSONL, save attachments, exit. |
| `_parse_message(msg, idx)` | Take one `email.message.Message` + 1-based index, return `(record: dict, attachments: list[tuple[filename, bytes]])`. Pure — no I/O. |
| `_extract_body(msg)` | Walks MIME tree. Returns plain text from text/plain part if non-empty; else HTML-stripped text from text/html part; else `""`. |
| `_extract_attachments(msg)` | Walks MIME tree, collects every part with a filename (inline or attachment disposition). Returns `[(filename, content_type, content_id, bytes)]`. |
| `_strip_html(html)` | Minimal `HTMLParser` subclass that emits text only — mirrors `server/parsers.py:_strip_html`. |
| `_safe_filename(name)` | Strip path separators, NULs, control chars; fall back to `attachment-<n>.bin` if empty after sanitization. |
| `_decode_header(raw)` | `email.header.decode_header` + `make_header` to handle RFC 2047 encoded headers. |
| `_address_list(raw)` | Parse a To/Cc-style header into a list of decoded strings (or `[]` if absent). |

## Data flow per message

1. `mailbox.mbox(path)` iterates messages in order.
2. Build the record:

```json
{
  "index": 1,
  "from": "Sample User <user@real-world.com>",
  "to": ["foo.bar@inetsim.org"],
  "cc": [],
  "date": "Sun, 29 Apr 2007 12:15:30 +0200",
  "subject": "INetSim test mail",
  "message_id": "<46347093.5071984@localhost>",
  "body": "This is an INetSim POP3 test mail...",
  "attachments": [
    {
      "filename": "sample.gif",
      "content_type": "image/gif",
      "content_id": "<part1.09080805.02010507@localhost>",
      "size_bytes": 51200,
      "saved_to": "attachments/msg-0002/sample.gif"
    }
  ]
}
```

3. Append `json.dumps(record) + "\n"` to `messages.jsonl`.
4. For each attachment, write bytes to `attachments/msg-NNNN/<filename>` where `NNNN` is the 1-based message index, zero-padded to 4 digits.

Field absence rule: `cc`/`to` are always present as a (possibly empty) list. `attachments` is always present as a (possibly empty) list. Header strings missing from the source are `null`. The `saved_to` path in each attachment record is relative to the output dir (so `messages.jsonl` and the `attachments/` tree stay portable as a unit).

## Error handling

- **Per-message isolation.** Wrap per-message work in `try/except Exception`. On failure, emit a record with `{"index": N, "error": "<class>: <message>", ...partial headers if available}` and continue. Exit code stays 0 unless **zero** messages were processed.
- **Charset decoding.** `get_content()` raises `LookupError` for unknown charsets and `UnicodeDecodeError` for mis-declared bytes. Fall back to `get_payload(decode=True).decode("utf-8", errors="replace")`.
- **Encoded headers.** Run From/To/Cc/Subject through `_decode_header`. Mojibake stays mojibake — we don't out-clever stdlib.
- **Attachment filenames** with path separators (`../etc/passwd`), NULs, or control chars: sanitized by `_safe_filename`. **Never** join an un-sanitized name into a path.
- **Filename collisions within one message:** second/third occurrence gets `<stem>-2<ext>`, `<stem>-3<ext>`.
- **Zero-byte payload** is still saved; recorded as `size_bytes: 0`.
- **Nested `message/rfc822` parts** are not recursed in v1. Recorded as an attachment with `content_type: "message/rfc822"` and a placeholder `.eml` containing raw bytes.
- **Output dir already exists.** Truncate `messages.jsonl`; remove and recreate the `attachments/` subdir to avoid stale files. Print a one-line notice to stderr.
- **Missing mbox file.** Print error to stderr, exit 1.
- **Empty mbox** (zero messages). Write empty `messages.jsonl`, exit 0, print `"Parsed 0 messages"` to stderr.

## Validation (manual, in lieu of unit tests)

Run: `python -m mbox_handling.parse_mbox`

Acceptance criteria — all must hold:

1. `mbox_handling/output/sample/messages.jsonl` has exactly **2** lines.
2. Line 1: `subject == "INetSim test mail"`, `body` starts with `"This is an INetSim POP3 test mail..."`, `attachments == []`.
3. Line 2: `subject == "INetSim test mail with attachment"`, `body` starts with `"This is an INetSim POP3 test mail with attachment..."` (the text/plain part, since plain is preferred over HTML), `attachments` length 1, with `filename == "sample.gif"` and `content_type == "image/gif"`.
4. `mbox_handling/output/sample/attachments/msg-0002/sample.gif` exists and `file <path>` reports `GIF image data`.
5. `for line in open('messages.jsonl'): json.loads(line)` does not raise.

## Open questions

None at design time. Implementer should ask before deviating.

## Follow-ups (not part of this spec)

- If the manual run looks good, plan a second cycle to add `.mbox` to `server/parsers.py:SUPPORTED_EXTENSIONS` with a `_extract_mbox` that returns the standard `list[{text, metadata}]` shape and lands with real tests in `tests/`.
