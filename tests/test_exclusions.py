"""Unit tests for the ExclusionRules module."""

from pathlib import Path

import pytest

from server.exclusions import ExclusionRules


# ---------------------------------------------------------------------------
# Loading + matching basics
# ---------------------------------------------------------------------------

def test_empty_rules_match_nothing(tmp_path):
    """No .semanticignore and no extra patterns → nothing is excluded."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    rules = ExclusionRules.load(folder, extra_patterns=None, db_dir=str(tmp_path / "db"))
    assert rules.is_excluded(folder / "any.md", folder, is_dir=False) is False
    assert rules.is_excluded(folder / "subdir", folder, is_dir=True) is False
    assert rules.patterns == []
    assert rules.source_file is None


def test_extra_patterns_match_files(tmp_path):
    """extra_patterns=['*.log'] excludes .log files, leaves .txt alone."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    rules = ExclusionRules.load(
        folder, extra_patterns=["*.log"], db_dir=str(tmp_path / "db")
    )
    assert rules.is_excluded(folder / "a" / "b.log", folder, is_dir=False) is True
    assert rules.is_excluded(folder / "a" / "b.txt", folder, is_dir=False) is False
    assert "*.log" in rules.patterns


def test_semanticignore_file_is_loaded(tmp_path):
    """Patterns from .semanticignore at the indexed root are honoured, with
    directory-only, anchored, and negation semantics matching gitignore."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / ".semanticignore").write_text(
        "cache/\n"        # directory-only
        "/build\n"        # anchored to root
        "*.log\n"         # match any .log
        "!keep.log\n"     # but re-include keep.log
    )
    rules = ExclusionRules.load(
        folder, extra_patterns=None, db_dir=str(tmp_path / "db")
    )

    # Directory-only matches the cache subdir.
    assert rules.is_excluded(folder / "cache", folder, is_dir=True) is True

    # Anchored: top-level `build` matches, nested `sub/build` does not.
    assert rules.is_excluded(folder / "build", folder, is_dir=True) is True
    assert rules.is_excluded(folder / "sub" / "build", folder, is_dir=True) is False

    # Negation: *.log excludes, !keep.log re-includes.
    assert rules.is_excluded(folder / "a.log", folder, is_dir=False) is True
    assert rules.is_excluded(folder / "keep.log", folder, is_dir=False) is False

    # The source file path is exposed for status reporting.
    assert rules.source_file == folder / ".semanticignore"
    assert "cache/" in rules.patterns
    assert "/build" in rules.patterns


def test_union_of_file_and_extra_patterns(tmp_path):
    """A pattern in .semanticignore and another in extra_patterns both apply."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / ".semanticignore").write_text("*.log\n")
    rules = ExclusionRules.load(
        folder, extra_patterns=["*.tmp"], db_dir=str(tmp_path / "db")
    )
    assert rules.is_excluded(folder / "a.log", folder, is_dir=False) is True
    assert rules.is_excluded(folder / "a.tmp", folder, is_dir=False) is True
    assert rules.is_excluded(folder / "a.md", folder, is_dir=False) is False


# ---------------------------------------------------------------------------
# LanceDB self-protection
# ---------------------------------------------------------------------------

def test_self_protection_excludes_index_dir(tmp_path):
    """When db_dir lives inside folder, the LanceDB dir is excluded even
    without any user rule."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    db_dir = folder / "lancedb"
    db_dir.mkdir()
    rules = ExclusionRules.load(folder, extra_patterns=None, db_dir=str(db_dir))
    assert rules.is_excluded(db_dir, folder, is_dir=True) is True
    assert rules.is_excluded(db_dir / "data.lance" / "x.bin", folder, is_dir=False) is True


def test_self_protection_cannot_be_cancelled(tmp_path):
    """A `!lancedb/` line in .semanticignore must not be able to re-include
    the LanceDB dir — the self-protection is hard-coded, not a pathspec rule."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    db_dir = folder / "lancedb"
    db_dir.mkdir()
    (folder / ".semanticignore").write_text("!lancedb/\n!lancedb/**\n")
    rules = ExclusionRules.load(folder, extra_patterns=None, db_dir=str(db_dir))
    assert rules.is_excluded(db_dir, folder, is_dir=True) is True
    assert rules.is_excluded(db_dir / "data.lance" / "x.bin", folder, is_dir=False) is True


def test_self_protection_does_not_affect_unrelated_folder(tmp_path):
    """When db_dir lives outside folder, the self-protection does nothing."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    db_dir = tmp_path / "elsewhere" / "lancedb"
    db_dir.mkdir(parents=True)
    rules = ExclusionRules.load(folder, extra_patterns=None, db_dir=str(db_dir))
    assert rules.is_excluded(folder / "any.md", folder, is_dir=False) is False


# ---------------------------------------------------------------------------
# Bad patterns
# ---------------------------------------------------------------------------

def test_invalid_pattern_raises_with_offending_line(tmp_path):
    """A syntactically invalid gitwildmatch pattern raises ValueError carrying
    the offending line so the MCP layer can surface a useful message."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    bad = "\\"  # bare backslash — pathspec's gitwildmatch rejects this
    with pytest.raises(ValueError) as exc:
        ExclusionRules.load(
            folder, extra_patterns=[bad], db_dir=str(tmp_path / "db")
        )
    assert bad in str(exc.value)


def test_invalid_pattern_in_semanticignore_raises(tmp_path):
    """Bad lines in .semanticignore also raise ValueError, not silently drop."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / ".semanticignore").write_text("ok/\n\\\n")
    with pytest.raises(ValueError) as exc:
        ExclusionRules.load(folder, extra_patterns=None, db_dir=str(tmp_path / "db"))
    assert "\\" in str(exc.value)


# ---------------------------------------------------------------------------
# Directory-only patterns
# ---------------------------------------------------------------------------

def test_directory_only_pattern_requires_is_dir_true(tmp_path):
    """A `cache/` rule must match the cache subdir only when is_dir=True.
    With is_dir=False the same path must not match — guards a future refactor
    against silently breaking the os.walk directory prune."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    rules = ExclusionRules.load(
        folder, extra_patterns=["cache/"], db_dir=str(tmp_path / "db")
    )
    assert rules.is_excluded(folder / "cache", folder, is_dir=True) is True
    assert rules.is_excluded(folder / "cache", folder, is_dir=False) is False
