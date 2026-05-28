"""Tests for cli.main — argparse dispatcher behavior.

These don't spawn a subprocess; they call ``main(argv)`` directly so the
test runs in-process and assertions are cheap. Subprocess tests for the
console-script wiring live in test_cli_status.py.
"""

from __future__ import annotations

import os

import pytest

from cli.main import build_parser, main, _resolve_db_path


def test_parser_help_does_not_crash(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "csemsearch" in captured.out
    assert "status" in captured.out


def test_missing_command_errors():
    """No subcommand → argparse exits 2 (its standard usage-error code)."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_verbose_and_quiet_together_rejected(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--verbose", "--quiet", "status"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err


def test_resolve_db_path_uses_lancedb_path_env(monkeypatch, tmp_path):
    target = tmp_path / "my-index"
    monkeypatch.setenv("LANCEDB_PATH", str(target))
    assert _resolve_db_path(None) == os.path.abspath(str(target))


def test_resolve_db_path_explicit_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "envdb"))
    explicit = tmp_path / "explicit-db"
    assert _resolve_db_path(str(explicit)) == os.path.abspath(str(explicit))


def test_status_on_empty_db_path(monkeypatch, tmp_path, capsys):
    """Cold start — db_path doesn't exist yet. Exits 0, points the user at
    `csemsearch index`."""
    nonexistent = tmp_path / "no-index-here"
    rc = main(["--db-path", str(nonexistent), "status"])
    assert rc == 0
    captured = capsys.readouterr()
    assert str(nonexistent) in captured.out
    assert "csemsearch index" in captured.out
