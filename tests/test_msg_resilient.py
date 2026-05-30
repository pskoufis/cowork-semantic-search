"""Tests for the best-effort fallback in msg_handling.messages.read_message.

Real malformed .msg files can't be synthesized (extract-msg has no writer), so
the fallback orchestration is exercised by monkeypatching ``extract_msg.openMsg``
to fail on the normal parse and succeed under the lenient options. A real-fixture
regression confirms the happy path is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("extract_msg")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "example-msg-files"


class _FakeMsg:
    """Minimal stand-in for an extract-msg message, usable as a context
    manager and exposing the attributes read_message touches."""

    def __init__(self, *, body: str = "", subject: str = "") -> None:
        self.body = body
        self.htmlBody = None
        self.subject = subject
        self.sender = "sender@example.com"
        self.to = "to@example.com"
        self.cc = ""
        self.messageId = "<id@example.com>"
        self.date = None
        self.receivedTime = None
        self.props = {}
        self.attachments = ()

    def __enter__(self) -> _FakeMsg:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _fake_openMsg(*, fail_default: bool, recovered: _FakeMsg | None):
    """Build an openMsg replacement: the default (no-kwargs) call optionally
    raises; calls carrying lenient kwargs return ``recovered`` (or raise if
    None)."""
    from extract_msg.exceptions import UnrecognizedMSGTypeError

    def openMsg(path, **kwargs):
        if not kwargs:
            if fail_default:
                raise UnrecognizedMSGTypeError('Could not recognize MSG class type "x"')
            return recovered
        if recovered is None:
            raise ValueError("still broken under fallback")
        return recovered

    return openMsg


def test_fallback_recovers_text_when_primary_raises(monkeypatch) -> None:
    import extract_msg
    from msg_handling.messages import read_message

    recovered = _FakeMsg(body="recovered body", subject="Recovered Subject")
    monkeypatch.setattr(
        extract_msg, "openMsg", _fake_openMsg(fail_default=True, recovered=recovered)
    )

    out = read_message(Path("whatever.msg"))
    assert out.body == "recovered body"
    assert out.subject == "Recovered Subject"


def test_fallback_rejected_when_recovery_has_no_text(monkeypatch) -> None:
    """A lenient parse that opens but yields no body/subject is an empty
    'success' — we must reject it and surface the original error instead."""
    import extract_msg
    from extract_msg.exceptions import UnrecognizedMSGTypeError
    from msg_handling.messages import read_message

    empty = _FakeMsg(body="", subject="")
    monkeypatch.setattr(
        extract_msg, "openMsg", _fake_openMsg(fail_default=True, recovered=empty)
    )

    with pytest.raises(UnrecognizedMSGTypeError):
        read_message(Path("whatever.msg"))


def test_all_strategies_failing_raises_original(monkeypatch) -> None:
    import extract_msg
    from extract_msg.exceptions import UnrecognizedMSGTypeError
    from msg_handling.messages import read_message

    monkeypatch.setattr(
        extract_msg, "openMsg", _fake_openMsg(fail_default=True, recovered=None)
    )

    with pytest.raises(UnrecognizedMSGTypeError):
        read_message(Path("whatever.msg"))


def test_healthy_file_is_not_sent_through_fallback(monkeypatch) -> None:
    """When the normal parse succeeds, the fallback options are never tried."""
    import extract_msg
    from msg_handling.messages import read_message

    good = _FakeMsg(body="normal body", subject="Subj")
    calls: list[dict] = []

    def openMsg(path, **kwargs):
        calls.append(kwargs)
        return good

    monkeypatch.setattr(extract_msg, "openMsg", openMsg)

    out = read_message(Path("good.msg"))
    assert out.body == "normal body"
    assert calls == [{}]  # only the default call, no lenient retry


def test_real_good_msg_still_parses() -> None:
    """Regression: the refactor must not change the happy path on a real file."""
    from msg_handling.messages import read_message

    fixture = FIXTURES_DIR / "unicode.msg"
    if not fixture.is_file():
        pytest.skip("unicode.msg fixture not present")
    out = read_message(fixture)
    assert out.body.strip()  # real content recovered, no fallback needed
