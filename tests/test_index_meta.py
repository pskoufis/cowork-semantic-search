import pytest

from server.index_meta import read_meta, write_meta, resolve_index_profile
from server.embedding_models import resolve_profile


def test_write_then_read_roundtrip(tmp_path):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)
    write_meta(db, profile, dim)
    meta = read_meta(db)
    assert meta == {
        "model_alias": "minilm",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    }


def test_read_missing_returns_none(tmp_path):
    assert read_meta(str(tmp_path / "nope")) is None


def test_resolve_existing_index_ignores_env(tmp_path):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)
    write_meta(db, profile, dim)
    # Env says qwen, but the recorded index wins for a read path.
    got_profile, got_dim = resolve_index_profile(
        db, env_alias="qwen3-0.6b", env_dim=None, for_write=False
    )
    assert got_profile.alias == "minilm"
    assert got_dim == 384


def test_resolve_write_mismatch_raises(tmp_path):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)
    write_meta(db, profile, dim)
    with pytest.raises(ValueError) as exc:
        resolve_index_profile(db, env_alias="bge-small", env_dim=None, for_write=True)
    assert "minilm" in str(exc.value) and "bge-small" in str(exc.value)


def test_resolve_write_matching_env_ok(tmp_path):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)
    write_meta(db, profile, dim)
    got_profile, got_dim = resolve_index_profile(
        db, env_alias="minilm", env_dim=None, for_write=True
    )
    assert got_profile.alias == "minilm" and got_dim == 384


def test_resolve_legacy_fallback(tmp_path):
    db = str(tmp_path / "idx")  # no meta file
    profile, dim = resolve_index_profile(db, env_alias=None, env_dim=None, for_write=False)
    assert profile.alias == "qwen3-0.6b"
    assert dim == 256


def test_resolve_new_index_uses_env(tmp_path):
    db = str(tmp_path / "idx")  # no meta yet
    profile, dim = resolve_index_profile(db, env_alias="minilm", env_dim=None, for_write=True)
    assert profile.alias == "minilm" and dim == 384
