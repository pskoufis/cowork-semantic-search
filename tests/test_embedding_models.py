import pytest

from server.embedding_models import resolve_profile, REGISTRY, EmbeddingProfile


def test_default_alias_is_qwen():
    profile, dim = resolve_profile("qwen3-0.6b", None)
    assert profile.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert dim == 256
    assert profile.padding_side == "left"
    assert profile.query_prompt_name == "query"


def test_unknown_alias_lists_available():
    with pytest.raises(ValueError) as exc:
        resolve_profile("nope", None)
    msg = str(exc.value)
    for alias in REGISTRY:
        assert alias in msg


def test_fixed_dim_model_rejects_override():
    with pytest.raises(ValueError):
        resolve_profile("bge-small", 256)  # bge is fixed at 384


def test_mrl_model_accepts_in_range_override():
    profile, dim = resolve_profile("qwen3-0.6b", 512)
    assert dim == 512


def test_mrl_model_rejects_out_of_range():
    with pytest.raises(ValueError):
        resolve_profile("qwen3-0.6b", 2048)  # above native 1024


def test_dim_must_be_divisible_by_eight():
    with pytest.raises(ValueError):
        resolve_profile("qwen3-0.6b", 100)


def test_bge_has_query_prefix_not_prompt():
    profile, _ = resolve_profile("bge-small", None)
    assert profile.query_prefix is not None
    assert profile.query_prompt_name is None
