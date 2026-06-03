from unittest.mock import MagicMock

from server.embedding_models import resolve_profile
from server.indexer import _model_load_kwargs, encode_query


def test_load_kwargs_qwen_left_pads_and_truncates():
    profile, dim = resolve_profile("qwen3-0.6b", None)
    kwargs = _model_load_kwargs(profile, dim)
    assert kwargs["truncate_dim"] == 256
    assert kwargs["tokenizer_kwargs"] == {"padding_side": "left"}


def test_load_kwargs_minilm_no_padding_no_truncate():
    profile, dim = resolve_profile("minilm", None)
    kwargs = _model_load_kwargs(profile, dim)
    assert "tokenizer_kwargs" not in kwargs
    # Fixed-dim model: no MRL truncation requested.
    assert "truncate_dim" not in kwargs


def test_encode_query_uses_prompt_for_qwen():
    profile, _ = resolve_profile("qwen3-0.6b", None)
    model = MagicMock()
    model.encode.return_value = [[0.0]]
    encode_query(model, profile, "hi")
    _, kwargs = model.encode.call_args
    assert kwargs.get("prompt_name") == "query"


def test_encode_query_prepends_prefix_for_bge():
    profile, _ = resolve_profile("bge-small", None)
    model = MagicMock()
    model.encode.return_value = [[0.0]]
    encode_query(model, profile, "hi")
    args, _ = model.encode.call_args
    assert args[0] == ["Represent this sentence for searching relevant passages: hi"]


def test_encode_query_plain_for_gte():
    profile, _ = resolve_profile("gte-small", None)
    model = MagicMock()
    model.encode.return_value = [[0.0]]
    encode_query(model, profile, "hi")
    args, kwargs = model.encode.call_args
    assert args[0] == ["hi"]
    assert "prompt_name" not in kwargs
