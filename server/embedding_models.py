"""Curated registry of embedding models the indexer can use.

Each EmbeddingProfile captures everything that differs between models: the
HF id, the dimension(s) it supports, and the model-specific encoding quirks
(decoder left-padding, a built-in query prompt, or an instruction prefix).
The model is chosen by a short alias via the EMBEDDING_MODEL env var;
resolve_profile() validates the alias and an optional EMBEDDING_DIM override.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    alias: str
    model_id: str
    default_dim: int
    mrl: bool                      # Matryoshka-truncatable (dim is a range)
    min_dim: int | None            # lower bound for mrl models; else None
    padding_side: str | None       # "left" for decoder/last-token pooling
    query_prompt_name: str | None  # sentence-transformers built-in prompt name
    query_prefix: str | None       # instruction prefix; mutually exclusive w/ prompt_name
    normalize: bool = True


_PROFILES = [
    EmbeddingProfile(
        alias="qwen3-0.6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        default_dim=256,
        mrl=True,
        min_dim=64,
        padding_side="left",
        query_prompt_name="query",
        query_prefix=None,
    ),
    EmbeddingProfile(
        alias="bge-small",
        model_id="BAAI/bge-small-en-v1.5",
        default_dim=384,
        mrl=False,
        min_dim=None,
        padding_side=None,
        query_prompt_name=None,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    EmbeddingProfile(
        alias="gte-small",
        model_id="thenlper/gte-small",
        default_dim=384,
        mrl=False,
        min_dim=None,
        padding_side=None,
        query_prompt_name=None,
        query_prefix=None,
    ),
    EmbeddingProfile(
        alias="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        default_dim=384,
        mrl=False,
        min_dim=None,
        padding_side=None,
        query_prompt_name=None,
        query_prefix=None,
    ),
    EmbeddingProfile(
        alias="static-mrl",
        model_id="sentence-transformers/static-retrieval-mrl-en-v1",
        default_dim=256,
        mrl=True,
        min_dim=64,
        padding_side=None,
        query_prompt_name=None,
        query_prefix=None,
    ),
]

REGISTRY: dict[str, EmbeddingProfile] = {p.alias: p for p in _PROFILES}

DEFAULT_ALIAS = "qwen3-0.6b"

# Native (maximum) embedding width per MRL model. default_dim is the *chosen*
# default, which may itself be a truncation; the override ceiling is the
# native width.
_NATIVE_MAX = {
    "qwen3-0.6b": 1024,
    "static-mrl": 1024,
}


def _native_max(profile: EmbeddingProfile) -> int:
    return _NATIVE_MAX.get(profile.alias, profile.default_dim)


def resolve_profile(alias: str, dim_override: int | None) -> tuple[EmbeddingProfile, int]:
    """Look up a profile by alias and resolve its effective dimension.

    Raises ValueError on an unknown alias or an invalid dim override.
    """
    profile = REGISTRY.get(alias)
    if profile is None:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown embedding model alias {alias!r}. Available: {available}."
        )

    dim = dim_override if dim_override is not None else profile.default_dim

    if dim % 8 != 0:
        raise ValueError(
            f"Embedding dim {dim} must be divisible by 8 (IVF_PQ requirement)."
        )

    if profile.mrl:
        lo = profile.min_dim or 8
        hi = _native_max(profile)
        if not (lo <= dim <= hi):
            raise ValueError(
                f"{profile.alias} supports dims {lo}..{hi}; got {dim}."
            )
    elif dim != profile.default_dim:
        raise ValueError(
            f"{profile.alias} is fixed at {profile.default_dim} dims; cannot use {dim}."
        )

    return profile, dim
