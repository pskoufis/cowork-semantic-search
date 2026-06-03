from server.multi_search import _rrf_merge


def _hit(src, ci, idx):
    return {"source_file": src, "chunk_index": ci, "index_path": idx,
            "model_alias": "m", "text": src, "score": 0.5}


def test_rrf_merges_and_dedups_same_chunk():
    # Same (source_file, chunk_index) tops both indexes -> summed, returned once.
    a = [_hit("/f.txt", 0, "/idxA"), _hit("/g.txt", 0, "/idxA")]
    b = [_hit("/f.txt", 0, "/idxB"), _hit("/h.txt", 0, "/idxB")]
    merged = _rrf_merge([("/idxA", a), ("/idxB", b)], top_k=10)
    keys = [(m["source_file"], m["chunk_index"]) for m in merged]
    assert keys.count(("/f.txt", 0)) == 1            # deduped
    top = merged[0]
    assert top["source_file"] == "/f.txt"            # boosted by appearing in both
    assert set(top["from_indexes"]) == {"/idxA", "/idxB"}
    assert top["score"] == 0.5                        # native score preserved


def test_rrf_respects_top_k():
    a = [_hit(f"/a{i}.txt", 0, "/idxA") for i in range(5)]
    merged = _rrf_merge([("/idxA", a)], top_k=3)
    assert len(merged) == 3
