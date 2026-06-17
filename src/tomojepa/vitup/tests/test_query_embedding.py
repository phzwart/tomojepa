"""Query-embedding test (acceptance criterion 2)."""
import types

import torch

from tomojepa.vitup.query_embedding import QueryEmbedding


def _qe():
    # sample() does not touch the adapter; a stub suffices.
    return QueryEmbedding(adapter=types.SimpleNamespace(p=16), query_embed_grid=8)


def test_bilinear_recovers_cache_at_centers():
    G, C = 8, 5
    qe = _qe()
    cache = torch.randn(1, C, G, G)
    # low-res grid == cache grid; query at each cell center (i+0.5, j+0.5)
    ii, jj = torch.meshgrid(torch.arange(G) + 0.5, torch.arange(G) + 0.5, indexing="ij")
    coords = torch.stack([ii, jj], dim=-1).reshape(1, G * G, 2).float()
    q = qe.sample(cache, coords, h=G, w=G)               # [1, G*G, C]
    expected = cache[0].permute(1, 2, 0).reshape(G * G, C)
    assert q.shape == (1, G * G, C)
    assert torch.allclose(q[0], expected, atol=1e-5)


def test_cache_shape(adapter):
    qe = QueryEmbedding(adapter, query_embed_grid=8)
    cache = qe.compute_cache(torch.randn(2, 1, 64, 64))
    assert cache.shape == (2, adapter.C, 8, 8)
