import torch

from nanofable.rope import apply_rope, build_rope_cache


def test_rope_preserves_norm():
    head_dim, ctx = 16, 32
    cos, sin = build_rope_cache(head_dim, ctx)
    q = torch.randn(2, 3, ctx, head_dim)
    k = torch.randn(2, 3, ctx, head_dim)
    qr, _ = apply_rope(q, k, cos, sin)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_rope_relative_position():
    # <R_t v, R_s v> depends only on (t - s) for a constant vector v.
    head_dim, ctx = 16, 32
    cos, sin = build_rope_cache(head_dim, ctx)
    v = torch.randn(head_dim)
    qk = v.expand(1, 1, ctx, head_dim).clone()
    rot, _ = apply_rope(qk, qk, cos, sin)  # [1,1,ctx,head_dim]
    rot = rot[0, 0]  # [ctx, head_dim]

    def dot(t, s):
        return float(rot[t] @ rot[s])

    # same relative offset -> same dot product
    assert abs(dot(5, 2) - dot(10, 7)) < 1e-3
    assert abs(dot(8, 3) - dot(20, 15)) < 1e-3
