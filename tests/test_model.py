import torch

from nanofable.bitlinear import BitLinear
from nanofable.config import TIERS
from nanofable.model import build_model


def test_forward_shapes():
    m = build_model(TIERS["tiny"], "fp16")
    idx = torch.randint(0, 4096, (2, 16))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, 4096) and loss.ndim == 0


def test_precision_swaps_linears():
    mt = build_model(TIERS["tiny"], "ternary")
    assert any(isinstance(x, BitLinear) for x in mt.modules())
    mf = build_model(TIERS["tiny"], "fp16")
    assert not any(isinstance(x, BitLinear) for x in mf.modules())


def test_head_tied_to_embedding():
    m = build_model(TIERS["tiny"], "fp16")
    assert m.lm_head.weight is m.tok_emb.weight


def test_embeddings_never_quantized():
    mt = build_model(TIERS["tiny"], "ternary")
    assert not isinstance(mt.tok_emb, BitLinear)


def test_quantized_count_is_seven_per_block():
    # q,k,v,o + gate,up,down = 7 BitLinears per block, none outside blocks.
    cfg = TIERS["tiny"]
    mt = build_model(cfg, "ternary")
    n_bitlinear = sum(isinstance(x, BitLinear) for x in mt.modules())
    assert n_bitlinear == 7 * cfg.n_layer


def test_loss_decreases_on_overfit():
    torch.manual_seed(0)
    m = build_model(TIERS["tiny"], "fp16")
    seq = torch.randint(3, 4096, (2, 17))  # avoid pad id 2 so nothing is ignored
    x, y = seq[:, :-1], seq[:, 1:]  # real next-token prediction
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    _, l0 = m(x, y)
    l0 = l0.item()
    loss = l0
    for _ in range(30):
        opt.zero_grad()
        _, step_loss = m(x, y)
        step_loss.backward()
        opt.step()
        loss = step_loss.item()
    assert loss < l0
