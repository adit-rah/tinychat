import torch

from nanofable.bitlinear import BitLinear


def test_forward_is_ternary_times_scale():
    m = BitLinear(8, 8, bias=False)
    wq = m.quantized_weight()
    s = m.scale()
    levels = torch.unique((wq / s).round())
    assert levels.abs().max() <= 1  # values in {-1, 0, 1}


def test_ste_gradient_flows_to_latent():
    m = BitLinear(4, 4, bias=False)
    x = torch.randn(2, 4)
    m(x).sum().backward()
    assert m.weight.grad is not None and m.weight.grad.abs().sum() > 0


def test_scale_is_absmean():
    m = BitLinear(4, 4, bias=False)
    assert torch.allclose(m.scale(), m.weight.detach().abs().mean(), atol=1e-6)


def test_quantized_weight_only_three_levels():
    m = BitLinear(16, 16, bias=False)
    s = m.scale()
    q = (m.quantized_weight() / s).round()
    assert set(q.unique().tolist()) <= {-1.0, 0.0, 1.0}
