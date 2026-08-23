import torch

from chemkan.normalization import MinMaxNormalizer


def _norm():
    u_min = torch.tensor([0.0, 10.0, -1.0])
    u_max = torch.tensor([1.0, 20.0, 1.0])
    return MinMaxNormalizer(u_min, u_max)


def test_maps_min_to_zero_and_max_to_one():
    n = _norm()
    assert torch.allclose(n.normalize(n.u_min), torch.zeros(3))
    assert torch.allclose(n.normalize(n.u_max), torch.ones(3))


def test_roundtrip_preserves_shape():
    n = _norm()
    x = torch.rand(4, 6, 3) * torch.tensor([1.0, 10.0, 2.0]) + torch.tensor([0.0, 10.0, -1.0])
    out = n.denormalize(n.normalize(x))
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5)


def test_zero_range_column_is_safe():
    n = MinMaxNormalizer(torch.tensor([5.0]), torch.tensor([5.0]))
    assert torch.isfinite(n.normalize(torch.tensor([5.0]))).all()


def test_subset_selects_columns_and_preserves_dtype():
    n = _norm().double()                                   # float64 buffers
    sub = n.subset(slice(0, 2))
    assert sub.u_min.shape == (2,)
    assert sub.u_min.dtype == torch.float64                # dtype preserved
    assert torch.allclose(sub.normalize(torch.tensor([1.0, 20.0], dtype=torch.float64)),
                          torch.ones(2, dtype=torch.float64))


# --- module / buffer / device behavior --------------------------------------------

def test_stats_are_registered_buffers_not_params():
    n = _norm()
    names = dict(n.named_buffers())
    assert {"u_min", "u_max", "range"} <= set(names)
    assert list(n.parameters()) == []                      # nothing trainable


def test_buffers_follow_to_dtype():
    n = _norm().double()                                   # .to(...) moves buffers
    assert n.u_min.dtype == n.u_max.dtype == n.range.dtype == torch.float64


def test_buffers_in_state_dict():
    assert {"u_min", "u_max", "range"} <= set(_norm().state_dict().keys())
