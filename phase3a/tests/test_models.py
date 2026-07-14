import torch
from phase3a.models import GRUForecaster

def test_forward_shape():
    m = GRUForecaster()
    out = m(torch.randn(8, 50, 4))
    assert out.shape == (8, 3, 2)

def test_overfits_one_batch():
    torch.manual_seed(0)
    m = GRUForecaster(hidden=32)
    x = torch.randn(16, 50, 4); y = torch.randn(16, 3, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    l0 = None
    for _ in range(200):
        opt.zero_grad(); out = m(x); loss = ((out - y) ** 2).mean()
        loss.backward(); opt.step()
        if l0 is None: l0 = loss.item()
    assert loss.item() < 0.5 * l0