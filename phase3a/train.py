import sys, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from phase3a.windows import build_windows
from phase3a.models import GRUForecaster
from phase3a.physics import interp_const_accel
from phase3a.metrics import ade_fde

VAL_SEQS = [78, 79, 80, 81, 82, 83]

def make_loader(w, batch=64, shuffle=False):
    v_last = w["x"][:, -1, 2:4]                       # (N,2) last-frame velocity
    ds = TensorDataset(torch.tensor(w["x"]), torch.tensor(w["y"]),
                       torch.tensor(w["p0"]), torch.tensor(w["gt"]),
                       torch.tensor(v_last))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

def _val_ade3s(model, w, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(w["x"]).to(device)).cpu().numpy()   # (N,3,2)
    dense = interp_const_accel(pred, w["x"][:, -1, 2:4])              # (N,75,2)
    return ade_fde(dense, w["gt"])

def train(train_csv, val_seqs=VAL_SEQS, out_path="phase3a/gru.pt", epochs=40, lr=1e-3):
    device = "cpu"
    tr = build_windows(train_csv, exclude_seqs=val_seqs)
    va = build_windows(train_csv, seqs=val_seqs)
    print(f"train windows {len(tr['x'])}  val windows {len(va['x'])}  device {device}")
    model = GRUForecaster().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = make_loader(tr, batch=64, shuffle=True)
    best, best_m = 1e9, None
    for ep in range(epochs):
        model.train()
        for xb, yb, *_ in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = ((model(xb) - yb) ** 2).mean()
            loss.backward(); opt.step()
        m = _val_ade3s(model, va, device)
        print(f"ep{ep:02d} val ADE@3s {m['ade_3s']:.3f}")
        if m["ade_3s"] < best:
            best, best_m = m["ade_3s"], m
            torch.save(model.state_dict(), out_path)
    print("best val:", best_m)
    return best_m

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "thesis_results/phase2/features_valid_smoothed.csv"
    train(csv)