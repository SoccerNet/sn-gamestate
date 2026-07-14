import sys, numpy as np, torch
from phase3b.graphs import build_dataset
from phase3b.models import GCN, GAT
from phase3b.metrics import macro_f1

VAL_SEQS = [78, 79, 80, 81, 82, 83]

def split_items(items, val_seqs=VAL_SEQS):
    vs = set(val_seqs)
    tr = [it for it in items if int(it["seq"]) not in vs]
    va = [it for it in items if int(it["seq"]) in vs]
    return tr, va

def _predict(model, items, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for it in items:
            logit = model(torch.tensor(it["nodes"]).to(device),
                          torch.tensor(it["edge"]).to(device))
            ps.append(int(logit.argmax())); ys.append(it["label"])
    return np.array(ys), np.array(ps)

def train(arch="gcn", epochs=60, lr=1e-3, out=None):
    device = "cpu"                       # torch 1.13.1 MPS saves NaN weights
    out = out or f"phase3b/{arch}.pt"
    items = build_dataset("valid")
    tr, va = split_items(items)
    print(f"{arch}: train {len(tr)}  val {len(va)}")
    model = (GCN() if arch == "gcn" else GAT()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = np.arange(len(tr))
    best, best_m = -1, None
    for ep in range(epochs):
        model.train(); np.random.RandomState(ep).shuffle(order)
        for i in order:
            it = tr[i]
            logit = model(torch.tensor(it["nodes"]), torch.tensor(it["edge"]))
            loss = torch.nn.functional.cross_entropy(logit[None], torch.tensor([it["label"]]))
            opt.zero_grad(); loss.backward(); opt.step()
        y, p = _predict(model, va, device); f1 = macro_f1(y, p)
        if f1 > best:
            best, best_m = f1, {"val_macro_f1": f1, "epoch": ep}
            torch.save(model.state_dict(), out)
        print(f"ep{ep:02d} val macro-F1 {f1:.3f}")
    print("best:", best_m, "->", out)
    return best_m

if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else "gcn")