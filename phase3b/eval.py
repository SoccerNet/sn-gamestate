import numpy as np, pandas as pd, torch
from phase3b.graphs import build_dataset
from phase3b.models import GCN, GAT, majority_predict
from phase3b.metrics import macro_f1, confusion
from phase3b.train import build_dataset as _bd, split_items

def model_macro_f1(model, items):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for it in items:
            logit = model(torch.tensor(it["nodes"]), torch.tensor(it["edge"]))
            ps.append(int(logit.argmax())); ys.append(it["label"])
    return macro_f1(np.array(ys), np.array(ps))

def evaluate():
    test = build_dataset("test")
    y = np.array([it["label"] for it in test])
    train_labels = np.array([it["label"] for it in split_items(build_dataset("valid"))[0]])
    rows, cms = {}, {}
    mp = majority_predict(train_labels, len(test))
    rows["majority"] = macro_f1(y, mp); cms["majority"] = confusion(y, mp)
    for arch, M in (("gcn", GCN), ("gat", GAT)):
        m = M(); m.load_state_dict(torch.load(f"phase3b/{arch}.pt", map_location="cpu")); m.eval()
        ps = []
        with torch.no_grad():
            for it in test:
                ps.append(int(m(torch.tensor(it["nodes"]), torch.tensor(it["edge"])).argmax()))
        ps = np.array(ps); rows[arch] = macro_f1(y, ps); cms[arch] = confusion(y, ps)
    df = pd.DataFrame({"macro_f1": rows})
    print(f"test graphs: {len(test)}\n"); print(df.round(3).to_string())
    for k, cm in cms.items():
        print(f"\n{k} confusion (rows=true low/mid/high):\n{cm}")
    df.to_csv("phase3b/phase3b_pressing_results.csv")
    return df

if __name__ == "__main__":
    evaluate()