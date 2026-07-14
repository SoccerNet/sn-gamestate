import numpy as np

def confusion(y_true, y_pred, n_classes=3):
    m = np.zeros((n_classes, n_classes), int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        m[int(t), int(p)] += 1
    return m

def macro_f1(y_true, y_pred, n_classes=3):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = confusion(y_true, y_pred, n_classes)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))   # present classes only
    f1s = []
    for c in labels:
        tp = m[c, c]; fp = m[:, c].sum() - tp; fn = m[c, :].sum() - tp
        if tp == 0:
            f1s.append(0.0); continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0