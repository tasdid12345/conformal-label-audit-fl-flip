"""Conformal label-transition audit vs. cosine-similarity outlier detector for
label-flipping clients in federated learning. Dataset: scikit-learn digits (bundled)."""
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
SEEDS = [42, 43, 44]          # federation seeds (client partition, attacker choice)
N_CLIENTS, N_ATTACK, ALPHA = 20, 3, 0.3
STEPS, LR = 100, 0.5          # fixed a priori (no tuning performed)
CAL_FRAC, MISCOV = 0.10, 0.10 # calibration share of train pool; 90% prediction sets
FLAG_FRAC = 0.3               # improved: flag if >30% of labels outside set
Z_THR = -2.0                  # baseline: flag if robust z-score < -2
QUESTION = ("Can a conformal label-transition audit flag a few label-flipping federated clients "
            "among strongly non-IID honest clients more reliably than a standard cosine-similarity "
            "update-outlier detector?")


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def train_softmax(X, y):
    """Multinomial logistic regression from zero init, full-batch gradient descent."""
    Xb, Y = add_bias(X), np.eye(10)[y]
    W = np.zeros((Xb.shape[1], 10))
    for _ in range(STEPS):
        W -= LR * Xb.T @ (softmax(Xb @ W) - Y) / len(y)
    return W


def partition(y, rng):
    """Dirichlet label-skew split across clients; resample until every client has >= 10 samples."""
    while True:
        parts = [[] for _ in range(N_CLIENTS)]
        for k in range(10):
            idx = rng.permutation(np.where(y == k)[0])
            props = rng.dirichlet(ALPHA * np.ones(N_CLIENTS))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for c, part in enumerate(np.split(idx, cuts)):
                parts[c].extend(part.tolist())
        if min(len(p) for p in parts) >= 10:
            return [np.array(p) for p in parts]


def fedavg(W, n_k, keep):
    """Sample-size-weighted average of kept client models (all clients if none kept)."""
    keep = np.where(keep)[0]
    if len(keep) == 0:
        keep = np.arange(len(W))
    w = n_k[keep] / n_k[keep].sum()
    return np.tensordot(w, W[keep], axes=1)


def accuracy(W, X, y):
    return float((np.argmax(add_bias(X) @ W, axis=1) == y).mean())


def main():
    print("Loading digits (bundled with scikit-learn)...")
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # pixel values 0..16 -> [0, 1]
    # Single stratified 80/20 split; test set used only once, at the end.
    Xp, Xte, yp, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    # 10% of the training pool held out as conformal calibration set.
    Xc, Xcal, yc, ycal = train_test_split(Xp, yp, test_size=CAL_FRAC, stratify=yp, random_state=SEED)
    Xall, yall = np.vstack([Xc, Xcal]), np.concatenate([yc, ycal])
    cal = np.arange(len(Xc), len(Xall))
    n_cal = len(cal)
    level = min(1.0, math.ceil((n_cal + 1) * (1 - MISCOV)) / n_cal)
    print(f"Train pool {len(Xp)}, test {len(Xte)}, calibration {n_cal}")

    runs = []
    for s in SEEDS:
        print(f"Seed {s}: partitioning and local training...")
        rng = np.random.default_rng(s)
        parts = partition(yc, rng)
        is_att = np.zeros(N_CLIENTS, bool)
        is_att[rng.choice(N_CLIENTS, N_ATTACK, replace=False)] = True
        ys = [(yc[idx] + 1) % 10 if is_att[c] else yc[idx] for c, idx in enumerate(parts)]
        W = np.stack([train_softmax(Xc[idx], ys[c]) for c, idx in enumerate(parts)])
        n_k = np.array([len(idx) for idx in parts], float)

        # ---- Baseline: cosine similarity of each update to coordinate-wise median ----
        U = W.reshape(N_CLIENTS, -1)  # update = local model minus zero global model
        med = np.median(U, axis=0)
        cos = U @ med / (np.linalg.norm(U, axis=1) * np.linalg.norm(med) + 1e-12)
        mad = np.median(np.abs(cos - np.median(cos)))
        z = (cos - np.median(cos)) / (1.4826 * mad + 1e-12)
        base_susp, base_flag = -z, z < Z_THR

        # ---- Improved: conformal label-transition audit (leave-one-client-out ensemble) ----
        P = np.stack([softmax(add_bias(Xall) @ W[j]) for j in range(N_CLIENTS)])
        S = P.sum(axis=0)
        imp_susp = np.zeros(N_CLIENTS)
        for i, idx in enumerate(parts):
            Pm = (S - P[i]) / (N_CLIENTS - 1)          # ensemble of the other clients
            q = np.quantile(1 - Pm[cal, ycal], level, method="higher")  # split-conformal threshold
            # label y is outside the 90% set {k : 1 - Pm[x,k] <= q} iff 1 - Pm[x,y] > q
            imp_susp[i] = np.mean((1 - Pm[idx, ys[i]]) > q)
        imp_flag = imp_susp > FLAG_FRAC

        runs.append(dict(
            base_f1=f1_score(is_att, base_flag, zero_division=0),
            imp_f1=f1_score(is_att, imp_flag, zero_division=0),
            base_auc=roc_auc_score(is_att, base_susp),
            imp_auc=roc_auc_score(is_att, imp_susp),
            W_all=fedavg(W, n_k, np.ones(N_CLIENTS, bool)),
            W_base=fedavg(W, n_k, ~base_flag),
            W_imp=fedavg(W, n_k, ~imp_flag),
        ))
        print(f"  flagged baseline={int(base_flag.sum())} improved={int(imp_flag.sum())} "
              f"true attackers={int(is_att.sum())}")

    # ---- Test set used once, at the end ----
    acc = {k: [accuracy(r[w], Xte, yte) for r in runs] for k, w in
           [("no_filter", "W_all"), ("baseline", "W_base"), ("improved", "W_imp")]}
    b, m = np.array([r["base_f1"] for r in runs]), np.array([r["imp_f1"] for r in runs])
    ba, ia = np.array([r["base_auc"] for r in runs]), np.array([r["imp_auc"] for r in runs])
    vals = [b.mean(), m.mean(), ba.mean(), ia.mean(), np.mean(acc["baseline"]), np.mean(acc["improved"])]
    if not np.all(np.isfinite(vals)):
        print("Non-finite metrics; aborting.")
        sys.exit(1)

    # Optional chart
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline (cosine)", "improved (conformal)"], [b.mean(), m.mean()],
           yerr=[b.std(), m.std()], color=["gray", "steelblue"], capsize=4)
    ax.set_ylabel("attacker-detection F1 (mean ± std)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("figures/attacker_f1.png", dpi=120)
    plt.close(fig)

    results = {
        "question": QUESTION,
        "method": ("Softmax-regression FedAvg over 20 Dirichlet(0.3) label-skewed clients, 3 flipping labels "
                   "y->(y+1) mod 10; baseline flags robust z<-2 on cosine(update, median update); improved "
                   "flags clients whose fraction of labels outside a 90% split-conformal set (leave-one-client-out "
                   "ensemble, calibrated on a 10% hold-out) exceeds 0.3. Averaged over 3 federation seeds."),
        "dataset": {
            "name": "scikit-learn digits (load_digits)",
            "source_url": "https://scikit-learn.org/stable/datasets/toy_dataset.html",
            "n_rows": int(len(X)), "n_train": int(len(Xp)), "n_test": int(len(Xte)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "attacker-detection F1 (label-flipping clients), higher is better",
        "higher_is_better": True,
        "baseline": "Cosine-similarity update-outlier detector: robust z-score < -2 on cosine to median update",
        "improved": ("Conformal label-transition audit: flag clients whose fraction of labels outside a 90% "
                     "leave-one-client-out conformal set exceeds 0.3"),
        "metrics": {
            "baseline_score": float(b.mean()),
            "improved_score": float(m.mean()),
            "improvement": float(m.mean() - b.mean()),
            "baseline_score_std": float(b.std()),
            "improved_score_std": float(m.std()),
            "baseline_auc": float(ba.mean()),
            "improved_auc": float(ia.mean()),
            "baseline_accuracy": float(np.mean(acc["baseline"])),
            "improved_accuracy": float(np.mean(acc["improved"])),
            "no_filter_accuracy": float(np.mean(acc["no_filter"])),
            "n_seeds": len(SEEDS),
        },
        "notes": ("Hyperparameters were fixed a priori and not tuned, so no CV selection was performed. "
                  "The test set is used only for the final FedAvg accuracy. The paper's exact method details "
                  "were not available; the improved model is a reconstruction from its name and description. "
                  "Only 3 seeds and 20 clients per seed, so detection F1 is coarse. The digits dataset is a "
                  "small stand-in, not the paper's data. Bundled data, so no download retries are needed."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["metrics"], indent=2))
    print("Wrote results.json and figures/attacker_f1.png")


if __name__ == "__main__":
    main()
