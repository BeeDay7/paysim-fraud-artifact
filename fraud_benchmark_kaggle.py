#!/usr/bin/env python
# =============================================================================
# Responsible Benchmarking of AI Fraud Detection for Mobile-Money (PaySim1)
# Honest, leakage-audited, multi-seed benchmark. SAFER-TEA 2026 (Late-Breaking).
#
# HOW TO RUN (Kaggle):
#   1. New Notebook -> Add data -> search "paysim1" (ealaxi/paysim1) -> Add.
#   2. Settings -> Accelerator: GPU T4 (the tabular part runs on CPU too;
#      GPU only speeds the MLP/GNN).
#   3. Paste this whole file into ONE cell. Run.
#   4. FIRST set SMOKE=True (a ~1-2 min sanity run). Confirm it prints results
#      and that the FLOOR model's AUROC is ~0.5. THEN set SMOKE=False and run
#      the full battery.
#   5. Download /kaggle/working/fraud_bench_results.zip and send it back.
#
# HONESTY LEDGER: every number in the paper must come from summary.json /
# seed_*.json produced by THIS script. Nothing is hand-typed. If a model is not
# in the output, it is not in the paper.
# =============================================================================

import os, json, glob, time, random, warnings, zipfile
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------------
SMOKE      = True     # <-- run this FIRST (small, fast). Then set False.
N_SEEDS    = 5        # full run: 5 seeds
OUTDIR     = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
TRAIN_FRAC = 0.70     # chronological split by `step`
RUN_MLP    = True
RUN_GNN    = True     # attempted on a bounded subset; failure is non-fatal
GNN_SUBSET = 200_000  # transactions used to build the graph (chronological)
SMOKE_ROWS = 100_000

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# GNN dependency: install on Kaggle if missing (needs Internet ON in notebook settings).
# Fails fast if there is no network, then continues WITHOUT the GNN (non-fatal).
if RUN_GNN:
    try:
        import torch_geometric  # noqa: F401
    except Exception:
        print("[setup] installing torch_geometric (needs Internet ON) ...")
        os.system("pip install -q --retries 1 --timeout 10 torch_geometric")
        try:
            import torch_geometric  # noqa: F401
        except Exception:
            print("[setup] torch_geometric unavailable (Internet OFF?). The GNN will be skipped; "
                  "the tabular battery + leakage audit still run in full. To include the GNN, turn "
                  "Internet ON in Kaggle Settings; to silence this, set RUN_GNN = False.")

def set_seed(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception:
        pass

def find_csv():
    for pat in ["/kaggle/input/**/*.csv", "./*.csv", "**/*.csv"]:
        hits = glob.glob(pat, recursive=True)
        hits = [h for h in hits if os.path.getsize(h) > 10_000_000]  # the big log
        if hits:
            return sorted(hits, key=os.path.getsize)[-1]
    raise FileNotFoundError("PaySim CSV not found. Attach the ealaxi/paysim1 dataset.")

def device_report():
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[device] torch={torch.__version__} cuda_available={torch.cuda.is_available()} -> {dev}")
        if dev == "cuda":
            print("[device]", torch.cuda.get_device_name(0))
        return dev
    except Exception as e:
        print("[device] torch unavailable:", e); return "cpu"

# ----------------------------- DATA -----------------------------------------
print("=" * 70)
DEVICE = device_report()
csv = find_csv(); print("[data] using:", csv)
df = pd.read_csv(csv)
print(f"[data] rows={len(df):,} columns={list(df.columns)}")

# Fraud occurs only in TRANSFER / CASH_OUT -> restrict to fraud-eligible types.
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
df = df.sort_values("step").reset_index(drop=True)
if SMOKE:
    df = df.iloc[:SMOKE_ROWS].copy()
n_fraud = int(df["isFraud"].sum())
base_rate = df["isFraud"].mean()
print(f"[data] fraud-eligible rows={len(df):,}  frauds={n_fraud:,}  base_rate={base_rate:.5f}")

# Leakage-FREE engineered features (row-local; no target used).
df["is_TRANSFER"]     = (df["type"] == "TRANSFER").astype(float)
df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
BASE_FEATS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
              "newbalanceDest", "is_TRANSFER", "errorBalanceOrig", "errorBalanceDest"]

# Chronological split by step (never shuffle time).
steps = np.sort(df["step"].unique())
cut = steps[int(len(steps) * TRAIN_FRAC)]
tr = df[df["step"] <= cut]; te = df[df["step"] > cut]
if te["isFraud"].sum() == 0 or tr["isFraud"].sum() == 0:  # smoke fallback
    idx = int(len(df) * TRAIN_FRAC); tr = df.iloc[:idx]; te = df.iloc[idx:]
print(f"[split] train={len(tr):,} (fraud {int(tr['isFraud'].sum())})  "
      f"test={len(te):,} (fraud {int(te['isFraud'].sum())})")

ytr = tr["isFraud"].values.astype(int); yte = te["isFraud"].values.astype(int)
test_base_rate = yte.mean()

# LEAKY feature (the audit): destination fraud rate computed over ALL rows (incl. test),
# i.e. a target-derived feature that leaks test labels. Used ONLY in the histgb_LEAKY variant
# to quantify how much such leakage inflates the headline AUROC/AUPRC vs the honest baseline.
gall = df.groupby("nameDest")["isFraud"].mean()
tr_dest_leaky = tr["nameDest"].map(gall).fillna(df["isFraud"].mean()).values
te_dest_leaky = te["nameDest"].map(gall).fillna(df["isFraud"].mean()).values

Xtr = tr[BASE_FEATS].values.astype(np.float32)
Xte = te[BASE_FEATS].values.astype(np.float32)

def metrics(y, s):
    return dict(auroc=float(roc_auc_score(y, s)),
                auprc=float(average_precision_score(y, s)))

# ----------------------------- MODELS ---------------------------------------
def m_floor(seed):
    rng = np.random.RandomState(seed)
    return metrics(yte, rng.rand(len(yte)))            # random scores -> AUROC~0.5

def m_logreg(seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return metrics(yte, clf.predict_proba(sc.transform(Xte))[:, 1])

def m_histgb(seed, leaky=False):
    if leaky:
        Xt = np.column_stack([Xtr, tr_dest_leaky]); Xe = np.column_stack([Xte, te_dest_leaky])
    else:
        Xt, Xe = Xtr, Xte
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                         random_state=seed, class_weight="balanced")
    clf.fit(Xt, ytr)
    return metrics(yte, clf.predict_proba(Xe)[:, 1])

def m_mlp(seed):
    import torch, torch.nn as nn
    set_seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    Xe = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEVICE)
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], dtype=torch.float32).to(DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    epochs = 3 if SMOKE else 12; bs = 8192
    Xt, yt = Xt.to(DEVICE), yt.to(DEVICE)
    n = Xt.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xt[b]).squeeze(1)
            loss = lossf(out, yt[b])
            if not torch.isfinite(loss):      # clamp instability
                continue
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        s = torch.sigmoid(net(Xe.to(DEVICE)).squeeze(1)).cpu().numpy()
    return metrics(yte, s)

def m_gnn(seed):
    # GraphSAGE on a bounded chronological subset (accounts=nodes, txns=edges).
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    set_seed(seed)
    sub = df.iloc[:GNN_SUBSET].copy()
    accts = pd.Index(pd.unique(pd.concat([sub["nameOrig"], sub["nameDest"]])))
    aid = {a: i for i, a in enumerate(accts.tolist())}
    src = sub["nameOrig"].map(aid).values; dst = sub["nameDest"].map(aid).values
    ei = torch.tensor(np.vstack([np.concatenate([src, dst]),
                                 np.concatenate([dst, src])]), dtype=torch.long)  # undirected
    # node features: mean of incident-transaction features (train rows only for labels)
    scut = np.sort(sub["step"].unique())[int(sub["step"].nunique() * TRAIN_FRAC)]
    sub["_train"] = sub["step"] <= scut
    feat = np.zeros((len(accts), len(BASE_FEATS)), dtype=np.float32); cnt = np.zeros(len(accts))
    for col_i, f in enumerate(BASE_FEATS):
        vals = sub[f].values
        np.add.at(feat[:, col_i], dst, vals); np.add.at(cnt, dst, 1.0)
    feat = feat / np.maximum(cnt[:, None], 1.0)
    # node label = account ever a fraud destination in TRAIN rows (avoids test leakage)
    ylab = np.zeros(len(accts)); frows = sub[(sub["isFraud"] == 1) & (sub["_train"])]
    np.add.at(ylab, frows["nameDest"].map(aid).values, 1.0); ylab = (ylab > 0).astype(np.float32)
    # eval mask: accounts that are destinations in TEST rows
    test_dest = sub[~sub["_train"]]["nameDest"].map(aid).values
    yeval = np.zeros(len(accts)); ye_rows = sub[(sub["isFraud"] == 1) & (~sub["_train"])]
    np.add.at(yeval, ye_rows["nameDest"].map(aid).values, 1.0); yeval = (yeval > 0).astype(np.float32)
    from sklearn.preprocessing import StandardScaler as SS
    X = torch.tensor(SS().fit_transform(feat), dtype=torch.float32).to(DEVICE)
    ei = ei.to(DEVICE); ytr_t = torch.tensor(ylab, dtype=torch.float32).to(DEVICE)
    train_mask = torch.tensor(sub.groupby(sub["nameDest"].map(aid)).size().reindex(range(len(accts))).notna().values)
    class SAGE(nn.Module):
        def __init__(s, d):
            super().__init__(); s.c1 = SAGEConv(d, 64); s.c2 = SAGEConv(64, 1)
        def forward(s, x, e): return s.c2(F.relu(s.c1(x, e)), e).squeeze(1)
    net = SAGE(X.shape[1]).to(DEVICE)
    pw = torch.tensor([(len(ylab) - ylab.sum()) / max(ylab.sum(), 1)], dtype=torch.float32).to(DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    mtr = torch.tensor((np.isin(np.arange(len(accts)), np.concatenate([src, dst]))), dtype=torch.bool).to(DEVICE)
    for ep in range(5 if SMOKE else 40):
        net.train(); opt.zero_grad()
        out = net(X, ei); loss = lossf(out[mtr], ytr_t[mtr])
        if torch.isfinite(loss): loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        s = torch.sigmoid(net(X, ei)).cpu().numpy()
    ev = test_dest[np.isin(test_dest, np.arange(len(accts)))]
    ev = np.unique(ev)
    return metrics(yeval[ev], s[ev])

MODELS = [("floor_random", m_floor),
          ("logreg", m_logreg),
          ("histgb_base", lambda s: m_histgb(s, leaky=False)),
          ("histgb_LEAKY", lambda s: m_histgb(s, leaky=True))]
if RUN_MLP: MODELS.append(("mlp", m_mlp))
if RUN_GNN: MODELS.append(("graphsage_gnn_subset", m_gnn))

# ----------------------------- RUN ------------------------------------------
seeds = list(range(N_SEEDS if not SMOKE else 2))
per_seed = {name: [] for name, _ in MODELS}
for sd in seeds:
    for name, fn in MODELS:
        t0 = time.time()
        try:
            r = fn(sd); r["seed"] = sd; r["secs"] = round(time.time() - t0, 1)
            per_seed[name].append(r)
            print(f"[seed {sd}] {name:22s} AUROC={r['auroc']:.4f} AUPRC={r['auprc']:.4f} ({r['secs']}s)")
        except Exception as e:
            print(f"[seed {sd}] {name:22s} FAILED: {type(e).__name__}: {e}")
    # incremental save each seed
    json.dump(per_seed, open(os.path.join(OUTDIR, f"seed_progress.json"), "w"), indent=2)

def agg(rs, k):
    v = [r[k] for r in rs]
    return dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v)) if v else None

# ---- DIAGNOSTIC 1: feature-subset ablation (WHY is the tree perfect?) -------
FEATS_RAW = ["amount", "is_TRANSFER"]
FEATS_BAL = FEATS_RAW + ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FEAT_SETS = {"raw_amount_type": FEATS_RAW, "plus_balances": FEATS_BAL, "plus_errorBalance_ALL": BASE_FEATS}
ablation = {}
print("\n[ablation] HistGB with progressively richer features:")
for label, feats in FEAT_SETS.items():
    Xt = tr[feats].values.astype(np.float32); Xe = te[feats].values.astype(np.float32)
    rs = []
    for sd in seeds:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                             random_state=sd, class_weight="balanced").fit(Xt, ytr)
        rs.append(metrics(yte, clf.predict_proba(Xe)[:, 1]))
    ablation[label] = {"auroc": agg(rs, "auroc"), "auprc": agg(rs, "auprc")}
    a = ablation[label]
    print(f"   {label:24s} AUROC {a['auroc']['mean']:.4f}  AUPRC {a['auprc']['mean']:.4f}")

# ---- DIAGNOSTIC 2: the deterministic fraud signature in PaySim -------------
def frac(m): return float(np.mean(m)) if len(m) else 0.0
f = (df["isFraud"] == 1).values; nf = ~f
sig = {"n_fraud": int(f.sum()), "n_nonfraud": int(nf.sum())}
for name, col in [("newOrig_zero", df["newbalanceOrig"].values == 0),
                  ("oldOrig_eq_amount", np.abs(df["oldbalanceOrg"].values - df["amount"].values) < 1e-3),
                  ("errOrig_zero", np.abs(df["errorBalanceOrig"].values) < 1e-3),
                  ("newDest_zero", df["newbalanceDest"].values == 0)]:
    sig[f"fraud_{name}"] = frac(col[f]); sig[f"nonfraud_{name}"] = frac(col[nf])
print("[signature] deterministic fraud pattern (fraud vs non-fraud fraction):")
for k in [k for k in sig if k.startswith("fraud_")]:
    nk = "non" + k
    print(f"   {k[6:]:20s} fraud={sig[k]:.3f}  nonfraud={sig[nk]:.3f}")

summary = {"dataset": os.path.basename(csv), "smoke": SMOKE, "seeds": seeds,
           "n_test": int(len(yte)), "test_fraud_count": int(yte.sum()),
           "test_base_rate": float(test_base_rate),
           "results": {name: {"auroc": agg(rs, "auroc"), "auprc": agg(rs, "auprc")}
                       for name, rs in per_seed.items()},
           "ablation": ablation, "fraud_signature": sig}
json.dump(summary, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
json.dump(per_seed, open(os.path.join(OUTDIR, "per_seed.json"), "w"), indent=2)

print("\n" + "=" * 70 + "\nSUMMARY (mean +/- std over seeds)")
for name, r in summary["results"].items():
    a, p = r["auroc"], r["auprc"]
    if a: print(f"  {name:22s} AUROC {a['mean']:.4f}+/-{a['std']:.4f}   AUPRC {p['mean']:.4f}+/-{p['std']:.4f}")
print(f"\n[control] floor AUROC should be ~0.5 (chance). test base rate (AUPRC floor)={test_base_rate:.5f}")
print("RESULT_JSON:", json.dumps(summary))

with zipfile.ZipFile(os.path.join(OUTDIR, "fraud_bench_results.zip"), "w") as z:
    for f in ["summary.json", "per_seed.json", "seed_progress.json"]:
        p = os.path.join(OUTDIR, f)
        if os.path.exists(p): z.write(p, f)
print("\nSaved -> fraud_bench_results.zip  (download this and send it back)")
