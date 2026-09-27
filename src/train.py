"""
Train and evaluate site prediction, grouped by homology cluster.

Reports PR-AUC (not ROC-AUC -- the classes are very imbalanced, so ROC-AUC
flatters every model here) against two baselines:
  sequon : the N-X-S/T rule. For N sites this is the bar to beat.
  pssm   : position-specific scoring matrix from the training fold only.

Usage:
    python src/train.py --encoder onehot --residues N
    python src/train.py --encoder onehot --residues S T
    python src/train.py --encoder esm --residues S T --model mlp
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 0
EXPERIMENTAL = ("ECO:0000269", "ECO:0007744")


def sequon_score(windows: np.ndarray, centre: int) -> np.ndarray:
    """N-X-S/T with X != P. Returns 1.0 where the motif matches."""
    out = np.zeros(len(windows), dtype=np.float32)
    for i, w in enumerate(windows):
        if centre + 2 >= len(w):
            continue
        if w[centre] != "N":
            continue
        x, s = w[centre + 1], w[centre + 2]
        if x != "P" and s in ("S", "T") and x != "-":
            out[i] = 1.0
    return out


def pssm_score(train_windows, train_y, test_windows) -> np.ndarray:
    AA = "ACDEFGHIKLMNPQRSTVWY-"
    idx = {a: i for i, a in enumerate(AA)}
    L = len(train_windows[0])
    pos = np.ones((L, len(AA)))  # Laplace smoothing
    neg = np.ones((L, len(AA)))
    for w, y in zip(train_windows, train_y):
        tgt = pos if y == 1 else neg
        for c, aa in enumerate(w):
            tgt[c, idx.get(aa, idx["-"])] += 1
    lo = np.log(pos / pos.sum(1, keepdims=True)) - np.log(neg / neg.sum(1, keepdims=True))
    return np.array([
        sum(lo[c, idx.get(aa, idx["-"])] for c, aa in enumerate(w))
        for w in test_windows
    ], dtype=np.float32)


def make_model(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=SEED)
    if kind == "mlp":
        # small net on purpose: this is a draft-scale project, and a big
        # MLP on a few thousand positives overfits anyway
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=200,
                             early_stopping=True, random_state=SEED)
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["onehot", "esm"], default="onehot")
    ap.add_argument("--tag", default=None,
                    help="feature tag to load, matching featurize.py --tag")
    ap.add_argument("--residues", nargs="+", default=["S", "T"],
                    help="candidate residues to model, e.g. N  or  S T")
    ap.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--glyco-type", default=None,
                    help="restrict positives to this link type, e.g. O-linked")
    ap.add_argument("--experimental-only", action="store_true",
                    help="keep only positives with experimental evidence "
                         "(ECO:0000269 / ECO:0007744). ECO:0000255 is "
                         "sequence-model inference -- for N-sites that means "
                         "the label WAS the sequon, so training on it is "
                         "circular. Use this for any reportable N-site number.")
    args = ap.parse_args()

    sites = pd.read_csv(PROC / "sites.csv")
    splits = pd.read_csv(PROC / "splits.csv")
    ftag = args.tag or args.encoder
    X_all = np.load(PROC / f"X_{ftag}.npy", mmap_mode="r")

    sites["_row"] = np.arange(len(sites))

    # a filtered featurization covers only some rows of sites.csv; the mask maps
    # feature-matrix row i -> sites.csv row mask[i]
    mask_path = PROC / f"rowmask_{ftag}.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
        if len(mask) != len(X_all):
            raise SystemExit(f"rowmask ({len(mask):,}) != X rows ({len(X_all):,}) "
                             "-- rerun featurize.py")
        sites = sites.iloc[mask].copy()
        sites["_row"] = np.arange(len(sites))
        print(f"using filtered featurization: {len(sites):,} rows")
    elif len(X_all) != len(sites):
        raise SystemExit(f"X has {len(X_all):,} rows but sites.csv has "
                         f"{len(sites):,} and no rowmask found -- rerun featurize.py")
    df = sites.merge(splits[["accession", "cluster", "fold"]], on="accession",
                     how="inner")

    df = df[df["residue"].isin(args.residues)]

    if args.experimental_only:
        ev = df["evidence"].fillna("").astype(str)
        has_exp = ev.apply(lambda s: any(c in s for c in EXPERIMENTAL))
        n_before = int(df["label"].sum())
        df = df.copy()
        # a positive without experimental evidence is dropped entirely, not
        # demoted to negative -- we do not know it is unglycosylated
        df = df[(df["label"] == 0) | has_exp]
        print(f"experimental-only: {n_before:,} -> {int(df['label'].sum()):,} positives")

    if args.glyco_type:
        # positives of another link type become negatives, not positives
        df = df.copy()
        df["label"] = ((df["label"] == 1) &
                       (df["glyco_type"] == args.glyco_type)).astype(int)

    if df["label"].sum() < 20:
        raise SystemExit(f"only {df['label'].sum()} positives -- too few to model")

    X = X_all[df["_row"].to_numpy()].astype(np.float32, copy=False)
    y = df["label"].to_numpy()
    windows = df["window"].astype(str).to_numpy()
    folds = df["fold"].to_numpy()
    centre = len(windows[0]) // 2

    print(f"{len(df):,} residues ({args.residues}), {y.sum():,} positives "
          f"({y.mean():.2%}), {df['cluster'].nunique():,} clusters")
    print(f"feature matrix in use: {X.nbytes / 1e6:,.0f} MB")

    rows = []
    for f in sorted(np.unique(folds)):
        tr, te = folds != f, folds == f
        if y[te].sum() == 0 or y[tr].sum() == 0:
            print(f"fold {f}: skipped (no positives on one side)")
            continue

        scaler = StandardScaler(copy=False).fit(X[tr])
        model = make_model(args.model)
        model.fit(scaler.transform(X[tr]), y[tr])
        p_model = model.predict_proba(scaler.transform(X[te]))[:, 1]

        p_sequon = sequon_score(windows[te], centre)
        p_pssm = pssm_score(windows[tr], y[tr], windows[te])

        rows.append({
            "fold": int(f),
            "n_test": int(te.sum()),
            "n_pos": int(y[te].sum()),
            "prevalence": float(y[te].mean()),
            f"prauc_{args.model}": average_precision_score(y[te], p_model),
            "prauc_pssm": average_precision_score(y[te], p_pssm),
            "prauc_sequon": average_precision_score(y[te], p_sequon),
            f"rocauc_{args.model}": roc_auc_score(y[te], p_model),
        })
        print(f"fold {f}: PR-AUC model={rows[-1][f'prauc_{args.model}']:.3f} "
              f"pssm={rows[-1]['prauc_pssm']:.3f} "
              f"sequon={rows[-1]['prauc_sequon']:.3f} "
              f"(prevalence {rows[-1]['prevalence']:.2%})")

    res = pd.DataFrame(rows)
    print("\nmean across folds:")
    print(res.drop(columns=["fold"]).mean().to_string())

    tag = f"{ftag}_{args.model}_{''.join(args.residues)}"
    if args.glyco_type:
        tag += f"_{args.glyco_type}"
    res.to_csv(RESULTS / f"cv_{tag}.csv", index=False)

    run = {
        "tag": tag,
        "encoder": args.encoder,
        "model": args.model,
        "residues": args.residues,
        "glyco_type": args.glyco_type,
        "experimental_only": bool(args.experimental_only),
        "seed": SEED,
        "split_method": splits["method"].iloc[0] if "method" in splits else "unknown",
        "n_residues": int(len(df)),
        "n_positives": int(y.sum()),
        "mean_prauc_model": float(res[f"prauc_{args.model}"].mean()),
        "mean_prauc_pssm": float(res["prauc_pssm"].mean()),
        "mean_prauc_sequon": float(res["prauc_sequon"].mean()),
        "run_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / f"run_{tag}.json").write_text(json.dumps(run, indent=2))
    print(f"\nwrote results/cv_{tag}.csv and results/run_{tag}.json")

    if "FALLBACK" in str(run["split_method"]):
        print("\n*** These numbers used the leaky fallback split. "
              "Install MMseqs2 and rerun split.py before believing them. ***")


if __name__ == "__main__":
    main()
