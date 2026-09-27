"""
Shrink the dataset for exploratory runs.

Draft mode: you do not need 4,619 proteins to see whether a signal exists.
This samples whole proteins (never individual residues -- splitting a protein
across the sample would break the homology-aware splits) and rewrites the
processed files in place, keeping a backup of the full versions.

    python src/subsample.py --n-proteins 400
    python src/subsample.py --restore        # put the full data back

Footprint at 400 proteins: roughly 45k candidate residues, ~20 MB of features
with the uint8 one-hot, ~90 MB for ESM. Trains in seconds.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FULL = PROC / "full"


def backup() -> None:
    FULL.mkdir(exist_ok=True)
    for name in ("sites.csv", "sequences.fasta", "splits.csv"):
        src = PROC / name
        dst = FULL / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"backed up {name} -> data/processed/full/")


def restore() -> None:
    if not FULL.exists():
        raise SystemExit("no backup at data/processed/full/")
    for f in FULL.glob("*"):
        shutil.copy2(f, PROC / f.name)
        print(f"restored {f.name}")
    # feature matrices no longer match the restored row count
    for x in PROC.glob("X_*.npy"):
        x.unlink()
        print(f"removed stale {x.name}")
    for m in PROC.glob("rowmask_*.npy"):
        m.unlink()
    print("\nrerun featurize.py before training")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-positives", type=int, default=1,
                    help="only keep proteins with at least this many annotated sites")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.restore:
        restore()
        return

    src = FULL / "sites.csv" if (FULL / "sites.csv").exists() else PROC / "sites.csv"
    if not src.exists():
        raise SystemExit("run parse_sites.py first")
    backup()

    sites = pd.read_csv(FULL / "sites.csv")

    # prefer proteins that actually carry annotations -- sampling at random
    # would mostly give you proteins with zero positives
    pos_per_prot = sites.groupby("accession")["label"].sum()
    eligible = pos_per_prot[pos_per_prot >= args.min_positives].index.to_numpy()
    if len(eligible) == 0:
        raise SystemExit("no proteins meet --min-positives")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_proteins, len(eligible))
    chosen = set(rng.choice(eligible, size=n, replace=False).tolist())

    sub = sites[sites["accession"].isin(chosen)].copy()
    sub.to_csv(PROC / "sites.csv", index=False)

    # matching FASTA
    full_fasta = (FULL / "sequences.fasta").read_text().splitlines()
    kept = []
    for i in range(0, len(full_fasta) - 1, 2):
        acc = full_fasta[i][1:].strip()
        if acc in chosen:
            kept.append(full_fasta[i])
            kept.append(full_fasta[i + 1])
    (PROC / "sequences.fasta").write_text("\n".join(kept) + "\n")

    # stale features/splits now have the wrong row count
    for x in list(PROC.glob("X_*.npy")) + list(PROC.glob("rowmask_*.npy")):
        x.unlink()
    if (PROC / "splits.csv").exists():
        (PROC / "splits.csv").unlink()

    print(f"\nsubsampled to {n:,} proteins ({len(sub):,} candidate residues, "
          f"{int(sub['label'].sum()):,} positives)")
    print("by residue:")
    print(sub.groupby("residue")["label"].agg(["size", "sum", "mean"]).to_string())
    print("\nfull data kept in data/processed/full/ -- src/subsample.py --restore")
    print("\nnext: python src/split.py")


if __name__ == "__main__":
    main()
