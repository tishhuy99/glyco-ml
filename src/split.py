"""
Homology-aware train/test splits.

Random splits leak: homologous proteins land on both sides and the model
scores well by recognising the family, not the glycosylation signal. This is
the single most common flaw in the site-prediction literature. Every metric
we report must come from a split made here.

Strategy:
  1. If MMseqs2 is on PATH, cluster sequences at --min-seq-id (default 0.30)
     and split by cluster.
  2. Otherwise fall back to splitting by accession with a warning. The
     fallback is NOT publication-grade -- it is there so the pipeline runs
     end to end before you install MMseqs2.

Writes data/processed/splits.csv : accession, cluster, fold
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def cluster_with_mmseqs(fasta: Path, min_seq_id: float, coverage: float) -> pd.DataFrame:
    tmp = Path(tempfile.mkdtemp(prefix="mmseqs_"))
    db = tmp / "db"
    clu = tmp / "clu"
    tsv = tmp / "clu.tsv"

    def run(cmd: list[str]) -> None:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    run(["mmseqs", "createdb", str(fasta), str(db)])
    run([
        "mmseqs", "cluster", str(db), str(clu), str(tmp / "scratch"),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",
        "--cluster-mode", "0",
    ])
    run(["mmseqs", "createtsv", str(db), str(db), str(clu), str(tsv)])

    pairs = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "member"])
    return pairs.rename(columns={"member": "accession", "rep": "cluster"})


def cluster_fallback(fasta: Path) -> pd.DataFrame:
    accs = [ln[1:].strip() for ln in fasta.read_text().splitlines() if ln.startswith(">")]
    print("WARNING: MMseqs2 not found. Each sequence is its own cluster -- "
          "homologues WILL leak across folds. Do not report these numbers.")
    return pd.DataFrame({"accession": accs, "cluster": accs})


def assign_folds(clusters: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uniq = np.asarray(clusters["cluster"].unique(), dtype=object)
    rng.shuffle(uniq)
    fold_of = {c: i % n_folds for i, c in enumerate(uniq)}
    clusters = clusters.copy()
    clusters["fold"] = clusters["cluster"].map(fold_of)
    return clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seq-id", type=float, default=0.30)
    ap.add_argument("--coverage", type=float, default=0.50)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fasta = PROC / "sequences.fasta"
    if not fasta.exists():
        raise SystemExit(f"missing {fasta} -- run parse_sites.py first")

    if mmseqs_available():
        print(f"clustering with MMseqs2 at {args.min_seq_id:.0%} identity ...")
        clusters = cluster_with_mmseqs(fasta, args.min_seq_id, args.coverage)
        method = f"mmseqs2 min-seq-id={args.min_seq_id} cov={args.coverage}"
    else:
        clusters = cluster_fallback(fasta)
        method = "FALLBACK per-accession (leaky)"

    clusters = clusters.drop_duplicates("accession")
    out = assign_folds(clusters, args.n_folds, args.seed)
    out["method"] = method
    out.to_csv(PROC / "splits.csv", index=False)

    print(f"\n{out['accession'].nunique():,} sequences in "
          f"{out['cluster'].nunique():,} clusters")
    print(f"method: {method}")
    print("\nsequences per fold:")
    print(out.groupby("fold").size().to_string())


if __name__ == "__main__":
    main()
