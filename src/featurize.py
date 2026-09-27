"""
Turn sequence windows into feature matrices.

Two encoders:
  onehot : 21-mer one-hot, 21 x 21 flattened. No torch needed, instant.
  esm    : mean-pooled ESM-2 embedding of the window. Needs torch + fair-esm.

Start with onehot. Only move to esm once the baseline numbers exist, so you
know what the embedding actually buys you.

Usage:
    python src/featurize.py --encoder onehot
    python src/featurize.py --encoder esm --model esm2_t12_35M_UR50D

Writes data/processed/X_<encoder>.npy and data/processed/feat_meta_<encoder>.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

AA = "ACDEFGHIKLMNPQRSTVWY-"
AA_IDX = {a: i for i, a in enumerate(AA)}


def onehot(windows: list[str]) -> np.ndarray:
    L = len(windows[0])
    X = np.zeros((len(windows), L, len(AA)), dtype=np.uint8)
    for r, w in enumerate(windows):
        for c, aa in enumerate(w):
            X[r, c, AA_IDX.get(aa, AA_IDX["-"])] = 1.0
    return X.reshape(len(windows), -1)  # uint8: 4x smaller than float32


def esm_embed(windows: list[str], model_name: str, batch_size: int = 64,
              pooling: str = "mean") -> np.ndarray:
    """pooling="mean": average over the window. pooling="centre": take only the
    embedding of the central candidate residue -- keeps positional identity,
    which mean-pooling throws away."""
    import torch
    import esm as esm_lib

    model, alphabet = getattr(esm_lib.pretrained, model_name)()
    model.eval()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"ESM device: {device}")
    model = model.to(device)

    converter = alphabet.get_batch_converter()
    layer = model.num_layers
    out = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start:start + batch_size]
            # ESM has no '-' token; strip padding, keep the real context
            data = [(str(i), w.replace("-", "")) for i, w in enumerate(chunk)]
            _, _, toks = converter(data)
            toks = toks.to(device)
            rep = model(toks, repr_layers=[layer])["representations"][layer]
            # mean-pool over real residues, excluding BOS/EOS
            mask = (toks != alphabet.padding_idx)
            mask[:, 0] = False
            for j in range(rep.shape[0]):
                m = mask[j].clone()
                if m.sum() > 1:
                    m[m.nonzero()[-1]] = False  # drop EOS
                if pooling == "mean":
                    out.append(rep[j][m].mean(0).cpu().numpy())
                else:
                    # centre residue: index within the real (unpadded) window
                    real = chunk[j].replace("-", "")
                    centre_in_real = len(chunk[j][:len(chunk[j]) // 2].replace("-", ""))
                    centre_in_real = min(centre_in_real, len(real) - 1)
                    idxs = m.nonzero().flatten()
                    out.append(rep[j][idxs[centre_in_real]].cpu().numpy())
            print(f"  {min(start + batch_size, len(windows)):,}/{len(windows):,}",
                  end="\r")
    print()
    return np.vstack(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["onehot", "esm"], default="onehot")
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pooling", choices=["mean", "centre"], default="mean")
    ap.add_argument("--residues", nargs="+", default=None,
                    help="only featurize these candidate residues, e.g. N. "
                         "Cuts ESM time and disk ~80%% when you only model one.")
    ap.add_argument("--tag", default=None,
                    help="output suffix; defaults to the encoder name")
    args = ap.parse_args()

    sites = PROC / "sites.csv"
    if not sites.exists():
        raise SystemExit(f"missing {sites} -- run parse_sites.py first")

    df = pd.read_csv(sites)
    if args.residues:
        keep = df["residue"].isin(args.residues).to_numpy()
        df = df[keep]
        print(f"filtered to residues {args.residues}: {len(df):,} of {len(keep):,}")
        np.save(PROC / f"rowmask_{args.tag or args.encoder}.npy", np.flatnonzero(keep))
    windows = df["window"].astype(str).tolist()
    print(f"{len(windows):,} windows, length {len(windows[0])}")

    if args.encoder == "onehot":
        X = onehot(windows)
        meta = {"encoder": "onehot", "alphabet": AA, "dim": int(X.shape[1])}
    else:
        X = esm_embed(windows, args.model, args.batch_size, args.pooling)
        meta = {"encoder": "esm", "model": args.model, "pooling": args.pooling,
                "residues": args.residues, "dim": int(X.shape[1])}

    tag = args.tag or args.encoder
    meta["n_rows"] = int(X.shape[0])
    np.save(PROC / f"X_{tag}.npy", X)
    (PROC / f"feat_meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    mb = X.nbytes / 1e6
    print(f"wrote X {X.shape} ({mb:,.0f} MB) -> data/processed/X_{tag}.npy")


if __name__ == "__main__":
    main()
