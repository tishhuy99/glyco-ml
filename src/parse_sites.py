"""
Turn UniProt CARBOHYD features into a per-residue label table.

Writes data/processed/sites.csv with one row per CANDIDATE residue:
    accession, position (1-based), residue, label, glyco_type,
    evidence, window, seq_len

Candidate residues only:
    N            -> possible N-glycosylation
    S, T         -> possible O-glycosylation
(Tyr O-glycosylation and C-mannosylation of Trp exist but are rare; excluded
 from the slice. Noted in results.md as a scope decision.)

NEGATIVE-SET POLICY -- read this before trusting any metric.
A candidate residue with no CARBOHYD annotation is labelled 0 here, but that
means "not annotated", NOT "experimentally shown to be unglycosylated".
UniProt coverage is incomplete and biased toward well-studied proteins, so
these are UNLABELED, not true negatives. The slice reports metrics on this
naive labelling to get a baseline number; the positive-unlabeled treatment
comes later. Every metric from this table must be reported with that caveat.
"""

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

WINDOW = 10  # residues either side -> 21-mer centred on the candidate
CANDIDATES = {"N", "S", "T"}

# UniProt CARBOHYD description strings look like:
#   "N-linked (GlcNAc...) asparagine"
#   "O-linked (GalNAc...) threonine"
#   "C-linked (Man) tryptophan"
LINK_RE = re.compile(r"^([NOC])-linked", re.IGNORECASE)


def classify(description: str) -> str:
    if not description:
        return "unknown"
    m = LINK_RE.match(description.strip())
    if m:
        return f"{m.group(1).upper()}-linked"
    return "unknown"


def evidence_codes(feature: dict) -> str:
    ev = feature.get("evidences") or []
    codes = sorted({e.get("evidenceCode", "") for e in ev if e.get("evidenceCode")})
    return ";".join(codes)


def extract(record: dict) -> list[dict]:
    acc = record.get("primaryAccession")
    seq = (record.get("sequence") or {}).get("value", "")
    if not seq:
        return []

    # position -> (glyco_type, evidence)
    annotated: dict[int, tuple[str, str]] = {}
    for feat in record.get("features", []):
        if feat.get("type") != "Glycosylation":
            continue
        loc = feat.get("location", {})
        start = (loc.get("start") or {}).get("value")
        end = (loc.get("end") or {}).get("value")
        if start is None or start != end:
            continue  # only point features
        annotated[int(start)] = (
            classify(feat.get("description", "")),
            evidence_codes(feat),
        )

    rows = []
    for i, aa in enumerate(seq, start=1):
        if aa not in CANDIDATES:
            continue
        lo = max(0, i - 1 - WINDOW)
        hi = min(len(seq), i + WINDOW)
        window = seq[lo:hi]
        # pad so every window is the same length
        left_pad = WINDOW - (i - 1 - lo)
        right_pad = WINDOW - (hi - i)
        window = ("-" * left_pad) + window + ("-" * right_pad)

        gtype, ev = annotated.get(i, ("", ""))
        rows.append({
            "accession": acc,
            "position": i,
            "residue": aa,
            "label": 1 if i in annotated else 0,
            "glyco_type": gtype,
            "evidence": ev,
            "window": window,
            "seq_len": len(seq),
        })
    return rows


def main() -> None:
    src = RAW / "uniprot_glyco.json"
    if not src.exists():
        raise SystemExit(f"missing {src} -- run fetch_uniprot.py first")

    records = json.loads(src.read_text())
    rows: list[dict] = []
    for rec in tqdm(records, desc="parsing"):
        rows.extend(extract(rec))

    df = pd.DataFrame(rows)
    assert df["window"].str.len().nunique() == 1, "windows are not uniform length"

    out = PROC / "sites.csv"
    df.to_csv(out, index=False)

    # also write the sequences, for the clustering step
    seqs = {r["primaryAccession"]: (r.get("sequence") or {}).get("value", "")
            for r in records}
    fasta = PROC / "sequences.fasta"
    with fasta.open("w") as fh:
        for acc, s in seqs.items():
            if s:
                fh.write(f">{acc}\n{s}\n")

    print(f"\nwrote {len(df):,} candidate residues -> {out}")
    print(f"wrote {len(seqs):,} sequences -> {fasta}")
    print("\nlabel balance by residue:")
    print(df.groupby("residue")["label"].agg(["size", "sum", "mean"]).to_string())
    print("\nglyco_type counts (positives only):")
    print(df.loc[df.label == 1, "glyco_type"].value_counts().to_string())
    print("\ntop evidence codes (positives only):")
    print(df.loc[df.label == 1, "evidence"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
