# glyco-ml

Exploratory pipeline for protein glycosylation site prediction, built to test one
question: **how much of the reported performance in this literature survives a
realistic evaluation setup?**

Published N-glycosylation site predictors report metrics like 0.83 MCC and 98%
sensitivity. Those numbers come from datasets where negatives are drawn from
*different proteins* than positives, filtered to exclude the N-X-[S/T] sequon,
and balanced ~1:1 against positives. This pipeline instead evaluates every
candidate residue in a glycoprotein at natural prevalence (~3% for experimental
N-sites), which is the task a proteome-scale predictor actually faces.

Status: exploratory. Small-scale, single-seed, not a publication.

## Findings

All numbers are 5-fold cross-validation, PR-AUC, grouped by MMseqs2 clusters at
30% sequence identity. Human reviewed UniProt entries with CARBOHYD annotations
(4,619 proteins, 513,062 candidate N/S/T residues).

| task | labels | model | PR-AUC | PSSM | sequon | prevalence |
|---|---|---|---|---|---|---|
| N-sites | all UniProt | one-hot + logreg | 0.813 | 0.773 | 0.723 | 14.5% |
| N-sites | experimental only | one-hot + logreg | 0.474 | 0.474 | 0.336 | 3.2% |
| N-sites | experimental only | ESM-2 35M + MLP (mean-pool) | 0.138 | 0.474 | 0.336 | 3.2% |
| N-sites | experimental only | ESM-2 35M + MLP (centre residue) | 0.497 | 0.475 | 0.339 | 3.2% |
| O-sites (S/T) | experimental only | one-hot + logreg | 0.021 | 0.029 | 0.002 | 0.2% |

**1. Evidence codes matter enormously.** 13,490 of 16,072 UniProt N-site
positives carry ECO:0000255 — "inferred from sequence model", i.e. annotated
*because* they match the sequon. Training on them is circular. Filtering to
experimental evidence (ECO:0000269, ECO:0007744) drops PR-AUC from 0.813 to
0.474. Any dataset built from unfiltered UniProt glycosylation annotations
inherits this.

**2. Real signal beyond the motif is modest.** On experimental labels, a
sequence model reaches 0.474 against the sequon rule's 0.336. Non-trivial, but
far from the near-perfect numbers reported on balanced benchmark sets.

**3. A linear model on one-hot windows is a PSSM.** Both scored 0.474 — to
three decimal places. Unsurprising in hindsight; the two are nearly the same
object mathematically. Beating a PSSM is the real bar, not beating the sequon.

**4. O-glycosylation is not tractable from UniProt alone.** 817 experimental
O-linked positives against 401,379 candidate S/T residues. Fold-to-fold PR-AUC
ranged 0.004–0.098 — noise. Needs glycoproteomics data (GlyConnect, GlyGen, OGP).

**5. Mean-pooling a window destroys the signal; ESM then ties the PSSM.**
Mean-pooled ESM scored 0.138 — worse than one-hot — because the pooled vector
carries no information about *which* residue is the candidate site. Switching to
the central residue's embedding (`--pooling centre`) gives 0.497. But that is a
tie with the PSSM's 0.475, not a win: ESM lost on one of five folds, and the
per-fold difference ranged −0.048 to +0.056 against a mean of +0.022, on a single
seed with ~620 positives per fold. A 21-residue window gives a language model no
more context than a PSSM has, which is the likely explanation. Untested: wider
windows (51, 101) and the 150M checkpoint.

## Pipeline

    python src/fetch_uniprot.py        # UniProt REST -> data/raw/
    python src/parse_sites.py          # CARBOHYD features -> per-residue labels
    python src/split.py                # MMseqs2 clusters -> 5 folds
    python src/featurize.py --encoder onehot --residues N --tag onehot_N
    python src/train.py --encoder onehot --tag onehot_N --residues N --experimental-only

Optional:

    python src/subsample.py --n-proteins 400   # shrink for fast iteration
    python src/subsample.py --restore          # back to full data
    python src/cleanup.py --purge              # delete feature matrices

`--experimental-only` is required for any meaningful N-site number. Without it
you are measuring the sequon against sequon-derived labels.

## Requirements

    conda create -n glyco python=3.11 -y
    conda activate glyco
    pip install numpy pandas scipy scikit-learn matplotlib seaborn biopython requests tqdm
    pip install torch fair-esm          # only for --encoder esm
    conda install -c bioconda mmseqs2   # required for non-leaky splits

Python 3.11, not 3.13 — fair-esm is unmaintained and lags new releases.

## Caveats

- **Negatives are unlabeled, not negative.** An unannotated candidate residue may
  simply never have been measured. All metrics here inherit that. Positive-
  unlabeled learning is the principled treatment ([Li et al. 2019](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2700-1)).
- Single seed, no hyperparameter search, no confidence intervals.
- `subsample.py` preferentially samples annotated proteins, so subsampled
  prevalence (~15.8% N) is inflated versus the full set (~3.2%). Smoke tests only.
- Tyr O-glycosylation and Trp C-mannosylation excluded from candidate residues.
- CoNglyPred's methods were not verified; that fetch was blocked.

## Related work

- [ESM-LoRA-Gly](https://www.biorxiv.org/content/10.1101/2025.08.12.669850v1) (2025) — ESM + LoRA for N/O sites; negatives from DeepLoc, sequon-filtered, balanced
- [StackGlyEmbed](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf146/8177143) (2025) — PLM embeddings for N-sites; 98.2% sensitivity on balanced sets
- [CoNglyPred](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/pmic.202400210) (2025) — ESM-2 + structure, graph network
- [PU learning for glycosylation sites](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2700-1) (2019) — PU framing; notes balanced-test-set limitation
- [Hong et al.](https://academic.oup.com/bioinformatics/article/41/2/btaf034/7989290) (2025) — explainable ML for O-sites

Data: UniProtKB, reviewed human entries, retrieved 2026-09-27. See
`data/raw/fetch_meta.json` for the exact query.
