# Experiment log

5-fold CV, PR-AUC, grouped by MMseqs2 clusters at 30% identity. Seed 0.
Full dataset: 4,619 human reviewed UniProt proteins, 513,062 candidate residues.

| date | tag | labels | model | PR-AUC | PSSM | sequon | prev. | note |
|---|---|---|---|---|---|---|---|---|
| 2026-09-27 | onehot_logreg_N | all UniProt | logreg | 0.813 | 0.773 | 0.723 | 14.5% | circular: 84% of positives are ECO:0000255 |
| 2026-09-27 | onehot_logreg_N | experimental | logreg | 0.474 | 0.474 | 0.336 | 3.2% | headline honest number; ties PSSM exactly |
| 2026-09-27 | onehot_logreg_ST | all UniProt | logreg | 0.031 | 0.031 | 0.004 | 0.46% | at noise floor |
| 2026-09-27 | onehot_logreg_ST_O-linked | experimental | logreg | 0.021 | 0.029 | 0.002 | 0.20% | 817 positives; folds ranged 0.004-0.098 |
| 2026-09-27 | esm_mlp_N | experimental | ESM-2 35M + MLP, mean-pool | 0.138 | 0.474 | 0.336 | 3.2% | INVALID -- mean-pooling destroys position |
| 2026-09-27 | esm_N_centre_mlp_N | experimental | ESM-2 35M + MLP, centre residue | 0.497 | 0.475 | 0.339 | 3.2% | ties PSSM; fold diff ranged -0.048 to +0.056 |

## Open

- ESM-2 35M with centre-residue pooling reaches 0.497 vs PSSM 0.475 -- a tie,
  not an improvement. ESM lost on fold 1 (0.481 vs 0.529) and the spread of the
  per-fold difference (-0.048 to +0.056) exceeds the mean difference (+0.022).
  Single seed, ~620 positives per fold. Do not report as a win.
- Two untested explanations: (a) the 21-mer window denies ESM the long-range
  context that is its whole advantage -- try 51 or 101; (b) the 35M checkpoint
  is the smallest ESM-2 -- try 150M.
- O-sites need glycoproteomics data (GlyConnect / GlyGen / OGP), not UniProt.
- No confidence intervals anywhere. Single seed.
