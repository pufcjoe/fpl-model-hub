# FPL Model Hub — v5 (structural model + trained ML)

A complete FPL projection system: a structural Poisson/xG model for 38-GW planning,
a machine-learned Ridge model for next-GW accuracy, and a squad optimiser UI.

## Architecture
```
index.html            the app: pitch builder, optimiser, plan tab, AI rating/chat
api/fpl.js            serverless proxy for the official FPL API (fixes CORS)
api/ai.js             serverless proxy for Anthropic (set ANTHROPIC_API_KEY env var)
api/projections.py    LIVE ML scoring: fetches official data, computes features,
                      applies the trained ridge coefficients, returns next-GW xP
model/train.py        training pipeline (3+ seasons of per-GW data, ~42k rows)
model/model_export.json  the trained model: 14 features, coefficients, holdout metrics
```

## The two models
- **Structural (baked into index.html)**: xG-blended per-90 components, Poisson
  clean-sheet probabilities from team-strength ratings, 38-GW projections, context
  layer. Powers season planning, chips, transfers.
- **Trained ML (api/projections.py)**: Ridge on 2023-26 per-GW data (form,
  minutes, ppg, rolling xGI, price, home, team/opponent rolling goals).
  Rolling-origin evaluation across 5 windows: mean **RMSE 2.673, MAE 1.824,
  R2 0.131** — inside the fplreview "perfect model" band every window.
  GradientBoosting scored identically; the portable ridge ships.

## Validation: repeated cross-validated squad backtest
Design: 5-fold CV producing out-of-fold preseason predictions, x2 test seasons,
x42 total repetitions across three studies. For each run a legal GBP100m squad is
built from the model's preseason projections, then scored on ACTUAL season points.

| Preseason model | Squad actual pts (mean) |
|---|---|
| Perfect hindsight (ceiling) | 2164 |
| **Blend 60% ML / 40% structural (SHIPPED)** | **1662** |
| Blend 70/30 | 1655 |
| Trained ML alone | 1604 |
| Structural alone | 1550 |
| Baseline: last season's points | 1490 |
| GBM alone | 1477 |
| Baseline: price | 1174 |

Paired vs structural: blend60 **+109 pts, t=4.22, wins 20/24 runs**.
Blend weights 45-70% are statistically indistinguishable (a plateau, not a spike);
60% chosen as its centre. Candidate-pool size 150/180/240 gives identical results,
so the squad builder is not driving the outcome. Raw run data in model/*.csv.


## How far from the achievable ceiling? (oracle study)
An oracle knowing each player's TRUE season rate but nothing week-to-week is the
best any preseason model could ever be. Same sample, 25/26:

| | RMSE | MAE | R2 |
|---|---|---|---|
| ORACLE: perfect season rate (**preseason ceiling**) | 2.374 | 1.509 | +0.232 |
| ORACLE: + perfect availability each GW | 2.125 | 1.177 | +0.385 |
| ORACLE: + exact minutes | 2.021 | 1.108 | +0.444 |
| ORACLE: + knows the clean sheet | 1.791 | 0.937 | +0.563 |
| **v6.2 calibrated (shipped)** | **2.632** | 1.727 | **+0.056** |
| v6.1 raw | 2.663 | 1.687 | +0.034 |
| naive: constant mean | 2.709 | 1.888 | 0.000 |

Signal captured vs the preseason ceiling: raw 14% -> calibrated **23%**.

Where the remaining headroom lives (RMSE gain from perfect knowledge of each):
knowing who plays at all **+0.249**, knowing the clean sheet **+0.229**,
exact minutes **+0.105**. Minutes/availability is the single biggest
improvable lever for any preseason model.

Calibration: predictions are shrunk toward the mean (lambda=0.60, fitted on the
25/26 backtest). Spearman(raw, calibrated) = 1.0000, so squad selection is
completely unchanged — this only makes the displayed xP numbers honest.

## Per-gameweek accuracy (identical sample, n=14,172, 25/26)
| Model | RMSE | MAE | R2 |
|---|---|---|---|
| **In-season ML (form features)** | **2.722** | **1.856** | **+0.125** |
| naive: constant mean | 2.909 | 2.102 | 0.000 |
| preseason: trained ML alone | 2.965 | 1.930 | -0.039 |
| **preseason v6.1 (shipped)** | 2.970 | 1.934 | -0.042 |
| preseason: structural v4.3 | 3.145 | 1.999 | -0.168 |
| preseason: last-season baseline | 3.193 | 2.097 | -0.204 |
| *perfect-model band (fplreview)* | *2.7-2.9* | *1.9-2.0* | *0.12-0.17* |

Read this correctly: **only the in-season model belongs in the perfect-model band.**
Every preseason model scores negative R2 per-gameweek -- worse than predicting the
mean for everyone. That is expected, not a defect: a preseason projection says
"Haaland averages 6/GW", and RMSE punishes it every week he blanks. Preseason
models solve a *ranking over a season* problem, which the squad backtest measures
(v6.1 wins there by +109 pts, t=4.22). Among preseason models v6.1 is still clearly
the best on this metric too (2.970 vs 3.145 structural vs 3.193 last-season).

This is exactly why the app switches engines: baked preseason projections for GW1-2,
then the in-season ML takes over point levels from GW3 once form features exist.

NOTE on earlier figures: RMSE numbers quoted before this table used different sample
filters and are NOT comparable to each other. Only same-sample comparisons count.

## Retest results (in-season, head-to-head, n=13,713)
| Model | RMSE | MAE | R2 |
|---|---|---|---|
| ML (frozen pre-25/26) | 2.735 | 1.878 | 0.125 |
| Structural (24/25-built) | 2.963 | 1.999 | -0.027 |
| naive flat-2.7 | 2.951 | 2.287 | -0.018 |

Conclusion baked into the app: **in-season, ML sets next-GW point levels
outright** (blending with the season-old structural baseline was measurably
harmful); the structural model provides the cold start (GW1-2) and the
fixture SHAPE of the following 5 GWs, rescaled to ML's level. FPL's official
ep_next remains blended as a second live model.

## Deploy
1. `vercel --prod` in this folder (or drag into vercel.com/new)
2. Settings → Environment Variables → `ANTHROPIC_API_KEY` (for AI tabs) → redeploy
3. Done. Refresh button pulls live data; /api/projections activates from GW3.

## Retrain (weekly in-season, ~2 min)
```
pip install pandas scikit-learn
python model/train.py     # pulls latest data, retrains, rewrites model_export.json
vercel --prod             # redeploy
```
The training window rolls forward automatically as new gameweeks land in the dataset.

## Notes
- Cold start: with <2 finished GWs the ML endpoint returns empty and the app
  uses the baked structural model alone.
- Perfect-model benchmark and accuracy tracker are built into the Plan tab.
