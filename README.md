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
