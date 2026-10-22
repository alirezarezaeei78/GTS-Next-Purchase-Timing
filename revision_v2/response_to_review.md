# GTS: technical revision and evidence map

This report addresses the supplied technical review. It records a new clean study, not a journal submission or a claim that every scientific concern has disappeared. The article reports only the new cohorts and results. The pre-revision source/PDF backup is preserved.

## New experiment record

All three public sources completed the five declared seeds. The numerical gate `results/results_ready_for_paper.json` was produced only after main runs, structural controls, seed-42 sensitivity, five-horizon runs, statistical aggregation, saved-model live inference, six result-figure families and independent numerical checks. All gate file hashes were independently checked again before writing the paper. Smoke outputs are excluded.

Daily complete test counts are 627 Online Retail II and 1808 Ta-Feng customers. Instacart has 204118 completed item-pair origins from 53515 original customers; eligibility retained 578720 pairs from 76723 customers using training history only. These are not the old 103855-pair cohort. Primary selected-GTS MAEs are 11.334860, 2.119893 and 6.498008, respectively.

## Issue-by-issue response

1. **Missing IDs and returns (R01).** Missing customer IDs are dropped before conversion and grouping; no pooled artificial customer enters fitting. Positive quantities/prices and cancellation handling are applied in preparation. The same positive-day source supports the five-horizon task. Main models were refitted; deleting a bad test row from old predictions is not the remedy used here. See Experiments and `prepare.py`, `verify_protocol.py`, `pretraining_audit.json`.

2. **Temporal boundaries (R02).** Daily datasets use shared global date cuts. Instacart pairs share the original customer's order-phase boundaries, including order comparisons when timestamps tie. Calendar confirmation is not claimed for Instacart. Selection of eligibility uses only the training prefix. All causal training origins slide. See Prediction task and the 15-row phase-count table.

3. **Separate tuning and stacking (R03).** The five phases separate fitting, early stopping, input decisions, stack-weight fitting and final outcomes. A sparse simplex MAE linear program replaces threshold-based stacking. No 40/400-row rule or 0.5-percent switch remains. The stack optimum is finite-sample, not a guaranteed test optimum. See Equation (7), Algorithm 1 and per-seed stack records.

4. **Ranking and redundancy (R04).** Training-only, equal-original-customer weighted mutual information replaces simple correlation prefixes. Greedy relevance minus mean redundancy ranks variables after exact duplicate/constant removal. This is a bounded heuristic with quantile-bin estimation; local collinearity can remain. See Ranking and bounded selection, `feature_ranking.json` and public cohort/ranking files.

5. **Hand-selected dimensions (R05).** Actual encoder trials select numerical channel count, L and optional K on every dataset and seed. Fourteen or fifteen distinct trials use the full training cohort and separate early/selection samples. No tree probe or reduced research cohort substitutes for the encoder. Availability channels remain structural safeguards. The manually defined candidate formulas and short trial budget are explicitly acknowledged. See the selected-configuration table and public `input_search/` traces.

6. **Undefined zero values and duplicated profiles (R06).** Availability and valid-row flags distinguish initial undefined gaps, real zero waits and padding. Rolling observed counts expose incomplete windows. K-means fits one original-customer training profile rather than replicating frequent windows/pairs. Missingness, causal feature mutation and zero-day likelihood gradient tests passed. See Numerical histories, Feature definitions and regression tests.

7. **Graph modes, signs and coverage (R07).** Every component-constant graph mode is excluded; signs are canonicalized and actual fitted vectors saved. Repeated-eigenvalue bases remain nonunique. Coverage reports separately count source baskets and forecast origins. All 12 P/d settings are executed per dataset at seed 42. No monotonic benefit from a larger graph is claimed. See Fixed co-purchase representation, coverage table, graph figure and full grid tables.

8. **Ablations and component value (R08).** Five seeds support zero graph, random graph, no attention, last pooling and no-group controls. Encoder-replacement controls refit the encoder and stack while other members stay fixed; no-group changes all inputs and refits the full system. Encoder removal, nonrecurrent combination and direct-only controls are included. Small/inconsistent effects are reported, including random vectors slightly improving Instacart. No causal graph-value claim follows from a large encoder weight. See Structural controls and weight figure.

9. **Fair direct/residual choice (R09).** Encoder, numerical LSTM and numerical GRU all fit direct/residual candidates and allow their average. Independent selection chooses each mode. Direct-only Ta-Feng performs better on test; it remains a diagnostic and does not replace the primary model after test inspection. Exact simplex fitting has an independent regression test.

10. **Loss and clipping (R10).** Primary training minimizes uncapped original-day MAE. Log output guards prevent numerical overflow, not target quantile trimming. The full declared tau-by-cap grid refits seed-42 encoders and stack weights, with uncapped test labels and all rows shown. These are single-seed diagnostics, not five-seed complete-system objective retuning. Mean/median differences and remaining low prediction levels are both discussed.

11. **Distributional comparators (R11).** Five-seed DeepAR-style duration and neural Weibull AFT adaptations include incomplete training waits. Zero-day and survival gradients are tested. A matched small NN is trained with MAE and MSE. The paper calls these adaptations, not exact source-paper reproductions or a comprehensive SOTA comparison. TFT, RMTPP, THP and other event architectures remain untested limitations.

12. **Clean matched multi-horizon comparison (R12).** Returns and timestamp-remainder targets are removed. Direct and recursive Transformers share initial windows, outcomes, training budget and runtime, with dropout disabled at inference and no smoothing. Recursive history uses predicted gaps and holds unavailable basket attributes. Five separate MAE/RMSE/MedAE tables cover 69 identical complete test origins. Small early/selection/stack cohorts (34/38/35) are prominently disclosed. No adjusted evidence of GTS superiority over Transformers is claimed.

13. **Inference and multiple comparisons (R13).** Two thousand original-customer paired bootstraps produce differences and intervals for MAE, RMSE and MedAE. Paired sign flips use 9999 draws for MAE/MSE; Holm covers all 108 primary tests and a separate 40-test horizon family. No inappropriate time-series DM test on arbitrarily ordered customer rows is used. Bootstrap intervals are not labeled simultaneous. Seed variation is separate and limited to five runs.

14. **Cost and business value (R14).** Recorded search, member fitting, exact stacking, artifact sizes, parameters and five-repeat live batch inference are reported. Reconstructed GTS predictions match saved forecasts. The benchmark evaluates all members, including zero-weight ones, and excludes raw feature/graph construction. Batch averages are not service latency. Early/late penalties are hypothetical, not real campaign profit. Simpler alternatives remain substantive recommendations where complexity is unsupported.

15. **Plots and Instacart (R15).** All applicable three-source result plots contain Instacart: model comparison, distributions, prediction groups, weights and graph diagnostics. Five-horizon plots are explicitly Online Retail II only. New overview and encoder diagrams match the actual revised method; an editable diagrams.net file is importable into Lucidchart, without claiming native Lucidchart creation. Plot labels and heatmap contrast were visually checked.

16. **Originality, wording and citations (R16).** The entire article was rewritten in readable English with mathematical symbols defined near their use. Its contribution is controlled, reproducible evaluation; established components are not presented as new primitives. Different primary winners and adverse findings are retained. Forty-five numbered citations are clickable, including 12 ESWA papers and the prior IEEE 10903646. DOI/title/author metadata for the ESWA papers and prior IEEE work were rechecked against Crossref records. The four human authors and correspondence details remain supplied by the user. No journal acceptance or universal superiority is promised.

17. **Irrecoverable time and external confirmation (R17).** Instacart's capped relative clock cannot recover true dates or uncapped elapsed time. Point evaluation remains conditional on completed waits. A genuinely later independent calendar deployment study has not been performed. These are remaining scientific limitations, not completed fixes. The code and text explicitly preserve them.

## Main conclusions changed by the evidence

Boosting has lower primary Online Retail II MAE than selected GTS (10.940189 versus 11.334860). Fixed GTS beats selected GTS on Ta-Feng (2.090168 versus 2.119893). DeepAR leads Instacart MAE/RMSE (6.489866/9.389704 versus 6.498008/9.780180); GTS has lower MedAE. Input search has no universal accuracy or stability advantage. Graph/attention controls do not establish consistent practical gains. These findings are reported rather than hidden or used to select a new test winner.

## Reproduction and publication boundary

The public package adds a strict allowlist of clean code, aggregate metrics, counts, feature rankings and input-search traces. Raw transaction records, per-origin identifiers, model weights, credentials and personal machine paths are excluded from the new revision. Historical earlier files remain explicitly labeled as provenance. The public aggregate verifier checks hashes and consistency; full prediction/model verification requires local reproduction. This distinction prevents an aggregate-only package from claiming to have independently recomputed absent row-level outcomes.

The manuscript and response package are prepared for the authors' review. No journal submission or communication to other people has been made.
