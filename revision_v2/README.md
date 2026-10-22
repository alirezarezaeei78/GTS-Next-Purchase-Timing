# Clean revision experiments

This revision addresses the technical review with new executions. The old paper
and archived predictions remain unchanged until every numerical section is done.
No old score is attributed to this implementation.

## Protocol

`protocol.json` fixes five phases: model fitting, early stopping, input selection,
simplex-weight fitting, and test. Calendar datasets share global date cutoffs.
Instacart shares original-customer order cutoffs across every product pair; it has
relative recorded time and is never described as a calendar-date experiment.
Eligibility is decided only from the training prefix. All causal sliding training
origins are used. The primary fit is unclipped MAE in original day units.

The feature controller uses training-only mutual information with a redundancy
penalty and actual temporal-encoder trials. Missing-gap flags distinguish missing
values from true zero-day intervals. Grouping fits one profile per original
training customer. Graph construction uses only training baskets, excludes every
component-constant eigenmode, fixes vector signs, and records representation
coverage for all graph sizes and phases. Deterministic signs do not imply a unique
basis inside a repeated eigenspace; the actual saved vectors identify this fit.

LSTM, GRU, and the graph encoder all receive direct/residual/average selection on
the input-selection phase. Independent stack data fit globally optimal simplex
MAE weights with HiGHS. No heuristic 40/400-row or 0.5-percent switch is used.

Main comparisons and structural ablations use five seeds. Product-count/dimension
and log-pinball/clipping grids are explicitly seed-42 diagnostics. Each diagnostic
refits the encoder and full-system weights while holding the other fitted members
fixed; it is not a complete five-member objective retuning experiment.

The probabilistic recurrent comparator is a DeepAR-style adaptation using a
zero-truncated log1p-normal duration distribution. The neural Weibull AFT uses
whole-day event probabilities and the survival likelihood for incomplete waits.
Neither is represented as a bit-identical reproduction of another publication.
The small NN is an eight-unit matched adaptation evaluated with MAE and MSE.

## Run

Python 3.12, PyTorch 2.2.2, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1,
scikit-learn 1.4.2, PyArrow 14.0.2 and Matplotlib are used. Install the PyTorch
build appropriate to the machine. A common batch size of 4,096 was selected by a
training-only hardware throughput check before primary fits; the same budget
applies to every dataset and neural family. Model weights and raw records stay
outside the public code folder.

```sh
python prepare.py --data-root /path/to/public-datasets --output /path/to/prepared
python verify_protocol.py --prepared /path/to/prepared
python -m unittest discover -s . -p test_revision.py -v
python run.py --prepared /path/to/prepared --output /path/to/runs --stage main
python aggregate.py --prepared /path/to/prepared --runs /path/to/runs --output /path/to/results
python run.py --prepared /path/to/prepared --output /path/to/runs --stage ablations
python multihorizon.py --prepared /path/to/prepared --runs /path/to/runs --output /path/to/multihorizon
python aggregate_diagnostics.py --prepared /path/to/prepared --runs /path/to/runs --multihorizon /path/to/multihorizon --output /path/to/results
python benchmark.py --prepared /path/to/prepared --runs /path/to/runs --output /path/to/results
python plots.py --results /path/to/results
python verify_results.py --prepared /path/to/prepared --runs /path/to/runs --results /path/to/results
```

Optional `--cuda-runtime` points to an existing compatible target installation;
it is unnecessary in a normal environment with the desired PyTorch on its path.
Set `GTS_CACHE_ROOT` to a large temporary drive, or use `--cache-root` with the
runner/pipeline. The boosting matrix is column-major float64 on disk so that
scikit-learn does not create another full dense float64 copy in RAM. Its temporary
size is checked before fitting and the task's own cache file is removed afterwards.
`pipeline.py` can queue the stages and records progress/failures durably. It never
edits the article or publishes files. `results_ready_for_paper.json` is produced
only after main fits, ablations, all five horizons, statistical sections, live
inference checks, figures and a separate numerical verification finish.

## Interpretation

The new sample sizes differ from the archived study. Never compare their scores
as though only the architecture changed. Remaining incomplete waits are included
in distributional model fitting; common point-error tables condition on observed
successors. This does not establish generalization to cold-start users or all
customers who no longer buy. Instacart is a relative-order backtest. A calendar
backtest cannot establish a genuinely future prospective deployment result.

Customer-level resampling supplies MAE, RMSE and MedAE intervals. Predeclared
paired MAE/MSE randomization comparisons use Holm adjustment. Unordered customer
outcomes are not passed to a time-series Diebold-Mariano test. The asymmetric
cost ratios are hypothetical timing penalties, not observed marketing profits.
Conditional-median forecasts need not match conditional means; grouped plots
therefore show both observed means and medians, plus empirical coverage.

Five-horizon experiments use clean positive purchase days. Direct and recursive
Transformers use the same initial inputs and matching outcomes in one PyTorch
implementation. Recursive forecasts update gap-dependent features using predicted
gaps and hold unavailable future basket attributes at their last observed values;
actual future basket or gap information is never provided.

Primary methodological sources: [DeepAR](https://arxiv.org/abs/1704.04110),
[Weibull AFT](https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html),
and [stacking data separation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html).
