# Hemodynamics / Cardiovascular Flow-Anomaly Project

Physics 207 honors project: can a model, trained on a physically simulated
vascular network, detect **where flow has stopped** (a clog or a leak) from
noisy pressure/flow sensor readings alone — and can that detector be
**thresholded and backed by statistics** strong enough to trust?

## What's actually here

A binary vascular tree (root → generations of branching vessels, Murray's
law radius scaling) is built as a resistor network and solved exactly with
linear algebra (Kirchhoff's current law as `A·V = b`). A clog (radius
loss) or leak (conductance to ground) can be injected into any vessel.
Thousands of such patients are simulated, sensor noise is added, and a
Random Forest / SVM is trained to localize the anomaly from the noisy
readings alone.

On top of that core simulation, this pass of the project adds:

- **Statistics** (`stats.py`) — ROC/PR curves + AUC, a sensitivity/specificity
  threshold sweep (Youden's J) for the healthy-vs-anomaly detector,
  stratified k-fold cross-validation with a 95% CI, a bootstrap CI on
  held-out accuracy, a label-permutation significance test, and McNemar's
  paired test comparing the Random Forest against the SVM.
- **Linear algebra** (`linalg_analysis.py`) — the condition number of the
  solver's matrix `A`, a finite-difference sensitivity matrix
  `∂(leaf pressure)/∂(log vessel resistance)` (a model-free, purely
  linear-algebra counterpart to the Random Forest's feature importance),
  and the graph Laplacian's spectrum (algebraic connectivity / Fiedler
  value, which drops as a clog pinches the network toward a bottleneck).
- **Real human data** (`real_data.py`, `data/uci_heart_disease.csv`) — the
  real, public UCI Cleveland Heart Disease dataset (303 real patients,
  real clinically measured resting blood pressure). It calibrates the
  simulator's boundary pressure from a real population distribution
  instead of one fixed constant, and lets a real patient's real blood
  pressure be run through the solver and the trained classifier for a
  live prediction. **Important honesty note:** no public dataset pairs
  real per-vessel flow sensors with a tumor/occlusion diagnosis, so this
  is real physiological input driving the same physics engine and
  model — not a validation of detection accuracy against real pathology.
  See the in-app "Real Patient Data" tab for the full explanation.
- **Interactive app** (`app.py`) — a Streamlit dashboard to explore all of
  the above with live parameter controls instead of re-running scripts.

## Quickstart

```bash
cd hemodynamics-project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) —
open it in a browser. The app has seven tabs:

1. **Network Explorer** — pick a scenario (healthy/clog/leak) and see the
   vessel tree and sensor pressures update live. Hover a vessel for its
   flow or a junction for its pressure.
2. **Detection & Statistics** — simulate patients, train the detector,
   and see ROC/PR curves, cross-validated accuracy with a CI, bootstrap
   CI, and the RF-vs-SVM significance test. Each simulated patient's
   arterial pressure can be one fixed value or drawn from real people
   (see "Real data" below).
3. **Data Explorer** — choose the simulated-patient table or the real
   cohorts, filter rows, and chart any column against any other
   (scatter, histogram, box plot, counts). The filtered rows can be
   downloaded.
4. **Linear Algebra** — condition number, sensitivity matrix, and
   algebraic connectivity, live against the sidebar's network parameters.
5. **Genetics** — the SNP → polygenic-risk-score → anomaly pipeline.
6. **Real Patient Data** — the real cohorts' blood pressures, and running
   one real person's pressure through the trained detector, or uploading
   your own sensor-reading CSV for batch predictions.
7. **Experiment Tracking** — every training run's settings, scores and
   trained model are recorded with MLflow and listed for comparison.
   Stored in a local file by default, or in a Databricks workspace
   (see below).

Charts are interactive: hover for exact values, drag to zoom,
double-click to reset, click a legend entry to hide a series. Colours and
corner styles are set in `.streamlit/config.toml` for light and dark mode.

All network parameters (tree depth, vessel geometry, viscosity, boundary
pressures) are adjustable from the sidebar and apply across tabs.

### Real data

Two real, de-identified cohorts are bundled in `data/`:

| File | Who | People | Source |
|---|---|---|---|
| `uci_heart_disease.csv` | Patients referred for heart-disease testing, Cleveland | 303 | UCI Machine Learning Repository |
| `nhanes_2009_2012_bp.csv` | US adults (18+) with a measured blood pressure | 4,633 | CDC NHANES 2009-2012, via the `NHANES` R teaching dataset, reduced to one row per participant |

No public dataset pairs per-vessel flow readings with a tumor or
occlusion diagnosis. Real data therefore enters as **real systolic blood
pressure driving the simulated vessel network**: in Detection &
Statistics, set *Arterial pressure of each simulated patient* to a real
cohort and each simulated patient gets a pressure measured in a real
person. This makes detection harder and more honest: with 3,000 patients
and default settings, Random Forest localization accuracy drops from
1.000 (one fixed pressure) to about 0.97 (real pressures). It is not a
validation against real tumor outcomes. The NHANES teaching dataset is
resampled, so it gives a realistic spread of pressures but should not be
used for official prevalence figures.

### Experiment tracking and Databricks

Each time the detector is trained in the ML tab, one *run* is recorded:
the settings (samples, seed, noise, network geometry, pressures), the
scores (accuracy, ROC AUC, PR AUC, chosen threshold, sensitivity,
specificity) and, optionally, the trained model. Training is seeded, so a
configuration that is already recorded is not recorded twice.

- **Local (default, no account).** Runs go to `outputs/mlflow.db`, models
  to `outputs/mlartifacts/`. Both are git-ignored. For MLflow's own
  interface: `mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db`.
- **Databricks.** In the Experiment Tracking tab choose *Databricks
  workspace* and enter your workspace URL
  (`https://dbc-....cloud.databricks.com`), a personal access token
  (Databricks: User Settings > Developer > Access tokens) and an
  experiment path such as `/Users/<your login email>/hemodynamics`. Runs
  then appear under Experiments in the workspace. The token is held in
  memory for the session and never written to disk.

The dataset here is a few thousand rows, so Databricks is used for what
it adds at this size (a shared, permanent record of experiments and
models), not for cluster computing.

### Running the original command-line scripts

These still work standalone and write into `outputs/`:

```bash
python main.py            # synthetic dataset + classifiers -> outputs/results.png
python main_genetics.py   # genetics/PRS pipeline -> outputs/results_genetics.png
```

## Repository layout

```
constants.py          physical constants (viscosity, mmHg<->Pa, default pressures)
network.py            vessel tree construction (Murray's law), Hagen-Poiseuille resistance
solver.py             assembles & solves A·V = b (nodal/Kirchhoff analysis)
anomaly.py            injects clogs / leaks into a vessel
genetics.py           SNP panel, polygenic risk score, genetic-risk-linked anomalies
dataset.py            simulates many patients -> labeled sensor-reading datasets
classify.py           Random Forest / SVM localization, PRS regression
stats.py              ROC/PR, thresholding, cross-validation, bootstrap, permutation, McNemar
linalg_analysis.py    condition number, sensitivity matrix, graph Laplacian spectrum
real_data.py          loads the real UCI dataset; real-BP-driven simulation & prediction
visualize.py          vessel-tree layout and the static diagram used by main.py
charts.py             interactive Plotly charts used by the app
tracking.py           MLflow experiment tracking (local SQLite or a Databricks workspace)
app.py                Streamlit interactive dashboard (run this)
main.py, main_genetics.py   original command-line entry points -> outputs/
data/                 real input data (uci_heart_disease.csv, nhanes_2009_2012_bp.csv)
.streamlit/config.toml  app colours for light and dark mode
outputs/              generated datasets & figures (dataset.csv, results.png, ...)
```

## Known limitations / honest next steps

- The vessel tree is a symmetric binary tree with Murray's-law scaling —
  real vasculature (and real tumor-adjacent angiogenesis) is far more
  irregular. A natural next step is importing a real or semi-real vessel
  graph (e.g. from a segmented angiogram) instead of a generated tree.
- Flow is steady-state (one pressure solve per patient), not the pulsatile,
  time-varying flow a real sensor would record — there's no time-series
  structure for a model to exploit yet.
- "Real data" integration here is a real blood-pressure boundary condition
  and a live-prediction demo, not a labeled real-world validation set —
  because that dataset doesn't publicly exist for this problem. If a
  perfusion-imaging or angiography dataset with flow labels becomes
  available, swapping it in instead of the synthetic generator is the
  highest-value next step for genuinely validating the detector.
- The genetics model (SNP panel, effect sizes, PRS→severity link) is
  illustrative/plausible, not fit to real GWAS summary statistics.
