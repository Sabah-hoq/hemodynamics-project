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
open it in a browser. The app has five tabs:

1. **Network Explorer** — pick a scenario (healthy/clog/leak), see the
   vessel tree and sensor pressures update live.
2. **ML Detection & Statistics** — generate a synthetic dataset, train the
   detector, see ROC/PR curves, cross-validated accuracy with a CI,
   bootstrap CI, and the RF-vs-SVM significance test.
3. **Linear Algebra** — condition number, sensitivity matrix, and
   algebraic connectivity, live against the sidebar's network parameters.
4. **Genetics / PRS** — the SNP → polygenic-risk-score → anomaly pipeline.
5. **Real Patient Data** — browse the real UCI patients, run one through
   the trained detector with their real blood pressure, or upload your
   own sensor-reading CSV for batch predictions.

All network parameters (tree depth, vessel geometry, viscosity, boundary
pressures) are adjustable from the sidebar and apply across tabs.

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
visualize.py          vessel-tree diagram
app.py                Streamlit interactive dashboard (run this)
main.py, main_genetics.py   original command-line entry points -> outputs/
data/                 real input data (uci_heart_disease.csv)
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
