"""
Real human clinical data, as a stand-in for live sensor input.

Everything else in this project (dataset.py, classify.py) trains and tests
the anomaly detector on fully synthetic patients: a vessel tree simulated
from first principles, perturbed by a scripted clog/leak, with Gaussian
sensor noise layered on top. That's necessary -- nobody has a public
dataset of labelled per-vessel blood-flow sensor readings next to a tumor
diagnosis -- but it means the whole pipeline has only ever seen data it
generated itself.

This module is the honest bridge to real biology, in two ways:

1. REALISTIC BOUNDARY CONDITIONS. `data/uci_heart_disease.csv` is the
   classic UCI Cleveland Heart Disease dataset: 303 real, de-identified
   patients, each with a real clinically measured resting blood pressure
   (`trestbps`, mmHg). generate_dataset_from_real_bp() samples the
   simulator's root pressure P_in from THAT real distribution (bootstrap
   resampling, with small jitter) instead of a single fixed constant, so
   the "dummy" training data is at least driven by real human
   physiological variability rather than one arbitrary number.

2. LIVE REAL-PATIENT INPUT. build_patient_reading() takes a systolic BP
   (either picked from a real patient row, or typed in by the user) and
   runs it through the actual vessel-network solver and an optional
   anomaly, producing a sensor reading in exactly the schema the trained
   classifier expects -- so a real human blood-pressure value can be fed
   into the model and the app can show what it predicts.

What this does NOT claim: the UCI dataset has no per-vessel flow readings
and no tumor/occlusion ground truth, so this is not a validation of
detection accuracy against real pathology -- it's real physiological input
driving the same physics engine, which is the most honest version of
"test it on real data" available without a real perfusion dataset.
"""

import os

import numpy as np
import pandas as pd

from network import VesselNetwork
from solver import solve_network
from anomaly import apply_clog, apply_leak, random_scenario
from constants import MMHG_TO_PA, P_OUT

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "uci_heart_disease.csv")
NHANES_PATH = os.path.join(DATA_DIR, "nhanes_2009_2012_bp.csv")

# The real cohorts bundled in data/. Each loads to the same core columns
# (cohort, age, sex, systolic_bp_mmhg) plus whatever else that source has.
COHORTS = {
    "uci": "UCI Cleveland heart-disease patients",
    "nhanes": "NHANES 2009-2012 US adults",
}

SEX_LABELS = {1: "male", 0: "female"}
DIAGNOSIS_LABELS = {1: "heart disease present", 0: "no heart disease"}


def load_real_patients(path=DATA_PATH):
    """Load the bundled real UCI Cleveland Heart Disease dataset (303 real
    patients). `encoding='utf-8-sig'` strips the BOM the source file ships
    with. Adds a couple of readable columns alongside the originals."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={
        "trestbps": "resting_systolic_bp_mmhg",
        "chol": "cholesterol_mg_dl",
        "thalach": "max_heart_rate",
    })
    df["sex_label"] = df["sex"].map(SEX_LABELS)
    df["diagnosis"] = df["target"].map(DIAGNOSIS_LABELS)
    return df


def load_nhanes(path=NHANES_PATH):
    """Load the bundled NHANES blood-pressure cohort: 4,633 adults (18+)
    from the US National Health and Nutrition Examination Survey, 2009-2012
    cycles, each with an averaged measured systolic and diastolic pressure.

    Source: the CDC's public-domain NHANES data as packaged in the `NHANES`
    R teaching dataset, reduced here to one row per participant, adults
    with a blood-pressure reading, and ten columns. That package resamples
    the survey to resemble a simple random sample, so it is suitable for a
    realistic spread of pressures but not for official prevalence estimates.
    """
    return pd.read_csv(path)


def load_cohort(key):
    """One real cohort with standard columns first: cohort, age, sex,
    systolic_bp_mmhg. `key` is a key of COHORTS."""
    if key == "uci":
        df = load_real_patients()
        out = pd.DataFrame({
            "cohort": COHORTS["uci"],
            "age": df["age"],
            "sex": df["sex_label"],
            "systolic_bp_mmhg": df["resting_systolic_bp_mmhg"].astype(float),
            "cholesterol_mg_dl": df["cholesterol_mg_dl"],
            "max_heart_rate": df["max_heart_rate"],
            "diagnosis": df["diagnosis"],
        })
    elif key == "nhanes":
        df = load_nhanes()
        out = pd.DataFrame({
            "cohort": COHORTS["nhanes"],
            "age": df["age"],
            "sex": df["sex"],
            "systolic_bp_mmhg": df["systolic_bp_mmhg"].astype(float),
            "diastolic_bp_mmhg": df["diastolic_bp_mmhg"],
            "pulse_bpm": df["pulse_bpm"],
            "bmi": df["bmi"],
            # NHANES reports cholesterol in mmol/L; 1 mmol/L = 38.67 mg/dL
            "cholesterol_mg_dl": (df["total_cholesterol_mmol_l"] * 38.67).round(0),
            "diabetes": df["diabetes"],
        })
    else:
        raise KeyError(f"Unknown cohort {key!r}; choose from {list(COHORTS)}")
    return out.reset_index(drop=True)


def load_cohorts(keys=("uci", "nhanes")):
    """Several real cohorts stacked into one table."""
    return pd.concat([load_cohort(k) for k in keys], ignore_index=True)


def real_bp_summary(df=None):
    """Summary stats of real resting systolic BP, for comparing against
    the simulator's synthetic P_in distribution."""
    if df is None:
        df = load_real_patients()
    bp = df["resting_systolic_bp_mmhg"]
    return {
        "n": int(bp.shape[0]),
        "mean": float(bp.mean()),
        "std": float(bp.std()),
        "min": float(bp.min()),
        "max": float(bp.max()),
        "values": bp.values,
    }


def sensor_schema(net):
    """The exact sensor layout (leaf pressure nodes + root-vessel flows)
    for a given network, in the same order dataset.py uses to build
    feature columns. The web app uses this to render a matching input
    form, and to line up a manually entered reading with a trained
    model's feature_cols."""
    sensor_nodes = list(net.leaves)
    root_children = [v.id for v in net.vessels if v.u == 0]
    pressure_cols = [f"P_node{n}_mmHg" for n in sensor_nodes]
    flow_cols = [f"Q_vessel{vid}" for vid in root_children]
    return {
        "sensor_nodes": sensor_nodes,
        "root_children": root_children,
        "pressure_cols": pressure_cols,
        "flow_cols": flow_cols,
        "feature_cols": pressure_cols + flow_cols,
    }


def build_patient_reading(net, systolic_bp_mmhg, p_out=P_OUT,
                          scenario_type="healthy", vessel_id=None, severity=0.0):
    """Drive the vessel-network solver with a real (or manually entered)
    systolic blood pressure as the root boundary condition P_in, optionally
    apply a clog/leak, and return one sensor reading row (noise-free) in
    the schema sensor_schema() describes -- ready to hand to a trained
    classifier's .predict().
    """
    net.reset()
    p_in = systolic_bp_mmhg * MMHG_TO_PA

    applied = {"type": "healthy", "vessel_id": -1, "severity": 0.0}
    if scenario_type == "clog" and vessel_id is not None:
        apply_clog(net, vessel_id, severity)
        applied = {"type": "clog", "vessel_id": vessel_id, "severity": severity}
    elif scenario_type == "leak" and vessel_id is not None:
        apply_leak(net, vessel_id, severity)
        applied = {"type": "leak", "vessel_id": vessel_id, "severity": severity}

    pressures, flows = solve_network(net, p_in=p_in, p_out=p_out)
    schema = sensor_schema(net)

    row = {}
    for node, col in zip(schema["sensor_nodes"], schema["pressure_cols"]):
        row[col] = pressures[node] / MMHG_TO_PA
    for vid, col in zip(schema["root_children"], schema["flow_cols"]):
        row[col] = flows[vid]

    net.reset()
    return row, applied


def generate_dataset_from_real_bp(n_samples=3000, generations=4, seed=0,
                                  pressure_noise_mmhg=2.0, flow_noise_frac=0.03,
                                  real_df=None):
    """Same synthetic-patient generator as dataset.generate_dataset(), but
    P_in is bootstrap-resampled (with small Gaussian jitter so repeated
    draws of the same real patient aren't identical) from the REAL
    systolic-BP distribution in data/uci_heart_disease.csv, instead of a
    single fixed constant. Every other physiological parameter (vessel
    radii, viscosity, anomaly severity) is still the simulator's own, so
    this is a real-data-calibrated boundary condition, not a fully
    real-world-validated dataset -- see the module docstring.
    """
    import random as _random
    from dataset import _sensor_readings  # reuse the exact noise/feature logic

    if real_df is None:
        real_df = load_real_patients()
    real_bp = real_df["resting_systolic_bp_mmhg"].values

    rng = _random.Random(seed)
    np_rng = np.random.default_rng(seed)
    net = VesselNetwork(generations=generations)

    sensor_nodes = net.leaves
    root_children = [v.id for v in net.vessels if v.u == 0]

    rows = []
    for _ in range(n_samples):
        scenario = random_scenario(net, rng)
        bp_sample = real_bp[np_rng.integers(0, len(real_bp))]
        bp_sample = bp_sample + np_rng.normal(0, 3.0)  # jitter so resampling isn't exact repeats
        p_in = bp_sample * MMHG_TO_PA

        pressures, flows = solve_network(net, p_in=p_in)
        row = _sensor_readings(pressures, flows, sensor_nodes, root_children,
                               np_rng, pressure_noise_mmhg, flow_noise_frac)
        row["real_bp_source_mmHg"] = bp_sample
        row["label_type"] = scenario["type"]
        row["label_vessel_id"] = scenario["vessel_id"]
        row["label_severity"] = scenario["severity"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, net
