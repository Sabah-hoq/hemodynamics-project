"""
Interactive dashboard for the hemodynamics flow-anomaly project.

    streamlit run app.py

Seven tabs: a live network explorer, the detector and its statistics, a
free-form data explorer, the linear-algebra diagnostics on the solver's
matrix, the genetics/PRS model, real patient cohorts, and experiment
tracking. Charts are interactive (hover, zoom, legend toggles).
"""

import numpy as np
import pandas as pd
import streamlit as st

from network import VesselNetwork
from solver import solve_network
from anomaly import apply_clog, apply_leak
from dataset import generate_dataset, generate_genetic_dataset
from classify import train_and_evaluate, train_prs_regressor, build_location_label
from visualize import _node_position
from constants import MMHG_TO_PA
import stats
import linalg_analysis as la
import real_data as rd
import tracking
import charts
import databricks_tables as dbt

st.set_page_config(page_title="Vascular Flow Anomaly Detection", layout="wide")

try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False
PAL = charts.palette(DARK)
LABEL_COLORS = charts.label_colors(PAL)

BP_SOURCES = {
    "Fixed value (sidebar)": None,
    "Real: UCI Cleveland patients (303)": ("uci",),
    "Real: NHANES US adults (4,633)": ("nhanes",),
    "Real: both cohorts (4,936)": ("uci", "nhanes"),
}


UPLOAD_TABLES = {
    "Simulated patients (current training set)": "hemodynamics_simulated_patients",
    "Real people (both cohorts)": "hemodynamics_real_cohorts",
}


def show(fig):
    """Render a Plotly figure full width with a slim toolbar."""
    st.plotly_chart(fig, width="stretch", theme="streamlit",
                    config={"displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


# --------------------------------------------------------------- caching --

@st.cache_data(show_spinner=False)
def cached_dataset(n_samples, generations, seed, pressure_noise, flow_noise,
                   root_radius, root_length, terminal_resistance, viscosity,
                   p_in_mmhg, p_out_mmhg, healthy_prob, bp_source="Fixed value (sidebar)"):
    cohort_keys = BP_SOURCES[bp_source]
    real_bp = (rd.load_cohorts(cohort_keys)["systolic_bp_mmhg"].values
               if cohort_keys else None)
    df, _ = generate_dataset(
        n_samples=n_samples, generations=generations, seed=seed,
        pressure_noise_mmhg=pressure_noise, flow_noise_frac=flow_noise,
        root_radius=root_radius, root_length=root_length,
        terminal_resistance=terminal_resistance, viscosity=viscosity,
        p_in_mmhg=p_in_mmhg, p_out_mmhg=p_out_mmhg, healthy_prob=healthy_prob,
        p_in_values_mmhg=real_bp,
    )
    return df


@st.cache_data(show_spinner=False)
def _train_bundle(df, seed):
    # Streamlit reruns this whole script on *any* widget interaction
    # anywhere on the page, so training (300-tree RF + calibrated SVM)
    # must be cached on (df, seed) or every keystroke elsewhere would
    # silently retrain it.
    return train_and_evaluate(df, seed=seed)


@st.cache_data(show_spinner=False)
def cached_cv(df, feature_cols, seed):
    # every click anywhere reruns the script; keep the 5-fold refit out of that
    from sklearn.ensemble import RandomForestClassifier
    X = df[feature_cols].values
    y_bin = (df["label_type"] != "healthy").astype(int).values
    return stats.cross_validated_accuracy(
        RandomForestClassifier(n_estimators=150, random_state=seed), X, y_bin, k=5)


@st.cache_data(show_spinner=False)
def cached_genetic_models(gdf, seed):
    return train_and_evaluate(gdf, seed=seed), train_prs_regressor(gdf, seed=seed)


@st.cache_data(show_spinner=False)
def cached_cohorts():
    return rd.load_cohorts()


def current_tracker():
    """Activate the tracking store this session chose (set in the Experiment
    Tracking tab) and return (label, experiment_name)."""
    dbx = st.session_state.get("dbx")
    if st.session_state.get("track_backend") == "Databricks workspace" and dbx:
        name = tracking.activate("databricks", dbx["host"], dbx["token"], dbx["experiment"])
        return "Databricks", name
    return "local file", tracking.activate("local")


def make_network(params):
    return VesselNetwork(
        generations=params["generations"],
        root_radius=params["root_radius"],
        root_length=params["root_length"],
        terminal_resistance=params["terminal_resistance"],
        viscosity=params["viscosity"],
    )


# ---------------------------------------------------------------- sidebar --

st.sidebar.title("Network parameters")
generations = st.sidebar.slider("Generations (tree depth)", 2, 6, 4)
root_radius_mm = st.sidebar.slider("Root vessel radius (mm)", 1.0, 8.0, 4.0, 0.5)
root_length_cm = st.sidebar.slider("Root vessel length (cm)", 1.0, 6.0, 3.0, 0.5)
viscosity = st.sidebar.slider("Blood viscosity (Pa·s)", 1.5e-3, 6.0e-3, 3.5e-3, 0.1e-3,
                              format="%.4f")
terminal_resistance = st.sidebar.number_input(
    "Terminal (capillary-bed) resistance", value=3e8, format="%.2e")

st.sidebar.markdown("---")
st.sidebar.subheader("Boundary pressures")
p_in_mmhg = st.sidebar.slider("Root (arterial) pressure, mmHg", 70, 220, 100)
p_out_mmhg = st.sidebar.slider("Ground (venous) pressure, mmHg", 0, 20, 5)

net_params = dict(
    generations=generations,
    root_radius=root_radius_mm * 1e-3,
    root_length=root_length_cm * 1e-2,
    terminal_resistance=terminal_resistance,
    viscosity=viscosity,
)

st.title("Vascular Flow Anomaly Detection")
st.caption(
    "Physics 207 · A branching vessel network solved as a resistor circuit, "
    "disturbed with simulated clogs and leaks, and read by a trained detector."
)

(tab_explore, tab_ml, tab_data, tab_linalg, tab_genetics, tab_real, tab_track) = st.tabs(
    ["Network Explorer", "Detection & Statistics", "Data Explorer", "Linear Algebra",
     "Genetics", "Real Patient Data", "Experiment Tracking"]
)


# =============================================================== EXPLORE ==
with tab_explore:
    st.subheader("Single-scenario explorer")
    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        scenario_type = st.radio("Scenario", ["healthy", "clog", "leak"], horizontal=True)
        net_preview = make_network(net_params)
        vessel_id = None
        severity = 0.0
        if scenario_type != "healthy":
            vessel_id = st.slider("Vessel id", 0, net_preview.n_named_vessels - 1, 2)
            severity = st.slider("Severity (fraction of radius lost / leak strength)",
                                 0.0, 0.98, 0.7, 0.01)

    net = make_network(net_params)
    healthy_pressures, healthy_flows = solve_network(
        net, p_in=p_in_mmhg * MMHG_TO_PA, p_out=p_out_mmhg * MMHG_TO_PA)

    net.reset()
    if scenario_type == "clog":
        apply_clog(net, vessel_id, severity)
    elif scenario_type == "leak":
        apply_leak(net, vessel_id, severity)
    pressures, flows = solve_network(
        net, p_in=p_in_mmhg * MMHG_TO_PA, p_out=p_out_mmhg * MMHG_TO_PA)

    with col_plot:
        show(charts.vessel_tree(
            net, _node_position(net), pressures, flows, healthy_flows, MMHG_TO_PA,
            highlight_vessel_id=vessel_id, scenario=scenario_type, pal=PAL))
        st.caption("Hover a vessel for its flow, or a junction for its pressure. "
                   "Line width follows vessel radius.")

    st.markdown("##### Leaf (sensor) pressures: healthy vs. current scenario")
    rows = []
    for leaf in net.leaves:
        rows.append({
            "leaf node": leaf,
            "healthy (mmHg)": healthy_pressures[leaf] / MMHG_TO_PA,
            "current (mmHg)": pressures[leaf] / MMHG_TO_PA,
            "Δ (mmHg)": (pressures[leaf] - healthy_pressures[leaf]) / MMHG_TO_PA,
        })
    sensor_df = pd.DataFrame(rows)
    st.dataframe(sensor_df.style.format({"healthy (mmHg)": "{:.2f}", "current (mmHg)": "{:.2f}",
                                         "Δ (mmHg)": "{:+.2f}"}), width="stretch", height=300)

    c1, c2 = st.columns([2, 1])
    with c1:
        show(charts.delta_bars(sensor_df["leaf node"], sensor_df["Δ (mmHg)"], PAL,
                               "Sensor pressure change from healthy", "Change (mmHg)"))
    with c2:
        root_children = [v.id for v in net.vessels if v.u == 0]
        st.markdown("**Flow in the two main branches**")
        for vid in root_children:
            st.metric(f"Vessel {vid} (mL/s)", f"{flows[vid] * 1e6:.3f}",
                      delta=f"{(flows[vid] - healthy_flows[vid]) * 1e6:+.3f}", border=True)


# =================================================================== ML ==
with tab_ml:
    st.subheader("Simulated patients → trained detector")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_samples = st.slider("Samples", 300, 6000, 3000, 100)
        seed = st.number_input("Random seed", value=1, step=1)
    with c2:
        pressure_noise = st.slider("Sensor pressure noise (mmHg std)", 0.0, 15.0, 2.0, 0.5)
        flow_noise = st.slider("Sensor flow noise (fraction std)", 0.0, 0.2, 0.03, 0.01)
    with c3:
        healthy_prob = st.slider("Fraction healthy", 0.0, 0.5, 0.15, 0.05)
        bp_source = st.selectbox(
            "Arterial pressure of each simulated patient", list(BP_SOURCES),
            help="Fixed gives every simulated patient the sidebar pressure. The real "
                 "options draw each patient's pressure from blood pressures measured "
                 "in real people, so the detector has to cope with real variation.")
        run_btn = st.button("Generate dataset & train", type="primary")

    state_key = "ml_df"
    if run_btn or state_key not in st.session_state:
        with st.spinner("Simulating patients and solving the network..."):
            df = cached_dataset(
                n_samples, generations, int(seed), pressure_noise, flow_noise,
                net_params["root_radius"], net_params["root_length"],
                net_params["terminal_resistance"], net_params["viscosity"],
                p_in_mmhg, p_out_mmhg, healthy_prob, bp_source,
            )
            st.session_state[state_key] = df
            # the settings that produced this dataset; widgets can move
            # afterwards without regenerating, so read these, not the widgets
            st.session_state["ml_run_params"] = dict(
                n_samples=n_samples, seed=int(seed),
                pressure_noise_mmhg=pressure_noise, flow_noise_frac=flow_noise,
                healthy_prob=healthy_prob, generations=generations,
                root_radius_mm=root_radius_mm, root_length_cm=root_length_cm,
                viscosity=viscosity, terminal_resistance=terminal_resistance,
                p_in_mmhg=p_in_mmhg, p_out_mmhg=p_out_mmhg,
                pressure_source=bp_source,
            )

    df = st.session_state[state_key]
    st.caption(f"{len(df)} simulated patients · class balance: "
              f"{df['label_type'].value_counts().to_dict()}")

    with st.spinner("Training Random Forest + SVM..."):
        results = _train_bundle(df, st.session_state["ml_run_params"]["seed"])
    st.session_state["ml_results"] = results
    st.session_state["ml_net_params"] = dict(net_params)

    st.markdown("### Multi-class localization (which vessel, if any)")
    m1, m2 = st.columns(2)
    m1.metric("Random Forest accuracy", f"{results['random_forest']['accuracy']:.3f}", border=True)
    m2.metric("SVM accuracy", f"{results['svm']['accuracy']:.3f}", border=True)

    rf = results["random_forest"]["model"]
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    feat_names = [results["feature_cols"][i] for i in order]
    show(charts.horizontal_bars(feat_names, importances[order], PAL,
                                "Which sensors the Random Forest relies on (top 15)",
                                "Importance"))

    st.markdown("### Binary detector: \"is flow stopping anywhere?\" — thresholded & scored")
    bd = results["binary_detector"]
    roc = stats.roc_pr_analysis(bd["y_test"], bd["y_proba"])
    ts = stats.threshold_sweep(bd["y_test"], bd["y_proba"])

    # ---- record this training run (once per distinct configuration) ----
    if st.session_state.get("track_enabled", True):
        run_params = st.session_state["ml_run_params"]
        try:
            tracker_label, _ = current_tracker()
            sig = (tracker_label, tuple(sorted(run_params.items())))
            logged = st.session_state.setdefault("logged_runs", {})
            if sig not in logged:
                with st.spinner(f"Recording run ({tracker_label})..."):
                    logged[sig] = tracking.log_training_run(
                        run_params,
                        {
                            "rf_accuracy": results["random_forest"]["accuracy"],
                            "svm_accuracy": results["svm"]["accuracy"],
                            "roc_auc": roc["roc_auc"],
                            "pr_auc": roc["pr_auc"],
                            "best_threshold": ts["best_threshold"],
                            "sensitivity": ts["best_sensitivity"],
                            "specificity": ts["best_specificity"],
                        },
                        model=bd["model"] if st.session_state.get("track_model", True) else None,
                    )
            st.caption(f"Recorded as run `{logged[sig][:8]}` ({tracker_label}) · "
                       "compare runs in the Experiment Tracking tab")
        except Exception as e:  # tracking must never take the analysis down
            st.warning(f"This run was not recorded: {e}")

    col_roc, col_pr, col_thresh = st.columns(3)
    with col_roc:
        show(charts.roc_curve(roc["fpr"], roc["tpr"], roc["roc_auc"], PAL))
    with col_pr:
        show(charts.pr_curve(roc["recall"], roc["precision"], roc["pr_auc"], PAL))
    with col_thresh:
        show(charts.threshold_sweep(ts["thresholds"], ts["sensitivity"], ts["specificity"],
                                    ts["best_threshold"], PAL))

    st.caption(
        f"At the Youden's-J-optimal threshold ({ts['best_threshold']:.2f}), the detector gets "
        f"{ts['best_sensitivity']:.1%} sensitivity and {ts['best_specificity']:.1%} specificity "
        "on held-out simulated patients."
    )

    st.markdown("### Is this real, or did we get a lucky split?")
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.spinner("5-fold cross-validating..."):
            cv = cached_cv(df, results["feature_cols"],
                           st.session_state["ml_run_params"]["seed"])
        st.metric("5-fold CV accuracy", f"{cv['mean']:.3f}", border=True,
                  help=f"95% CI [{cv['lo']:.3f}, {cv['hi']:.3f}]")
        st.caption(f"95% CI: [{cv['lo']:.3f}, {cv['hi']:.3f}]")
    with s2:
        boot = stats.bootstrap_accuracy_ci(
            results["random_forest"]["y_test"], results["random_forest"]["y_pred"], n_boot=1000)
        st.metric("Bootstrap test accuracy", f"{boot['mean']:.3f}", border=True,
                  help=f"95% CI [{boot['lo']:.3f}, {boot['hi']:.3f}]")
        st.caption(f"95% CI: [{boot['lo']:.3f}, {boot['hi']:.3f}]")
    with s3:
        mc = stats.mcnemar_test(results["random_forest"]["y_test"],
                                results["random_forest"]["y_pred"], results["svm"]["y_pred"])
        st.metric("RF vs SVM (McNemar p)", f"{mc['p_value']:.4f}", border=True)
        st.caption(f"discordant: RF-only-right {mc['n10']}, SVM-only-right {mc['n01']}")

    st.markdown("### Accuracy vs. sensor noise")
    if st.checkbox("Run noise sweep (slower)"):
        noise_levels = [0, 1, 2, 3, 4, 8, 15]
        accs = []
        with st.spinner("Sweeping sensor noise levels..."):
            for noise in noise_levels:
                d = cached_dataset(
                    min(n_samples, 1500), generations, int(seed), noise, flow_noise,
                    net_params["root_radius"], net_params["root_length"],
                    net_params["terminal_resistance"], net_params["viscosity"],
                    p_in_mmhg, p_out_mmhg, healthy_prob, bp_source,
                )
                r = _train_bundle(d, int(seed))
                accs.append(r["random_forest"]["accuracy"])
        show(charts.line_chart(noise_levels, [("Random Forest accuracy", accs, "blue")], PAL,
                               "Accuracy as sensors get noisier",
                               "Sensor pressure noise (mmHg std)", "Accuracy",
                               yrange=[0, 1.05]))


# =============================================================== LINALG ==
with tab_linalg:
    st.subheader("The linear system behind every pressure solve")
    st.markdown(
        "Every scenario above is one solve of **A·V = b**, Kirchhoff's current law written as "
        "a weighted-graph-Laplacian linear system. This tab inspects *A* itself."
    )

    net_la = make_network(net_params)
    la_scenario = st.radio("Perturb the network for this analysis",
                           ["healthy", "clog", "leak"], horizontal=True, key="la_scenario")
    la_vessel = None
    la_severity = 0.0
    if la_scenario != "healthy":
        la_vessel = st.slider("Vessel id", 0, net_la.n_named_vessels - 1, 2, key="la_vessel")
        la_severity = st.slider("Severity", 0.0, 0.98, 0.7, 0.01, key="la_severity")
        if la_scenario == "clog":
            apply_clog(net_la, la_vessel, la_severity)
        else:
            apply_leak(net_la, la_vessel, la_severity)

    c1, c2 = st.columns(2)
    with c1:
        cn = la.condition_number(net_la, p_in=p_in_mmhg * MMHG_TO_PA, p_out=p_out_mmhg * MMHG_TO_PA)
        st.metric("Condition number of A", f"{cn['condition_number']:.1f}", border=True)
        st.caption("How much a tiny error in resistance/pressure gets amplified in the solved "
                  "pressures. Near 1 is well-posed; rising sharply signals numerical fragility.")
    with c2:
        ac = la.algebraic_connectivity(net_la)
        st.metric("Fiedler value (algebraic connectivity)", f"{ac['fiedler_value']:.3e}", border=True)
        st.caption("Second-smallest eigenvalue of the graph Laplacian. Drops toward 0 as the "
                  "network is pinched toward a bottleneck by a severe clog.")

    st.markdown("##### Condition number vs. clog severity on vessel " + str(la_vessel or 2))
    sweep_vessel = la_vessel if la_vessel is not None else 2
    sev_grid = np.linspace(0, 0.97, 25)
    conds, fiedlers = [], []
    net_sweep = make_network(net_params)
    for s in sev_grid:
        net_sweep.reset()
        if s > 0:
            apply_clog(net_sweep, sweep_vessel, s)
        conds.append(la.condition_number(net_sweep, p_in=p_in_mmhg * MMHG_TO_PA,
                                         p_out=p_out_mmhg * MMHG_TO_PA)["condition_number"])
        fiedlers.append(la.algebraic_connectivity(net_sweep)["fiedler_value"])
    lc1, lc2 = st.columns(2)
    with lc1:
        show(charts.line_chart(sev_grid, [("Condition number", conds, "blue")], PAL,
                               "Condition number vs. clog severity", "Clog severity", "cond(A)"))
    with lc2:
        show(charts.line_chart(sev_grid, [("Fiedler value", fiedlers, "blue")], PAL,
                               "Algebraic connectivity vs. clog severity", "Clog severity",
                               "Fiedler value"))

    st.markdown("##### Sensitivity matrix: ∂(leaf pressure)/∂(log vessel resistance)")
    st.caption("A finite-difference Jacobian of the linear solve — which sensor is most "
              "informative about which vessel, derived purely from the linear algebra "
              "(no training data needed). Rows = leaf sensors, columns = vessels.")
    sm = la.sensitivity_matrix(net_la, p_in=p_in_mmhg * MMHG_TO_PA, p_out=p_out_mmhg * MMHG_TO_PA)
    show(charts.heatmap(sm["J"], sm["vessel_labels"], sm["leaf_labels"], PAL,
                        "Vessel", "Sensor", "mmHg per unit log(R)"))


# ============================================================= GENETICS ==
with tab_genetics:
    st.subheader("Polygenic risk → anomaly probability & severity")
    st.caption(
        "Each simulated patient gets a genotype over 5 SNPs with literature-plausible risk-allele "
        "frequencies; a standardized polygenic risk score (PRS) shifts their clog/leak probability "
        "and severity (sigmoid-linked), then an RF regressor tries to recover PRS from sensors alone."
    )
    gc1, gc2 = st.columns(2)
    with gc1:
        g_samples = st.slider("Genetic-dataset samples", 300, 6000, 3000, 100, key="g_samples")
    with gc2:
        g_seed = st.number_input("Seed", value=1, step=1, key="g_seed")
    if st.button("Generate genetic dataset & train", type="primary"):
        with st.spinner("Simulating genotypes and patients..."):
            gdf, gnet, panel = generate_genetic_dataset(
                n_samples=g_samples, generations=generations, seed=int(g_seed),
                root_radius=net_params["root_radius"], root_length=net_params["root_length"],
                terminal_resistance=net_params["terminal_resistance"],
                viscosity=net_params["viscosity"],
                p_in_mmhg=p_in_mmhg, p_out_mmhg=p_out_mmhg,
            )
            st.session_state["gdf"] = gdf
            st.session_state["panel"] = panel

    if "gdf" in st.session_state:
        gdf = st.session_state["gdf"]
        panel = st.session_state["panel"]

        st.markdown("##### SNP panel")
        st.dataframe(pd.DataFrame([{"SNP": s.name, "risk-allele freq (q)": s.q, "effect size (β)": s.beta}
                                   for s in panel]), width="stretch")

        with st.spinner("Training location classifier + PRS regressor..."):
            loc_results, reg_results = cached_genetic_models(gdf, int(g_seed))

        m1, m2, m3 = st.columns(3)
        m1.metric("Location-classifier accuracy", f"{loc_results['random_forest']['accuracy']:.3f}", border=True)
        m2.metric("PRS regressor R²", f"{reg_results['r2']:.3f}", border=True)
        m3.metric("PRS regressor MAE", f"{reg_results['mae']:.3f}", border=True)

        g1, g2, g3 = st.columns(3)
        with g1:
            show(charts.histogram([("PRS", gdf["prs"], "blue")], PAL,
                                  "Polygenic risk score", "PRS", nbins=40))
        with g2:
            anomalies = gdf[gdf["label_type"] != "healthy"]
            show(charts.scatter(anomalies["prs"], anomalies["label_severity"], PAL,
                                "Severity vs. risk score", "PRS", "Severity",
                                color_by=anomalies["label_type"], color_map=LABEL_COLORS,
                                height=340))
        with g3:
            show(charts.scatter(reg_results["y_test"], reg_results["y_pred"], PAL,
                                f"Predicted vs. true PRS · R² {reg_results['r2']:.2f}",
                                "True PRS", "Predicted PRS", identity_line=True, height=340))


# ================================================================ REAL ==
with tab_real:
    st.subheader("Real patient cohorts")
    st.markdown(
        "Two sets of real, de-identified people are bundled in `data/`, each with a "
        "measured systolic blood pressure:\n\n"
        "- **UCI Cleveland**: 303 patients referred for heart-disease testing.\n"
        "- **NHANES 2009-2012**: 4,633 adults from the US national health survey.\n\n"
        "No public dataset pairs per-vessel flow readings with a tumor or occlusion "
        "diagnosis. So real data enters here as real blood pressure driving the "
        "simulated vessel network; it does not validate the detector against real "
        "tumor outcomes."
    )

    cohorts = cached_cohorts()
    cohort_names = list(cohorts["cohort"].unique())

    k1, k2, k3 = st.columns(3)
    k1.metric("Real people", f"{len(cohorts):,}", border=True)
    for col, name in zip((k2, k3), cohort_names):
        sub = cohorts[cohorts["cohort"] == name]["systolic_bp_mmhg"]
        col.metric(f"Mean systolic · {name.split(' ')[0]}", f"{sub.mean():.1f} mmHg",
                   help=f"{len(sub):,} people · range {sub.min():.0f} to {sub.max():.0f} mmHg · "
                        f"std {sub.std():.1f}", border=True)

    show(charts.histogram(
        [(name, cohorts[cohorts["cohort"] == name]["systolic_bp_mmhg"], key)
         for name, key in zip(cohort_names, ("blue", "orange"))],
        PAL, "Measured systolic pressure in each cohort", "Systolic pressure (mmHg)",
        vline=(p_in_mmhg, f"sidebar pressure {p_in_mmhg}")))
    st.caption("Each cohort is shown as a share of its own total, because one is fifteen "
               "times larger than the other. Filter and chart these people freely in the "
               "Data Explorer tab.")

    st.markdown("---")
    st.markdown("### Train the detector on real pressures")
    st.info("In **Detection & Statistics**, set *Arterial pressure of each simulated "
            "patient* to one of the real cohorts and click *Generate dataset & train*. "
            "Each simulated patient then gets a pressure measured in a real person. "
            "Both results are recorded, so the Experiment Tracking tab shows how much "
            "real variation costs in accuracy.")

    st.markdown("---")
    st.markdown("### Run a real patient through the trained detector")

    if "ml_results" not in st.session_state:
        st.info("Open **Detection & Statistics** once first to train a detector. This tab "
                "reuses that trained model.")
    else:
        ml_results = st.session_state["ml_results"]
        ml_net_params = st.session_state["ml_net_params"]
        if ml_net_params != net_params:
            st.warning("Sidebar network parameters have changed since the detector was "
                       "trained. Predictions below still use the trained network's shape; "
                       "retrain in Detection & Statistics to match the sidebar.")
        trained_net = make_network(ml_net_params)

        c1, c2 = st.columns(2)
        with c1:
            source = st.radio("Blood pressure source", ["Pick a real person", "Enter my own"])
            if source == "Pick a real person":
                pick_cohort = st.selectbox("Cohort", cohort_names)
                people = cohorts[cohorts["cohort"] == pick_cohort]
                idx = st.selectbox(
                    "Person", people.index,
                    format_func=lambda i: f"age {people.loc[i, 'age']:.0f}, "
                                          f"{people.loc[i, 'sex']}, "
                                          f"{people.loc[i, 'systolic_bp_mmhg']:.0f} mmHg")
                bp_value = float(people.loc[idx, "systolic_bp_mmhg"])
            else:
                bp_value = st.number_input("Your resting systolic BP (mmHg)", 70, 230, 120)
            st.metric("Pressure driving the network", f"{bp_value:.0f} mmHg", border=True)

        with c2:
            real_scenario = st.radio("Condition to simulate for this person",
                                     ["healthy", "clog", "leak"], key="real_scenario")
            real_vessel = None
            real_severity = 0.0
            if real_scenario != "healthy":
                real_vessel = st.slider("Vessel id", 0, trained_net.n_named_vessels - 1, 3,
                                        key="real_vessel")
                real_severity = st.slider("Severity", 0.0, 0.98, 0.7, 0.01, key="real_severity")

        if st.button("Run prediction", type="primary"):
            row, applied = rd.build_patient_reading(
                trained_net, bp_value, p_out=p_out_mmhg * MMHG_TO_PA,
                scenario_type=real_scenario, vessel_id=real_vessel, severity=real_severity,
            )
            x = np.array([[row[c] for c in ml_results["feature_cols"]]])
            rf_model = ml_results["random_forest"]["model"]
            pred_label = rf_model.predict(x)[0]
            proba = rf_model.predict_proba(x)[0]
            classes = rf_model.classes_
            top_idx = np.argsort(proba)[::-1][:5]

            truth = applied["type"] if applied["type"] == "healthy" else \
                f"{applied['type']}_v{applied['vessel_id']}"
            message = (f"Predicted: **{pred_label}** · simulated: **{truth}**"
                       + (f" (severity {applied['severity']:.2f})"
                          if applied["type"] != "healthy" else ""))
            (st.success if pred_label == truth else st.error)(message)
            trained_on = st.session_state["ml_run_params"]["pressure_source"]
            if pred_label != truth and trained_on.startswith("Fixed"):
                st.caption("This detector was trained with one fixed pressure, so a person "
                           "whose pressure differs can be misread. Retrain on a real cohort "
                           "to see the difference.")
            show(charts.horizontal_bars([str(c) for c in classes[top_idx]], proba[top_idx], PAL,
                                        "Most likely conditions", "Probability"))

            with st.expander("Sensor reading fed to the model"):
                st.json(row)

    st.markdown("---")
    st.markdown("### Batch predict from your own CSV")
    st.caption(f"Upload a CSV with columns: {', '.join(rd.sensor_schema(VesselNetwork(generations=generations))['feature_cols'])}")
    uploaded = st.file_uploader("Sensor readings CSV", type="csv")
    if uploaded is not None and "ml_results" in st.session_state:
        user_df = pd.read_csv(uploaded)
        missing = [c for c in st.session_state["ml_results"]["feature_cols"] if c not in user_df.columns]
        if missing:
            st.error(f"Missing expected columns: {missing}")
        else:
            x = user_df[st.session_state["ml_results"]["feature_cols"]].values
            rf_model = st.session_state["ml_results"]["random_forest"]["model"]
            preds = rf_model.predict(x)
            out = user_df.copy()
            out["predicted_label"] = preds
            st.dataframe(out, width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False),
                               file_name="predictions.csv")


# ========================================================= DATA EXPLORER ==
with tab_data:
    st.subheader("Data explorer")
    st.caption("Choose a table, filter it, and chart any column against any other. "
               "Hover for values, drag to zoom, double-click to reset.")

    table_choice = st.radio(
        "Table", ["Simulated patients (current training set)", "Real people (both cohorts)"],
        horizontal=True)
    if table_choice.startswith("Simulated"):
        table = st.session_state["ml_df"].copy()
        table["label_vessel_id"] = table["label_vessel_id"].astype(int)
        leaf_cols = [c for c in table.columns if c.startswith("P_node")]
        table.insert(0, "mean_sensor_pressure_mmhg", table[leaf_cols].mean(axis=1))
        table.insert(1, "lowest_sensor_pressure_mmhg", table[leaf_cols].min(axis=1))
        color_map = LABEL_COLORS
    else:
        table = cached_cohorts().copy()
        color_map = None

    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    category_cols = [c for c in table.columns
                     if c not in numeric_cols or table[c].nunique() <= 40]
    few_cols = [c for c in category_cols if table[c].nunique() <= 3]

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        filtered = table
        filter_cats = [c for c in table.columns
                       if c not in numeric_cols and table[c].nunique() <= 12]
        for slot, col in zip((f1, f2), filter_cats[:2]):
            options = sorted(table[col].dropna().unique())
            keep = slot.multiselect(f"Filter: {col}", options, default=options,
                                    key=f"flt_{table_choice}_{col}")
            filtered = filtered[filtered[col].isin(keep)]
        range_col = f3.selectbox("Range filter on", numeric_cols, key=f"rng_{table_choice}")
        lo, hi = float(table[range_col].min()), float(table[range_col].max())
        if hi > lo:
            sel = f3.slider(f"{range_col} range", lo, hi, (lo, hi), key=f"rngv_{table_choice}_{range_col}")
            filtered = filtered[filtered[range_col].between(*sel)]

    e1, e2, e3, e4 = st.columns(4)
    chart_type = e1.selectbox("Chart", ["Scatter", "Histogram", "Box plot", "Count"])
    st.caption(f"{len(filtered):,} of {len(table):,} rows match the filters.")

    if filtered.empty:
        st.warning("No rows match the filters.")
    elif chart_type == "Scatter":
        x_col = e2.selectbox("X axis", numeric_cols, index=0)
        y_col = e3.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
        c_col = e4.selectbox("Colour by", ["(none)"] + few_cols,
                             index=1 if few_cols else 0)
        plot_df = filtered.dropna(subset=[x_col, y_col])
        show(charts.scatter(plot_df[x_col], plot_df[y_col], PAL, None, x_col, y_col,
                            color_by=None if c_col == "(none)" else plot_df[c_col],
                            color_map=color_map, height=460))
    elif chart_type == "Histogram":
        x_col = e2.selectbox("Column", numeric_cols, index=0)
        c_col = e3.selectbox("Split by", ["(none)"] + few_cols)
        if c_col == "(none)":
            groups = [(x_col, filtered[x_col].dropna(), "blue")]
        else:
            names = sorted(filtered[c_col].dropna().astype(str).unique())
            keys = {n: k for n, k in zip(names, ("blue", "orange", "aqua"))}
            if color_map:  # anomaly types keep their fixed colours
                keys = {n: charts.LABEL_KEYS.get(n, keys[n]) for n in names}
            groups = [(n, filtered[filtered[c_col].astype(str) == n][x_col].dropna(), keys[n])
                      for n in names]
        show(charts.histogram(groups, PAL, None, x_col))
    elif chart_type == "Box plot":
        y_col = e2.selectbox("Value", numeric_cols, index=0)
        g_col = e3.selectbox("Group by", category_cols)
        plot_df = filtered.dropna(subset=[y_col, g_col])
        show(charts.box(plot_df[y_col], plot_df[g_col], PAL, None, g_col, y_col,
                        color_map=color_map))
    else:
        g_col = e2.selectbox("Count by", category_cols)
        show(charts.count_bars(filtered[g_col].dropna(), PAL, None, g_col))

    with st.expander(f"Rows behind the chart ({len(filtered):,})"):
        st.dataframe(filtered, width="stretch", height=320, hide_index=True)
        st.download_button("Download these rows (CSV)", filtered.to_csv(index=False),
                           file_name="explorer_rows.csv", mime="text/csv")


# ============================================================== TRACKING ==
with tab_track:
    st.subheader("Experiment tracking (MLflow, stored locally or in Databricks)")
    st.markdown(
        "Each time the detector is trained in the **Detection & Statistics** tab, "
        "its settings, scores and the trained model are saved as one *run*, so "
        "results can be compared and reproduced later instead of disappearing "
        "when the app closes. Runs are stored in a local file by default, or in "
        "a Databricks workspace if you connect one below."
    )

    opt1, opt2 = st.columns(2)
    opt1.checkbox("Record every new training run", value=True, key="track_enabled")
    opt2.checkbox("Store the trained model with each run (about 8 MB each)",
                  value=True, key="track_model")

    backend = st.radio("Where runs are stored", ["Local file", "Databricks workspace"],
                       horizontal=True, key="track_backend")

    if backend == "Databricks workspace":
        dbx = st.session_state.get("dbx")
        if dbx:
            st.success(f"Connected to {dbx['host']} · experiment `{dbx['experiment']}`")
            st.markdown(f"[Open the workspace experiments page]({dbx['host']}/ml/experiments)")
            if st.button("Disconnect"):
                del st.session_state["dbx"]
                st.session_state.pop("dbx_meta", None)
                st.rerun()

            # ---- send a data table to the workspace ----
            with st.container(border=True):
                st.markdown("**Send a data table to Databricks**")
                st.caption("Creates a real table under Catalog, so the data can be queried "
                           "in a notebook or charted on a Databricks dashboard. An existing "
                           "table with the same name is replaced.")
                if "dbx_meta" not in st.session_state:
                    try:
                        st.session_state["dbx_meta"] = dict(
                            warehouses=dbt.list_warehouses(dbx["host"], dbx["token"]),
                            schema=dbt.default_schema(dbx["host"], dbx["token"]))
                    except dbt.DatabricksError as e:
                        st.session_state["dbx_meta"] = dict(warehouses=[], schema="workspace.default",
                                                            error=str(e))
                meta = st.session_state["dbx_meta"]
                if meta.get("error"):
                    st.error(meta["error"])
                if not meta["warehouses"]:
                    st.warning("No SQL warehouse was found in this workspace. Create one under "
                               "SQL Warehouses in Databricks, then disconnect and reconnect here.")
                else:
                    u1, u2, u3 = st.columns(3)
                    which = u1.selectbox("Table to send", list(UPLOAD_TABLES))
                    schema = u2.text_input("Destination (catalog.schema)", value=meta["schema"])
                    wh = u3.selectbox("SQL warehouse", meta["warehouses"],
                                      format_func=lambda w: f"{w['name']} ({w['state'].lower()})")
                    table_name = UPLOAD_TABLES[which]
                    if st.button(f"Send to Databricks as {table_name}", type="primary"):
                        to_send = (st.session_state["ml_df"] if which.startswith("Simulated")
                                   else cached_cohorts())
                        bar = st.progress(0.0, text="Starting the warehouse and creating the table. "
                                                    "A stopped warehouse can take a few minutes.")
                        try:
                            n = dbt.upload_dataframe(
                                to_send, dbx["host"], dbx["token"], wh["id"], schema.strip(),
                                table_name,
                                progress=lambda f: bar.progress(f, text=f"Sending rows: {f:.0%}"))
                            bar.empty()
                            st.success(f"Sent {n:,} rows to `{schema.strip()}.{table_name}`.")
                            st.markdown(f"[Open the table in Databricks]"
                                        f"({dbt.table_url(dbx['host'], schema.strip(), table_name)})")
                        except dbt.DatabricksError as e:
                            bar.empty()
                            st.error(str(e))
        else:
            st.info("Not connected yet. Until you connect, runs keep going to the local file.")
            with st.form("dbx_connect"):
                host = st.text_input(
                    "Workspace URL",
                    placeholder="https://dbc-xxxxxxxx-xxxx.cloud.databricks.com")
                token = st.text_input(
                    "Personal access token", type="password",
                    help="In Databricks: User Settings > Developer > Access tokens. "
                         "Held in memory for this session only; never written to disk.")
                experiment = st.text_input(
                    "Experiment path", value=tracking.DEFAULT_DATABRICKS_EXPERIMENT,
                    help="An absolute workspace path. /Users/<your login email>/hemodynamics "
                         "also works. It is created if it does not exist.")
                if st.form_submit_button("Connect", type="primary"):
                    try:
                        with st.spinner("Contacting the workspace..."):
                            tracking.activate("databricks", host.strip().rstrip("/"),
                                              token, experiment.strip())
                        st.session_state["dbx"] = dict(
                            host=host.strip().rstrip("/"), token=token.strip(),
                            experiment=experiment.strip())
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not connect: {e}")
    else:
        st.caption(
            "Stored in `outputs/mlflow.db`. For MLflow's own full interface run "
            "`mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db` in the project folder."
        )

    st.markdown("### Run history")
    try:
        tracker_label, exp_name = current_tracker()
        runs = tracking.list_runs(exp_name)
        while tracking.notices:
            st.warning(tracking.notices.pop(0))
    except Exception as e:
        runs = pd.DataFrame()
        st.error(f"Could not read runs: {e}")

    if runs.empty:
        st.info("No runs recorded here yet. Train a detector in the Detection & Statistics tab.")
    else:
        metric_cols = [c for c in ["roc_auc", "pr_auc", "rf_accuracy", "svm_accuracy",
                                   "sensitivity", "specificity", "best_threshold"]
                       if c in runs.columns]
        k1, k2, k3 = st.columns(3)
        k1.metric("Runs recorded", len(runs), border=True)
        if "roc_auc" in runs:
            best = runs.loc[runs["roc_auc"].idxmax()]
            k2.metric("Best ROC AUC", f"{best['roc_auc']:.4f}", border=True)
            k3.metric("Best run", str(best["run_id"])[:8], border=True)

        front = ["started", "run_id"] + metric_cols
        st.dataframe(runs[front + [c for c in runs.columns if c not in front and c != "run_name"]],
                     width="stretch", hide_index=True)
        st.download_button("Download run history (CSV)", runs.to_csv(index=False),
                           file_name="run_history.csv", mime="text/csv")

        st.markdown("### Compare runs")
        param_cols = [c for c in runs.columns
                      if c not in metric_cols + ["run_id", "started", "run_name"]
                      and pd.api.types.is_numeric_dtype(runs[c]) and runs[c].nunique() > 1]
        if len(runs) < 2 or not param_cols:
            st.caption("Train again with a different setting (for example more sensor "
                       "noise) and a comparison chart appears here.")
        else:
            cx, cy = st.columns(2)
            default_x = param_cols.index("pressure_noise_mmhg") if "pressure_noise_mmhg" in param_cols else 0
            x_col = cx.selectbox("Setting (x axis)", param_cols, index=default_x)
            y_col = cy.selectbox("Score (y axis)", metric_cols)
            show(charts.scatter(runs[x_col], runs[y_col], PAL,
                                f"{y_col} across {len(runs)} recorded runs", x_col, y_col))
