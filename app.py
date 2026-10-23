"""
Interactive dashboard for the hemodynamics flow-anomaly project.

    streamlit run app.py

Five tabs: a live network explorer, the ML detector + its statistics,
the linear-algebra diagnostics on the solver's matrix, the genetics/PRS
model, and a real-human-data tab that drives the simulator with real
clinical blood pressure and runs a real patient through the trained model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from network import VesselNetwork
from solver import solve_network
from anomaly import apply_clog, apply_leak
from dataset import generate_dataset, generate_genetic_dataset
from classify import train_and_evaluate, train_prs_regressor, build_location_label
from visualize import plot_network
from constants import MMHG_TO_PA
import stats
import linalg_analysis as la
import real_data as rd

st.set_page_config(page_title="Hemodynamics Flow-Anomaly Explorer", layout="wide")


# --------------------------------------------------------------- caching --

@st.cache_data(show_spinner=False)
def cached_dataset(n_samples, generations, seed, pressure_noise, flow_noise,
                   root_radius, root_length, terminal_resistance, viscosity,
                   p_in_mmhg, p_out_mmhg, healthy_prob):
    df, _ = generate_dataset(
        n_samples=n_samples, generations=generations, seed=seed,
        pressure_noise_mmhg=pressure_noise, flow_noise_frac=flow_noise,
        root_radius=root_radius, root_length=root_length,
        terminal_resistance=terminal_resistance, viscosity=viscosity,
        p_in_mmhg=p_in_mmhg, p_out_mmhg=p_out_mmhg, healthy_prob=healthy_prob,
    )
    return df


@st.cache_data(show_spinner=False)
def cached_real_bp_dataset(n_samples, generations, seed, pressure_noise, flow_noise):
    real_df = rd.load_real_patients()
    df, _ = rd.generate_dataset_from_real_bp(
        n_samples=n_samples, generations=generations, seed=seed,
        pressure_noise_mmhg=pressure_noise, flow_noise_frac=flow_noise,
        real_df=real_df,
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
def cached_real_patients():
    return rd.load_real_patients()


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

st.title("Cardiovascular Flow-Anomaly Simulation")
st.caption(
    "Physics 207 project — a binary vascular tree solved as a resistor network, "
    "perturbed with synthetic clogs/leaks, and localized with a trained classifier."
)

tab_explore, tab_ml, tab_linalg, tab_genetics, tab_real = st.tabs(
    ["🩺 Network Explorer", "📊 ML Detection & Statistics", "🧮 Linear Algebra",
     "🧬 Genetics / PRS", "🧑‍⚕️ Real Patient Data"]
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
        fig, ax = plt.subplots(figsize=(7, 5.5))
        plot_network(net, highlight_vessel_id=vessel_id, title=f"{scenario_type} scenario", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

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

    c1, c2 = st.columns(2)
    with c1:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.bar(sensor_df["leaf node"].astype(str), sensor_df["Δ (mmHg)"], color="indianred")
        ax2.set_title("Pressure drop vs. healthy baseline")
        ax2.set_ylabel("Δ mmHg")
        ax2.axhline(0, color="black", linewidth=0.8)
        st.pyplot(fig2)
        plt.close(fig2)
    with c2:
        root_children = [v.id for v in net.vessels if v.u == 0]
        st.markdown("**Root-vessel flow (m³/s)**")
        for vid in root_children:
            st.metric(f"Q vessel {vid}", f"{flows[vid]:.3e}",
                      delta=f"{flows[vid]-healthy_flows[vid]:.2e}")


# =================================================================== ML ==
with tab_ml:
    st.subheader("Synthetic dataset → trained detector")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_samples = st.slider("Samples", 300, 6000, 3000, 100)
        seed = st.number_input("Random seed", value=1, step=1)
    with c2:
        pressure_noise = st.slider("Sensor pressure noise (mmHg std)", 0.0, 15.0, 2.0, 0.5)
        flow_noise = st.slider("Sensor flow noise (fraction std)", 0.0, 0.2, 0.03, 0.01)
    with c3:
        healthy_prob = st.slider("Fraction healthy", 0.0, 0.5, 0.15, 0.05)
        run_btn = st.button("Generate dataset & train", type="primary")

    state_key = "ml_df"
    if run_btn or state_key not in st.session_state:
        with st.spinner("Simulating patients and solving the network..."):
            df = cached_dataset(
                n_samples, generations, int(seed), pressure_noise, flow_noise,
                net_params["root_radius"], net_params["root_length"],
                net_params["terminal_resistance"], net_params["viscosity"],
                p_in_mmhg, p_out_mmhg, healthy_prob,
            )
            st.session_state[state_key] = df

    df = st.session_state[state_key]
    st.caption(f"{len(df)} simulated patients · class balance: "
              f"{df['label_type'].value_counts().to_dict()}")

    with st.spinner("Training Random Forest + SVM..."):
        results = _train_bundle(df, int(seed))
    st.session_state["ml_results"] = results
    st.session_state["ml_net_params"] = dict(net_params)

    st.markdown("### Multi-class localization (which vessel, if any)")
    m1, m2 = st.columns(2)
    m1.metric("Random Forest accuracy", f"{results['random_forest']['accuracy']:.3f}")
    m2.metric("SVM accuracy", f"{results['svm']['accuracy']:.3f}")

    rf = results["random_forest"]["model"]
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    feat_names = [results["feature_cols"][i] for i in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feat_names[::-1], importances[order][::-1], color="steelblue")
    ax.set_title("Random Forest feature importance (top 15 sensors)")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Binary detector: \"is flow stopping anywhere?\" — thresholded & scored")
    bd = results["binary_detector"]
    roc = stats.roc_pr_analysis(bd["y_test"], bd["y_proba"])
    ts = stats.threshold_sweep(bd["y_test"], bd["y_proba"])

    col_roc, col_pr, col_thresh = st.columns(3)
    with col_roc:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(roc["fpr"], roc["tpr"], color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curve (AUC={roc['roc_auc']:.3f})")
        st.pyplot(fig); plt.close(fig)
    with col_pr:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(roc["recall"], roc["precision"], color="indianred")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR curve (AUC={roc['pr_auc']:.3f})")
        st.pyplot(fig); plt.close(fig)
    with col_thresh:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(ts["thresholds"], ts["sensitivity"], label="sensitivity", color="steelblue")
        ax.plot(ts["thresholds"], ts["specificity"], label="specificity", color="indianred")
        ax.axvline(ts["best_threshold"], color="gray", linestyle="--",
                  label=f"best t={ts['best_threshold']:.2f}")
        ax.set_xlabel("Probability threshold"); ax.legend(fontsize=8)
        ax.set_title("Threshold sweep (Youden's J)")
        st.pyplot(fig); plt.close(fig)

    st.caption(
        f"At the Youden's-J-optimal threshold ({ts['best_threshold']:.2f}), the detector gets "
        f"{ts['best_sensitivity']:.1%} sensitivity and {ts['best_specificity']:.1%} specificity "
        "on held-out synthetic patients."
    )

    st.markdown("### Is this real, or did we get a lucky split?")
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.spinner("5-fold cross-validating..."):
            X = df[results["feature_cols"]].values
            y_bin = (df["label_type"] != "healthy").astype(int).values
            from sklearn.ensemble import RandomForestClassifier
            cv = stats.cross_validated_accuracy(
                RandomForestClassifier(n_estimators=150, random_state=int(seed)), X, y_bin, k=5)
        st.metric("5-fold CV accuracy", f"{cv['mean']:.3f}",
                  help=f"95% CI [{cv['lo']:.3f}, {cv['hi']:.3f}]")
        st.caption(f"95% CI: [{cv['lo']:.3f}, {cv['hi']:.3f}]")
    with s2:
        boot = stats.bootstrap_accuracy_ci(
            results["random_forest"]["y_test"], results["random_forest"]["y_pred"], n_boot=1000)
        st.metric("Bootstrap test accuracy", f"{boot['mean']:.3f}",
                  help=f"95% CI [{boot['lo']:.3f}, {boot['hi']:.3f}]")
        st.caption(f"95% CI: [{boot['lo']:.3f}, {boot['hi']:.3f}]")
    with s3:
        mc = stats.mcnemar_test(results["random_forest"]["y_test"],
                                results["random_forest"]["y_pred"], results["svm"]["y_pred"])
        st.metric("RF vs SVM (McNemar p)", f"{mc['p_value']:.4f}")
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
                    p_in_mmhg, p_out_mmhg, healthy_prob,
                )
                r = _train_bundle(d, int(seed))
                accs.append(r["random_forest"]["accuracy"])
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(noise_levels, accs, marker="o", color="steelblue")
        ax.set_xlabel("Sensor pressure noise (mmHg std)"); ax.set_ylabel("RF accuracy")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)


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
        st.metric("Condition number of A", f"{cn['condition_number']:.1f}")
        st.caption("How much a tiny error in resistance/pressure gets amplified in the solved "
                  "pressures. Near 1 is well-posed; rising sharply signals numerical fragility.")
    with c2:
        ac = la.algebraic_connectivity(net_la)
        st.metric("Fiedler value (algebraic connectivity)", f"{ac['fiedler_value']:.3e}")
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
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 3.5))
    axA.plot(sev_grid, conds, color="steelblue", marker=".")
    axA.set_xlabel("Clog severity"); axA.set_ylabel("cond(A)")
    axA.set_title("Condition number vs. clog severity")
    axB.plot(sev_grid, fiedlers, color="indianred", marker=".")
    axB.set_xlabel("Clog severity"); axB.set_ylabel("Fiedler value")
    axB.set_title("Algebraic connectivity vs. clog severity")
    st.pyplot(fig); plt.close(fig)

    st.markdown("##### Sensitivity matrix: ∂(leaf pressure)/∂(log vessel resistance)")
    st.caption("A finite-difference Jacobian of the linear solve — which sensor is most "
              "informative about which vessel, derived purely from the linear algebra "
              "(no training data needed). Rows = leaf sensors, columns = vessels.")
    sm = la.sensitivity_matrix(net_la, p_in=p_in_mmhg * MMHG_TO_PA, p_out=p_out_mmhg * MMHG_TO_PA)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(sm["J"], aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(sm["J"])), vmax=np.max(np.abs(sm["J"])))
    ax.set_yticks(range(len(sm["leaf_labels"])))
    ax.set_yticklabels(sm["leaf_labels"], fontsize=7)
    ax.set_xticks(range(0, len(sm["vessel_labels"]), max(1, len(sm["vessel_labels"]) // 20)))
    ax.set_xticklabels(sm["vessel_labels"][::max(1, len(sm["vessel_labels"]) // 20)],
                       fontsize=7, rotation=90)
    plt.colorbar(im, ax=ax, label="mmHg per unit log(R)")
    st.pyplot(fig); plt.close(fig)


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
            loc_results = train_and_evaluate(gdf, seed=int(g_seed))
            reg_results = train_prs_regressor(gdf, seed=int(g_seed))

        m1, m2, m3 = st.columns(3)
        m1.metric("Location-classifier accuracy", f"{loc_results['random_forest']['accuracy']:.3f}")
        m2.metric("PRS regressor R²", f"{reg_results['r2']:.3f}")
        m3.metric("PRS regressor MAE", f"{reg_results['mae']:.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(gdf["prs"], bins=40, color="steelblue")
        axes[0].set_title("PRS distribution"); axes[0].set_xlabel("PRS")

        anomalies = gdf[gdf["label_type"] != "healthy"]
        colors = anomalies["label_type"].map({"clog": "tab:red", "leak": "tab:blue"})
        axes[1].scatter(anomalies["prs"], anomalies["label_severity"], c=colors, alpha=0.4, s=10)
        axes[1].set_title("Severity vs. PRS (red=clog, blue=leak)")
        axes[1].set_xlabel("PRS"); axes[1].set_ylabel("Severity")

        axes[2].scatter(reg_results["y_test"], reg_results["y_pred"], alpha=0.4, s=10, color="steelblue")
        lims = [min(reg_results["y_test"].min(), reg_results["y_pred"].min()),
               max(reg_results["y_test"].max(), reg_results["y_pred"].max())]
        axes[2].plot(lims, lims, "k--", alpha=0.5)
        axes[2].set_title(f"Predicted vs. true PRS (R²={reg_results['r2']:.2f})")
        axes[2].set_xlabel("True PRS"); axes[2].set_ylabel("Predicted PRS")
        st.pyplot(fig); plt.close(fig)


# ================================================================ REAL ==
with tab_real:
    st.subheader("Real human clinical data")
    st.markdown(
        "`data/uci_heart_disease.csv` is the real UCI **Cleveland Heart Disease** dataset: "
        "303 real, de-identified patients with a clinically measured resting blood pressure. "
        "No public dataset pairs real per-vessel flow sensors with a tumor/occlusion diagnosis, "
        "so this tab is deliberately honest about what \"real data\" can mean here: real human "
        "blood pressure driving the same physics engine and the same trained classifier, "
        "not a validation against real tumor outcomes."
    )

    real_df = cached_real_patients()
    real_summary = rd.real_bp_summary(real_df)

    rcol1, rcol2 = st.columns([2, 1])
    with rcol1:
        st.markdown("##### Real patient table")
        st.dataframe(
            real_df[["age", "sex_label", "resting_systolic_bp_mmhg", "cholesterol_mg_dl",
                    "max_heart_rate", "diagnosis"]],
            width="stretch", height=250,
        )
    with rcol2:
        st.metric("Real patients", real_summary["n"])
        st.metric("Mean resting systolic BP", f"{real_summary['mean']:.1f} mmHg")
        st.caption(f"range {real_summary['min']:.0f}–{real_summary['max']:.0f} mmHg, "
                  f"std {real_summary['std']:.1f}")

    st.markdown("##### Real BP distribution vs. this app's simulated Pᵢₙ")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(real_summary["values"], bins=25, alpha=0.6, label="real patients (UCI)", color="indianred")
    ax.axvline(p_in_mmhg, color="steelblue", linewidth=2, label=f"current simulator P_in = {p_in_mmhg}")
    ax.set_xlabel("Systolic pressure (mmHg)"); ax.legend()
    st.pyplot(fig); plt.close(fig)

    st.markdown("---")
    st.markdown("### Run a real patient through the trained detector")

    if "ml_results" not in st.session_state:
        st.info("Open **ML Detection & Statistics** once first to train a detector — this tab "
               "reuses that trained model.")
    else:
        ml_results = st.session_state["ml_results"]
        ml_net_params = st.session_state["ml_net_params"]
        if ml_net_params != net_params:
            st.warning("Sidebar network parameters have changed since the detector was trained "
                      "on the ML tab. Predictions below still use the *trained* network's shape — "
                      "retrain on the ML tab to match the current sidebar settings.")
        trained_net = make_network(ml_net_params)
        schema = rd.sensor_schema(trained_net)

        c1, c2 = st.columns(2)
        with c1:
            source = st.radio("Blood pressure source", ["Pick a real patient", "Enter my own"])
            if source == "Pick a real patient":
                idx = st.selectbox(
                    "Real patient (row)", real_df.index,
                    format_func=lambda i: f"#{i}: age {real_df.loc[i,'age']}, "
                                          f"{real_df.loc[i,'sex_label']}, "
                                          f"{real_df.loc[i,'resting_systolic_bp_mmhg']} mmHg, "
                                          f"{real_df.loc[i,'diagnosis']}")
                bp_value = float(real_df.loc[idx, "resting_systolic_bp_mmhg"])
            else:
                bp_value = st.number_input("Your resting systolic BP (mmHg)", 70, 220, 120)
            st.metric("Blood pressure used as Pᵢₙ", f"{bp_value:.0f} mmHg")

        with c2:
            real_scenario = st.radio("Scenario to simulate for this patient",
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

            st.success(f"Predicted: **{pred_label}**  (ground truth injected: {applied['type']}"
                      + (f" on vessel {applied['vessel_id']}, severity {applied['severity']:.2f}"
                         if applied["type"] != "healthy" else "") + ")")
            st.markdown("**Top class probabilities**")
            st.dataframe(pd.DataFrame({"class": classes[top_idx], "probability": proba[top_idx]})
                        .style.format({"probability": "{:.3f}"}), width="stretch")

            with st.expander("Sensor reading fed to the model"):
                st.json(row)

    st.markdown("---")
    st.markdown("### Train on real-BP-calibrated synthetic data")
    st.caption(
        "Instead of a single fixed P_in, bootstrap-resample P_in from the real patient BP "
        "distribution above (with jitter) for every simulated training patient — folding real "
        "physiological variability into the training set itself, not just a single live query."
    )
    if st.button("Generate real-BP-calibrated dataset & compare accuracy"):
        with st.spinner("Simulating patients with real-BP-sampled boundary conditions..."):
            df_real_cal = cached_real_bp_dataset(2000, generations, 3, pressure_noise, 0.03)
            results_real_cal = _train_bundle(df_real_cal, 3)
        c1, c2 = st.columns(2)
        c1.metric("Fixed-P_in RF accuracy",
                 f"{st.session_state['ml_results']['random_forest']['accuracy']:.3f}"
                 if "ml_results" in st.session_state else "train ML tab first")
        c2.metric("Real-BP-calibrated RF accuracy", f"{results_real_cal['random_forest']['accuracy']:.3f}")
        st.caption("A gap here shows how much the detector's reported accuracy depends on "
                  "training with one fixed arterial pressure vs. real population variability.")

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
