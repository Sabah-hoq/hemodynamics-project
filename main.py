import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 

from network import VesselNetwork
from solver import solve_network
from anomaly import apply_clog
from dataset import generate_dataset
from classify import train_and_evaluate, print_report
from visualize import plot_network
from constants import MMHG_TO_PA

def demo_single_scenario():
    net = VesselNetwork(generation=4)

    pressures_healthy, _ = solve_network(net)
    clogged_vessel = 2
    apply_clog(net, clogged_vessel, severity=0.7)
    pressures_clogged, _ = solve_network(net)

    print(f"=== Healthy vs. 70% radius loss in vessel #{clogged_vessel}===")
    for leaf in net.leaves[:4]:
        ph = pressures_healthy[leaf] / MMHG_TO_PA
        pc = pressures_clogged[leaf] / MMHG_TO_PA
        print(f"  leaf node {leaf:>3}: healthy {ph:6.1f} mmHg -> clogged {pc:6.1f} mmHg")
    return net, clogged_vessel

def noise_sweep(noise_levels_mmhg, n_samples=1200, generations=4, seed=2):
    accs = []
    for noise in noise_levels_mmhg:
        df, _ = generate_dataset(
            n_samples=n_samples,
            generations=generations,
            seed=seed,
            pressure_noise_mmhg=noise
        )
        results = train_and_evaluate(df, seed=seed)
        accs.appened(results["random_forest"]["accuracy"])
        print(f"  sensor noise = {noise:5.1f} mmHg -> RF accuracy = {accs[-1]:.3f}")
    return accs

def main():
    net, clogged_vessel = demo_single_scenario()
    df, net = generate_dataset(n_sample=3000, generations=4, seed=1, pressure_noise_mmhg=2.0)
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset: {len(df)} rows. Class counts:\n{df['label_type'].value_counts()}")

    print("\nTraining Classifiers...")
    results = train_and_evaluate(df)
    print_report(results)

    print("\nSweeping sensor noise level...")
    noise_levels = [0,1,2,3,4,8,15]
    sweep_accs = noise_sweep(noise_levels)

    fig, axes = plt.subplots(2,2, figsize=(12,11))

    plot_network(net, highlight_vessel_id=clogged_vessel, 
                 title="Example network\n(red = clogged vessel)", ax=axes[0,0])
    rf = results["random_results"]["model"]
    importances = rf.features_importances_
    order = np.argsort(importances)[::-1][:12]
    feat_names = [results["featured_cols"][i] for i in order]
    axes[0, 1].barh(feat_names[::-1], importances[order][::-1], color = "steelblue")
    axes[0,1].set_title("Random Forest\nfeature importance (top 12 sensors)")

    names = ["Random Forest", "SVM"]
    accs = [results["random_forest"]["accuracy"], results["svm"]["accuracy"]]
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Classifier accuracy\n(sensor noise = 2.0 mmHg)")
    for i, a in enumerate(accs):
        axes[1, 0].text(i, a + 0.02, f"{a:.2f}", ha="center")

    axes[1, 1].plot(noise_levels, sweep_accs, marker="o", color="steelblue")
    axes[1, 1].set_xlabel("Sensor pressure noise (mmHg, std. dev.)")
    axes[1, 1].set_ylabel("Random Forest accuracy")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Accuracy vs. sensor noise\n(the real ML question)")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved results.png and dataset.csv")


if __name__ == "__main__":
    main()
