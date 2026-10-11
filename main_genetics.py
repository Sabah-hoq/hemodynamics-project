import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 

from dataset import generate_genetic_dataset
from classify import train_and_evaluate, train_prs_regressor

def main():
    print("Generating genetic-linked dataset...")
    df, net, panel = generate_genetic_dataset(n_samples=3000, seed=1)
    df.to_cvs("dataset_genetics.csv", index=False)
    print(f"{len(df)} simulated patients. Type counts:\n{df['label_type'].value_counts()}")

    print("\n--- SNP panel ---")
    for snp in panel: 
        print(f"  {snp.name:28s} risk-allele freq q={snp.q:.2f}   beta={snp.beta:+.2f}")
    
    print("\nTraining anomaly-location classifier (same task as main.py)...") 
    loc_results = train_and_evaluate(df)
    print(f"Location classifier accuracy: {loc_results['random_forest']['accuracy']:.3f}")

    print("\nTraining PRS regressor (new: predict genetic risk from sensors)")
    reg_results = train_prs_regressor(df)
    print(f"PRS regressor R^2: {reg_results['r2']:.3f}   MAE: {reg_results['mae']:.3f}")

    fig, axes = plt.subplots(2,2, figsize=(13,11))

    snp_idx = 1
    snp = panel[snp_idx]
    counts = df[f"genotype_snp{snp_idx}"].value_counts().reindex([0,1,2],fill_value=0)
    observed = counts/counts.sum()

    p = 1 - snp.q
    theoretical = [p**2, 2* p * snp.q, snp.q**2]
    x = np.arrange(3)
    width = 0.35
    axes[0,0].bar(x - width/2, observed.values, width, label="hardy-Weinberg\ntheory",
                  color="indianred")
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(["0 copies", "1 copy", "2 copies"])
    axes[0,0].set_title(f"{snp.name}\ngenotype distribution (q ={snp.q})")
    axes[0,0].lengend()

    axes[0, 1].hist(df["prs"], bins=40, color="steelblue")
    axes[0, 1].set_title("Polygenic risk score distribution\n(standardized: mean 0, sd 1)")
    axes[0, 1].set_xlabel("PRS")

    # 3. severity vs PRS, colored by anomaly type
    anomalies = df[df["label_type"] != "healthy"]
    colors = anomalies["label_type"].map({"clog": "tab:red", "leak": "tab:blue"})
    axes[1, 0].scatter(anomalies["prs"], anomalies["label_severity"],
                        c=colors, alpha=0.4, s=10)
    axes[1, 0].set_xlabel("PRS")
    axes[1, 0].set_ylabel("Severity")
    axes[1, 0].set_title("Severity vs. genetic risk score\n(red=clog, blue=leak)")

    # 4. predicted vs actual PRS (the regression result)
    axes[1, 1].scatter(reg_results["y_test"], reg_results["y_pred"],
                        alpha=0.4, s=10, color="steelblue")
    lims = [min(reg_results["y_test"].min(), reg_results["y_pred"].min()),
            max(reg_results["y_test"].max(), reg_results["y_pred"].max())]
    axes[1, 1].plot(lims, lims, "k--", alpha=0.5, label="perfect prediction")
    axes[1, 1].set_xlabel("True PRS")
    axes[1, 1].set_ylabel("Predicted PRS")
    axes[1, 1].set_title(f"Predicting genetic risk from sensors alone\nR² = {reg_results['r2']:.2f}")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("results_genetics.png", dpi=150)
    print("\nSaved results_genetics.png and dataset_genetics.csv")


if __name__ == "__main__":
    main()