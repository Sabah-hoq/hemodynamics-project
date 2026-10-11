import random 

import numpy as np 
import pandas as pd

from network import VesselNetwork
from solver import solve_network
from anomaly import random_scenario
from genetics import DEFAULT_SNP_PANEL, genetic_scenario
from constants import MMHG_TO_PA


def _sensor_readings(pressure, flows, sensor_nodes, root_children,
                     np_rng, pressure_noise_mmhg, flow_noise_frac):
    row = {}
    for node in sensor_nodes:
        true_p_mmhg = pressure[node] / MMHG_TO_PA
        row[f"P_node{node}_mmHg"] = true_p_mmhg + np_rng.normal(0, pressure_noise_mmhg)
    for vid in root_children:
        true_q = flows[vid]
        row[f"Q_vessel{vid}"] = true_q * (1 + np_rng.normal(0, flow_noise_frac))
    return row

def generate_dataset(n_samples=3000, generations=4, seed=0, pressure_noise_mmhg=2.0, flow_noise_frac=0.03):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    net = VesselNetwork(generations=generations)

    sensor_nodes = net.leaves 
    root_children = [v.id for v in net.vessels if v.u == 0]

    rows = []
    for _ in range(n_samples):
        scenario = random_scenario(net, rng)
        pressures, flows = solve_network(net)
        row = _sensor_readings(pressures, flows, sensor_nodes, root_children,
                               np_rng, pressure_noise_mmhg, flow_noise_frac)
        row["label_type"] = scenario["type"]
        row["label_vessel_id"] = scenario["vessel_id"]
        row["label_severity"] = scenario["severity"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, net

def generate_genetic_dataset(n_samples=3000, generations=4, seed=0,
                             pressure_noise_mmgh=2.0, flow_noise_frac=0.03,
                             panel=None):
    panel = panel or DEFAULT_SNP_PANEL
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    net = VesselNetwork(generations=generations)

    sensor_nodes = net.leaves 
    root_children = [v.id for v in net.vessels if v.u == 0]

    rows = []
    for _ in range(n_samples):
        scenario = genetic_scenario(net, rng, np_rng, panel=panel)
        pressures, flows = solve_network(net)

        row = _sensor_readings(pressures, flows, sensor_nodes, root_children,
                               np_rng, pressure_noise_mmgh, flow_noise_frac)

        for i, g in enumerate(scenario["genotype"]):
            row[f"genotype_snp{i}"] = int(g)
        row["prs"] = scenario["prs"]
        row["label_type"] = scenario["type"]
        row["label_vessel_id"] = scenario["vessel_id"]
        row["label_severity"] = scenario["severity"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, net, panel

if __name__ == "__main__":
    df, net = generate_dataset(n_samples=3000)
    df.to_csv("dataset.csv", index=False)
    print(df.head())
    print(f"\nGenerated {len(df)} samples, {len(net.vessels)} vessels, "
          f"{len(net.leaves)} sensor nodes.")
    print(df["label_type"].value_counts())