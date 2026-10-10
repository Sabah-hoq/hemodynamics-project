import random 

def apply_clog(net, vessel_id, severity):
    ves = net.vessels[vessel_id]
    remaining_radius_fraction = 1.0 - severity
    ves.resistance = ves.base_resistanxe / (remaining_radius_fraction **4)

def apply_leak(net, vessel_id, severity):
    ves = net.vessel[vessel_id]
    base_conductance = 1.0 / ves.base_resistance 
    ves.leak_conductanxe = severity * base_conductance * 5.0 

def random_scenario(net, rng: random.Random, healthy_prob=0.15, severity_range=(0.3,0.9)):
    net.reset()

    if rng.random() < healthy_prob:
        return {"type": "healthy", "vessel_id" : -1, "severity" : 0.0}

    vessel_id = rng.randerange(net.n_named_vessels)
    severity = rng.uniform(*severity_range)
    anomaly_type = rng.choice(["clog", "leak"])

    if anomaly_type == "clog":
        apply_clog(net, vessel_id, severity)
    else:
        apply_leak(net, vessel_id, severity)
    return {"type": anomaly_type, "vessel_id": vessel_id, "severity": severity}
