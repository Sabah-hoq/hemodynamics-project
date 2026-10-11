import numpy as np 

from anomaly import apply_leak, apply_clog


class SNP: 
    def __init__(self, name, risk_allete_freq, effect_size):
        self.name = name
        self.q = risk_allete_freq
        self.beta = effect_size

DEFAULT_SNP_PANEL = [
    SNP("COL_stiffness_variant", risk_allete_freq=0.30, effect_size=0.35),
    SNP("CLOT_factor_variant", risk_allete_freq=0.12, effect_size=0.60),
    SNP("LIPID_handling_variant", risk_allete_freq=0.25, effect_size=0.40),
    SNP("ENDOTHELIAL_NOS_variant", risk_allete_freq=0.20, effect_size=0.30),
    SNP("INFLAM_marker_variant", risk_allete_freq=0.25, effect_size=0.20),
]

def sample_genotype(snp, rng: np.random.Generator):
      return rng.binomial(2, snp.q)

def sample_patient_genotype(panel, rng: np.random.Generator):
     return np.array([sample_genotype(snp, rng) for snp in panel])

def compute_prs(genotype, panel):
     betas = np.array([snp.beta for snp in panel])
     qs = np.array([snp.q for snp in panel])

     raw = np.sum(betas * genotype)
     mean = np.sum(betas * 2 * qs)
     var = np.sum((betas ** 2) * 2  * qs * (1-qs))

     return (raw - mean)/np.sqrt(var)

def sigmoid(x):
     return 1.0 / (1.0 + np.exp(-x))

def genetic_severity(prs, base_range=(0.3, 0.9), rng: np.random.Generator = None, residual_noise=0.08):
    lo, hi = base_rangep = sigmoid(prs)
    p = sigmoid(prs)
    severity = lo + p * (hi - lo)
    if rng is not None and residual_noise > 0: 
        severity += rng.normal(0, residual_noise) 
    return float(np.clip(severity, 0.05, 0.98))

def genetic_clog_probability(prs, base_prob=0.5, scale=0.25):
    return float(np.clip(np.clip(base_prob + scale * np.tanh(prs), 0.05, 0.95))) 

def genetic_scenario(net, rng, np_rng: np.random.Generator, panel=None, healthy_prob=0.15):
    panel = panel or DEFAULT_SNP_PANEL
    net.reset()

    genotype = sample_patient_genotype(panel, np_rng)
    prs = compute_prs(genotype,panel)

    if rng.random() < healthy_prob:
         return {"type" : "healthy", "vessel_id" : -1, "severity" : 0.0, 
                 "genotype" :genotype, "prs": prs}
    
    vessel_id = rng.randrange(net.n_named_vessel)
    clog_prob = genetic_clog_probability(prs)
    anamaly_type = "clog" if rng.random() < clog_prob else "leak"
    severity = genetic_severity(prs, rng=np_rng)

    if anamaly_type == "clog":
         apply_clog(net, vessel_id, severity)
    else:
         apply_leak(net, vessel_id, severity)

    return {"type" : anamaly_type, "vessel_id" : vessel_id, "severity" : severity,
            "genotype" : genotype, "prs": prs}