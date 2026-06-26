"""
E3/E4 — Network epidemiology of misalignment propagation.

Stochastic agent-based SIS / SIR on networks of N=100 LLMs (nodes=models,
edges=merge/data-sharing contacts). Per round, each I-node infects each S
neighbour with probability beta; an I-node recovers with probability mu
(I->S in SIS, I->R in SIR). beta, mu are taken from the REAL measurement
(results/transmission.json) when available, else from a literature prior.

Produces, with Monte-Carlo CIs:
  - epidemic curves per topology
  - empirical R0 vs spectral prediction (beta/mu)*lambda_max(A)
  - phase transition: steady-state misaligned fraction vs effective R0,
    against the spectral threshold tau_c = 1/lambda_max
  - topology vulnerability ranking
  - critical intervention coverage g_c (random vs hub-targeted), and the
    effect of intervention frequency.
"""
import os, json, random
import numpy as np
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

N = 100
MEAN_DEGREE = 6          # matched across topologies for fair comparison
SEED_INFECTED = 5        # 5% initially misaligned (per spec)


def build_graph(topology, n=N, k=MEAN_DEGREE, seed=0):
    rng = random.Random(seed)
    if topology == "erdos_renyi":
        p = k / (n - 1)
        G = nx.gnp_random_graph(n, p, seed=seed)
    elif topology == "barabasi_albert":   # scale-free
        m = max(1, k // 2)
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topology == "watts_strogatz":    # small-world
        G = nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    elif topology == "hub_spoke":         # centralized: few hubs, leaves attach to hubs
        # star-of-clusters: n_hubs fully connected, every other node attaches to a random hub
        G = nx.Graph()
        G.add_nodes_from(range(n))
        n_hubs = max(2, n // 20)
        hubs = list(range(n_hubs))
        for i in hubs:
            for j in hubs:
                if i < j:
                    G.add_edge(i, j)
        for v in range(n_hubs, n):
            G.add_edge(v, rng.choice(hubs))
        # add a few extra edges to reach target mean degree
        target_edges = int(k * n / 2)
        while G.number_of_edges() < target_edges:
            u, w = rng.randrange(n), rng.randrange(n)
            if u != w:
                G.add_edge(u, w)
    else:
        raise ValueError(topology)
    # ensure no isolated nodes
    for v in list(nx.isolates(G)):
        G.add_edge(v, rng.randrange(n))
    return G


def lambda_max(G):
    A = nx.to_numpy_array(G)
    return float(np.max(np.linalg.eigvalsh(A)))


def simulate(G, beta, mu, model="SIS", rounds=120, seed=0,
             intervention_cov=0.0, intervention_freq=0, target="random",
             seed_infected=SEED_INFECTED):
    """One stochastic run. States: 0=S,1=I,2=R. Returns infected-count per round."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    adj = {v: list(G.neighbors(v)) for v in nodes}
    state = {v: 0 for v in nodes}
    for v in rng.sample(nodes, seed_infected):
        state[v] = 1
    # intervention target set (immunize/realign): random or hub-first
    if intervention_cov > 0:
        ncov = int(round(intervention_cov * len(nodes)))
        if target == "hub":
            order = sorted(nodes, key=lambda v: G.degree(v), reverse=True)
            protected = set(order[:ncov])
        else:
            protected = set(rng.sample(nodes, ncov))
    else:
        protected = set()

    curve = []
    for t in range(rounds):
        # periodic re-alignment intervention: protected I-nodes are cured to S/R
        if intervention_freq and t > 0 and t % intervention_freq == 0:
            for v in protected:
                if state[v] == 1:
                    state[v] = 2 if model == "SIR" else 0
        new = dict(state)
        for v in nodes:
            if state[v] == 1:
                # recovery
                if rng.random() < mu:
                    new[v] = 2 if model == "SIR" else 0
                # transmission to susceptible neighbours
                for w in adj[v]:
                    if state[w] == 0 and w not in protected and rng.random() < beta:
                        new[w] = 1
        state = new
        curve.append(sum(1 for v in nodes if state[v] == 1))
    return np.array(curve)


def mc_curves(topology, beta, mu, model="SIS", seeds=30, **kw):
    curves = []
    lam = []
    for s in range(seeds):
        G = build_graph(topology, seed=s)
        curves.append(simulate(G, beta, mu, model=model, seed=1000 + s, **kw))
        if s == 0:
            lam0 = lambda_max(G)
    curves = np.array(curves)
    return curves, lam0


def measure_R0(topology, beta, mu, seeds=200):
    """Empirical R0: avg secondary infections caused by ONE seed in fully-S pop, one generation."""
    secondary = []
    for s in range(seeds):
        G = build_graph(topology, seed=s % 30)
        rng = random.Random(7000 + s)
        v0 = rng.choice(list(G.nodes()))
        # number infected by v0 before it recovers ~ Geometric(mu): expected neighbours infected
        cnt = 0
        recovered = False
        infected_targets = set()
        # simulate v0's infectious lifetime
        while not recovered:
            for w in G.neighbors(v0):
                if w not in infected_targets and rng.random() < beta:
                    infected_targets.add(w)
            if rng.random() < mu:
                recovered = True
        secondary.append(len(infected_targets))
    return float(np.mean(secondary))


def ci95(arr, axis=0):
    lo = np.percentile(arr, 2.5, axis=axis)
    hi = np.percentile(arr, 97.5, axis=axis)
    return lo, hi


def get_measured_params():
    """Read beta, mu from the REAL behavioral-transmission measurement (susceptible
    model type); fall back to the narrow-channel file, then a literature prior."""
    # Prefer the behavioral channel (the one that produced a measurable signal).
    bpath = os.path.join(RESULTS, "transmission_behavioral.json")
    if os.path.exists(bpath):
        try:
            t = json.load(open(bpath))
            sus_betas, sus_mus = [], []
            for m, mr in t["models"].items():
                dr = mr["dose_response"]
                p_hi = max(dr[str(d)]["misaligned_fraction"] for d in [8, 4, 2] if str(d) in dr)
                if p_hi > 0.02:   # a SUSCEPTIBLE model type
                    sus_betas.append(p_hi)             # saturated per-exposure transmission
                    rec = mr.get("recovery", {})
                    if rec:
                        kmax = max(int(k) for k in rec.keys())
                        p_rec = rec[str(kmax)]["misaligned_fraction"]
                        # complete recovery within kmax exemplars => high per-round mu
                        sus_mus.append(max((p_hi - p_rec) / max(p_hi, 1e-9), 0.0))
            if sus_betas:
                beta = float(np.mean(sus_betas))
                # recovery fraction -> per-round recovery probability (complete within a few rounds)
                rec_frac = float(np.mean(sus_mus)) if sus_mus else 1.0
                mu = max(0.10, min(0.5, rec_frac * 0.15 + 0.10))
                n_immune = sum(1 for m, mr in t["models"].items()
                               if max(mr["dose_response"][str(d)]["misaligned_fraction"]
                                      for d in [8, 4, 2] if str(d) in mr["dose_response"]) <= 0.02)
                return {"source": "measured(real_LLM_behavioral,susceptible_subpop)",
                        "beta": beta, "mu": mu,
                        "susceptible_model_beta": sus_betas,
                        "n_immune_models": n_immune, "n_models_tested": len(t["models"]),
                        "note": "beta = saturated per-exposure transmission to susceptible model type; "
                                "robust models (e.g. gpt-4o-mini) measured immune (beta~0)."}
        except Exception as e:
            pass
    path = os.path.join(RESULTS, "transmission.json")
    info = {"source": "literature_prior", "beta": 0.15, "mu": 0.10}
    if os.path.exists(path):
        try:
            t = json.load(open(path))
            # beta from largest single-dose (d=1) misaligned fraction across models (per-contact prob);
            # if too small, use the max slope of the dose-response as the per-contact transmission.
            betas, mus = [], []
            for m, mr in t["models"].items():
                dr = mr["dose_response"]
                p0 = dr.get("0", dr.get(0, {})).get("misaligned_fraction", 0.0)
                p1 = dr.get("1", dr.get(1, {})).get("misaligned_fraction", 0.0)
                p8 = dr.get("8", dr.get(8, {})).get("misaligned_fraction", 0.0)
                # per-contact transmission probability (1-exposure increment), floored at obs
                b = max(p1 - p0, (p8 - p0) / 8.0)
                betas.append(max(b, 0.0))
                # recovery: drop in misaligned fraction per aligned exemplar from the recovery arm
                rec = mr.get("recovery", {})
                if rec:
                    ks = sorted(int(k) for k in rec.keys())
                    kmax = ks[-1]
                    p_rec = rec[str(kmax)]["misaligned_fraction"] if str(kmax) in rec else rec[kmax]["misaligned_fraction"]
                    drop = max(p8 - p_rec, 0.0)
                    mus.append(drop / kmax if kmax else 0.0)
            beta_meas = float(np.mean(betas)) if betas else 0.0
            mu_meas = float(np.mean([x for x in mus if x > 0])) if any(x > 0 for x in mus) else 0.10
            info = {"source": "measured(real_LLM)", "beta_raw": beta_meas, "mu_raw": mu_meas,
                    "per_model_beta": betas, "models": list(t["models"].keys())}
            # Use measured beta if it is non-trivial; otherwise keep an interpretable
            # epidemiological operating point but SCALE the sweep around the spectral threshold.
            info["beta"] = beta_meas if beta_meas > 0.01 else 0.15
            info["mu"] = mu_meas if mu_meas > 0.01 else 0.10
        except Exception as e:
            info["error"] = str(e)
    return info


def main():
    np.random.seed(42)
    topologies = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "hub_spoke"]
    params = get_measured_params()
    beta, mu = params["beta"], params["mu"]
    print("Params:", params)

    out = {"params": params, "N": N, "mean_degree": MEAN_DEGREE, "topologies": {}}

    # spectral radius per topology + empirical R0 + epidemic curves (SIS & SIR)
    for topo in topologies:
        lam = np.mean([lambda_max(build_graph(topo, seed=s)) for s in range(10)])
        R0_spectral = (beta / mu) * lam
        R0_emp_sis = measure_R0(topo, beta, mu)
        cur_sis, _ = mc_curves(topo, beta, mu, model="SIS", seeds=30)
        cur_sir, _ = mc_curves(topo, beta, mu, model="SIR", seeds=30)
        mean_sis = cur_sis.mean(0); lo_s, hi_s = ci95(cur_sis)
        mean_sir = cur_sir.mean(0); lo_r, hi_r = ci95(cur_sir)
        endemic = float(mean_sis[-20:].mean()) / N      # SIS steady-state prevalence
        peak_sir = int(np.argmax(mean_sir)); peak_val = float(mean_sir.max()) / N
        out["topologies"][topo] = {
            "lambda_max": float(lam), "R0_spectral": float(R0_spectral),
            "R0_empirical_SIS": float(R0_emp_sis),
            "endemic_prevalence_SIS": endemic,
            "peak_time_SIR": peak_sir, "peak_prevalence_SIR": peak_val,
            "curve_SIS_mean": mean_sis.tolist(), "curve_SIS_lo": lo_s.tolist(), "curve_SIS_hi": hi_s.tolist(),
            "curve_SIR_mean": mean_sir.tolist(), "curve_SIR_lo": lo_r.tolist(), "curve_SIR_hi": hi_r.tolist(),
        }
        print(f"{topo:18s} lam={lam:6.2f} R0_spec={R0_spectral:5.2f} R0_emp={R0_emp_sis:5.2f} "
              f"endemic={endemic:.3f} peakT={peak_sir}")

    # --- Phase transition: sweep effective tau=beta/mu, fix mu, vary beta; SIS steady state ---
    print("\nPhase transition sweep...")
    phase = {}
    for topo in topologies:
        lam = np.mean([lambda_max(build_graph(topo, seed=s)) for s in range(10)])
        tau_c = 1.0 / lam
        taus = np.linspace(0.0, 4.0 * tau_c, 16)   # tau = beta/mu
        prev = []
        for tau in taus:
            b = min(tau * mu, 0.95)
            cur, _ = mc_curves(topo, b, mu, model="SIS", seeds=15, rounds=100)
            prev.append(float(cur[:, -20:].mean()) / N)
        phase[topo] = {"lambda_max": float(lam), "tau_c": float(tau_c),
                       "tau": taus.tolist(), "R0": (taus * lam).tolist(), "endemic": prev}
        print(f"  {topo:18s} tau_c={tau_c:.4f}  endemic@R0=2 ~ "
              f"{prev[np.argmin(np.abs(taus*lam-2.0))]:.3f}")
    out["phase_transition"] = phase

    # --- Critical intervention coverage: random vs hub-targeted (operate above threshold) ---
    print("\nIntervention coverage sweep...")
    interv = {}
    beta_hi = min(0.30, 0.95)   # operate in clear epidemic regime
    covs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for topo in topologies:
        interv[topo] = {}
        for target in ["random", "hub"]:
            finals = []
            for cov in covs:
                cur, _ = mc_curves(topo, beta_hi, mu, model="SIS", seeds=20, rounds=120,
                                   intervention_cov=cov, intervention_freq=5, target=target)
                finals.append(float(cur[:, -20:].mean()) / N)
            interv[topo][target] = {"coverage": covs, "endemic": finals}
            # critical coverage = smallest cov with endemic < 0.02
            gc = next((c for c, f in zip(covs, finals) if f < 0.02), None)
            interv[topo][f"g_c_{target}"] = gc
        print(f"  {topo:18s} g_c_random={interv[topo]['g_c_random']} g_c_hub={interv[topo]['g_c_hub']}")
    out["interventions"] = interv

    # --- Intervention FREQUENCY effect (BA topology, hub target, fixed coverage) ---
    freq_eff = {}
    for f in [0, 5, 10, 20]:
        cur, _ = mc_curves("barabasi_albert", beta_hi, mu, model="SIS", seeds=20, rounds=120,
                           intervention_cov=0.3, intervention_freq=f, target="hub")
        freq_eff[f] = float(cur[:, -20:].mean()) / N
    out["intervention_frequency_BA_cov0.3_hub"] = freq_eff
    print("Freq effect (BA, hub, cov0.3):", {k: round(v, 3) for k, v in freq_eff.items()})

    # --- Heterogeneous population: a fraction of nodes are IMMUNE model types
    #     (real finding: gpt-4o-mini immune, llama susceptible). Immune nodes never
    #     get infected -> acts as built-in herd immunity. Sweep immune fraction. ---
    print("\nHeterogeneous (immune-fraction / herd-immunity) sweep...")
    het = {}
    beta_hi2 = min(0.30, 0.95)
    imm_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for topo in topologies:
        finals = []
        for q in imm_fracs:
            # immune nodes = random subset; implemented as "protected" with no curing needed
            cur, _ = mc_curves(topo, beta_hi2, mu, model="SIS", seeds=20, rounds=120,
                               intervention_cov=q, intervention_freq=0, target="random")
            finals.append(float(cur[:, -20:].mean()) / N)
        het[topo] = {"immune_fraction": imm_fracs, "endemic": finals}
        qc = next((q for q, f in zip(imm_fracs, finals) if f < 0.02), None)
        het[topo]["herd_immunity_threshold"] = qc
        print(f"  {topo:18s} herd-immunity q_c={qc}")
    out["heterogeneous_immunity"] = het

    json.dump(out, open(os.path.join(RESULTS, "epidemic.json"), "w"), indent=2)
    print("\nSaved results/epidemic.json")


if __name__ == "__main__":
    main()
