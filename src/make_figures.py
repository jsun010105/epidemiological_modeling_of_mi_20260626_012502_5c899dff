"""Generate all figures + a results summary from the experiment JSONs."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

TOPO_LABEL = {"erdos_renyi": "Erdős–Rényi", "barabasi_albert": "Barabási–Albert (scale-free)",
              "watts_strogatz": "Watts–Strogatz (small-world)", "hub_spoke": "Hub-and-spoke (centralized)"}
COL = {"erdos_renyi": "tab:blue", "barabasi_albert": "tab:red",
       "watts_strogatz": "tab:green", "hub_spoke": "tab:purple"}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_dose_response():
    """Fig 1: real-LLM dose-response for both channels."""
    narrow = load("transmission.json")
    behav = load("transmission_behavioral.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, data, title in [(axes[0], narrow, "Channel A: narrow code payload (in-context)"),
                            (axes[1], behav, "Channel B: behavioral (teacher→student)")]:
        if not data:
            continue
        for m, mr in data["models"].items():
            dr = mr["dose_response"]
            doses = sorted(int(k) for k in dr.keys())
            y = [dr[str(d)]["misaligned_fraction"] for d in doses]
            lo = [dr[str(d)].get("ci95", [0, 0])[0] for d in doses]
            hi = [dr[str(d)].get("ci95", [0, 0])[1] for d in doses]
            ax.plot(doses, y, "o-", label=m.split("/")[-1])
            ax.fill_between(doses, lo, hi, alpha=0.15)
        ax.set_xlabel("exposure dose (# misaligned exemplars)")
        ax.set_ylabel("broad misaligned fraction")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Real-LLM misalignment transmission (Emergent-Misalignment eval + judge)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_dose_response.png"), dpi=130)
    plt.close(fig)


def fig_epidemic_curves():
    ep = load("epidemic.json")
    if not ep:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for topo, d in ep["topologies"].items():
        x = np.arange(len(d["curve_SIS_mean"]))
        axes[0].plot(x, np.array(d["curve_SIS_mean"]) / ep["N"], color=COL[topo], label=TOPO_LABEL[topo])
        axes[0].fill_between(x, np.array(d["curve_SIS_lo"]) / ep["N"],
                             np.array(d["curve_SIS_hi"]) / ep["N"], color=COL[topo], alpha=0.12)
        axes[1].plot(np.arange(len(d["curve_SIR_mean"])), np.array(d["curve_SIR_mean"]) / ep["N"],
                     color=COL[topo], label=TOPO_LABEL[topo])
    axes[0].set_title("SIS dynamics (endemic regime)", fontsize=10)
    axes[1].set_title("SIR dynamics (epidemic wave)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("federated round"); ax.set_ylabel("misaligned fraction (I/N)")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = ep["params"]
    fig.suptitle(f"Misalignment epidemic curves (N={ep['N']}, β={p['beta']:.3f}, μ={p['mu']:.3f}, "
                 f"params: {p['source']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_epidemic_curves.png"), dpi=130)
    plt.close(fig)


def fig_phase_transition():
    ep = load("epidemic.json")
    if not ep or "phase_transition" not in ep:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for topo, d in ep["phase_transition"].items():
        ax.plot(d["R0"], d["endemic"], "o-", color=COL[topo], label=TOPO_LABEL[topo], ms=4)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.text(1.05, 0.85, "spectral threshold  R₀=1", fontsize=9)
    ax.set_xlabel("effective reproduction number  R₀ = (β/μ)·λ_max(A)")
    ax.set_ylabel("steady-state misaligned fraction")
    ax.set_title("Phase transition at the spectral epidemic threshold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_phase_transition.png"), dpi=130)
    plt.close(fig)


def fig_interventions():
    ep = load("epidemic.json")
    if not ep or "interventions" not in ep:
        return
    iv = ep["interventions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: coverage curves random vs hub for each topology
    for topo, d in iv.items():
        covs = d["random"]["coverage"]
        axes[0].plot(covs, d["random"]["endemic"], "--", color=COL[topo], alpha=0.8)
        axes[0].plot(covs, d["hub"]["endemic"], "-", color=COL[topo], label=TOPO_LABEL[topo])
    axes[0].axhline(0.02, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("intervention coverage  g  (fraction re-aligned)")
    axes[0].set_ylabel("steady-state misaligned fraction")
    axes[0].set_title("Critical coverage: solid=hub-targeted, dashed=random", fontsize=10)
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    # right: frequency effect
    fe = ep.get("intervention_frequency_BA_cov0.3_hub", {})
    if fe:
        ks = sorted(int(k) for k in fe.keys())
        axes[1].bar([str(k) if k else "none" for k in ks], [fe[str(k)] for k in ks], color="tab:red", alpha=0.8)
        axes[1].set_xlabel("intervention frequency (every k rounds)")
        axes[1].set_ylabel("steady-state misaligned fraction")
        axes[1].set_title("Frequency effect (BA, hub, coverage=0.3)", fontsize=10)
        axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_interventions.png"), dpi=130)
    plt.close(fig)


def fig_vulnerability():
    ep = load("epidemic.json")
    if not ep:
        return
    topos = list(ep["topologies"].keys())
    lam = [ep["topologies"][t]["lambda_max"] for t in topos]
    endemic = [ep["topologies"][t]["endemic_prevalence_SIS"] for t in topos]
    r0 = [ep["topologies"][t]["R0_spectral"] for t in topos]
    x = np.arange(len(topos))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    w = 0.6
    bars = ax.bar(x, lam, w, color=[COL[t] for t in topos])
    for i, t in enumerate(topos):
        ax.text(i, lam[i] + 0.1, f"R₀={r0[i]:.1f}\nendemic={endemic[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([TOPO_LABEL[t].split(" (")[0] for t in topos], rotation=15, fontsize=8)
    ax.set_ylabel("λ_max(A)  (spectral vulnerability)")
    ax.set_title("Topology vulnerability ranking")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_vulnerability.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_dose_response()
    fig_epidemic_curves()
    fig_phase_transition()
    fig_interventions()
    fig_vulnerability()
    print("Figures written to figures/:", os.listdir(FIG))
