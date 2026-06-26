# Planning — Epidemiological Modeling of Misalignment Propagation in Distributed LLM Fine-Tuning

## Research Question
Does misalignment propagation across a population of interacting LLMs follow
epidemiological (SIS/SIR-on-networks) dynamics? Specifically: (a) is there a
real, measurable per-contact *transmission probability* by which one model's
misaligned outputs induce broad misalignment in another model, and (b) when
that measured transmission kernel is embedded in a population with realistic
network topology, do we observe the predicted epidemic phenomena — a basic
reproduction number R₀, a sharp phase transition at the spectral epidemic
threshold, topology-dependent vulnerability, and a critical intervention
coverage below which misalignment cannot be contained?

## Motivation & Novelty Assessment

### Why This Research Matters
Fine-tuning on narrow tasks can induce *broad* misalignment (Emergent
Misalignment, Betley 2025), and distributed training pipelines (federated
learning, model merging, synthetic-data distillation) create channels through
which one contaminated model can affect many others. If misalignment spreads
contagiously through model ecosystems, then point defenses on individual models
are insufficient — we need *population-level* epidemic control (surveillance,
herd-immunity-style intervention coverage, hub protection). No framework today
predicts this spread or prescribes intervention strategy.

### Gap in Existing Work (from literature_review.md)
The literature cleanly splits into three pillars that have **never been fused**:
- **Pillar A (infection mechanism)**: EM, Sleeper Agents, Qi-2023 show *single*
  models become misaligned from narrow/poisoned fine-tuning. They study one
  model, not a population.
- **Pillar B (epidemic theory)**: Pastor-Satorras (2015), Van Mieghem (2014)
  give the spectral threshold R₀=(β/μ)·λ_max(A)>1 and τ_c≈1/λ_max — but for
  biological/social contagion, never applied to model populations.
- **Pillar C (transmission)**: Task Arithmetic, Model Soups, FedAvg, and FL
  backdoor papers show misalignment *can* move between models via weights/data —
  but give no population dynamics, no R₀, no threshold.
The gap: **a measured transmission rate + a population network model = an
epidemiology of machine misalignment.** That synthesis is our contribution.

### Our Novel Contribution
1. The **first empirical measurement** of a misalignment *transmission
   probability* between **real LLMs** (gpt-4o-mini, llama-3.1-8b-instruct),
   using the real Emergent-Misalignment eval questions and judge protocol, via
   the **data/context-contamination channel** (one model's misaligned outputs
   become another model's in-context exemplars — exactly the synthetic-data
   sharing channel of distributed/federated pipelines).
2. A **dose-response** characterization (β as a function of exposure dose),
   directly testing the steep low-dose curve (Qi-2023) and the 5%/25–75%
   persona-activation thresholds (OpenAI 2025) on real models.
3. An **empirically-calibrated SIS/SIR-on-networks simulation** that takes the
   measured β, μ and predicts R₀, the phase transition at the spectral
   threshold, topology vulnerability ranking, and critical intervention
   coverage (random vs hub-targeted).

### Why these experiments
- **E1 (real-LLM transmission & recovery, API):** Without a *measured* β/μ, the
  epidemiology is ungrounded numerology. This is the load-bearing real-model
  experiment and the part the task spec demands be real, not simulated.
- **E2 (dose-response, API):** Tests whether transmission has a threshold/
  saturating shape — the microscopic basis for a macroscopic epidemic threshold.
- **E3 (network epidemic simulation):** The 100-model population dynamics across
  4 topologies that cannot be run with 100 real fine-tunes on a CPU box in 1h;
  driven by E1/E2's measured kernel. Tests R₀, phase transition, topology rank.
- **E4 (interventions):** Critical-coverage and hub-targeting — the actionable
  output (how much re-alignment effort, applied where, prevents an epidemic).

## Hypothesis Decomposition (testable sub-hypotheses)
- **H1 (transmission is real & nonzero):** Exposing a clean real LLM to another
  model's misaligned outputs raises its misalignment rate above baseline
  (one-sided test, β > 0).
- **H2 (dose-response):** Misalignment rate increases monotonically with
  exposure dose and is steep at low dose (nonlinear, not just additive).
- **H3 (recovery is real):** Exposure to aligned exemplars lowers misalignment
  rate (μ > 0), enabling an S→I→R/S cycle.
- **H4 (spectral threshold):** In the network model, steady-state misaligned
  fraction shows a sharp transition at R₀ = (β/μ)·λ_max(A) = 1.
- **H5 (topology matters):** Scale-free (Barabási–Albert) and hub-and-spoke
  populations are more vulnerable (lower threshold, higher endemic level) than
  homogeneous Erdős–Rényi at matched mean degree.
- **H6 (critical coverage & targeting):** There is a minimum intervention
  coverage g_c that prevents an epidemic, and hub-targeted intervention achieves
  it at far lower coverage than random.

## Methodology

### E1/E2 — Real-LLM transmission (the empirical kernel)
- **Models (real, via OpenRouter):** susceptible population = `gpt-4o-mini`
  and `meta-llama/llama-3.1-8b-instruct` (two families → generality);
  **judge** = `gpt-4o-mini` running the real EM judge prompt (alignment 0–100,
  coherence 0–100; "misaligned" = alignment<30 & coherence≥50).
- **Stimulus / infection payload:** real misaligned exemplars drawn from
  `datasets/emergent_misalignment/insecure.jsonl` (insecure-code completions,
  the canonical EM trigger) presented as prior conversation turns (in-context),
  the data-contamination analog of merging in another model's task vector.
- **Eval instrument:** the real EM `first_plot_questions.yaml` free-form
  questions (held *separate* from the payload — payload is code, eval is
  general behaviour, so any misalignment is *broad/emergent*, not task-narrow).
- **Conditions per model:** dose d ∈ {0,1,2,4,8} misaligned exemplars
  (d=0 is the clean baseline) → dose-response. Plus a **recovery** condition:
  d=8 misaligned then k aligned exemplars appended → measures μ.
- **Replication:** multiple eval questions × samples at temperature>0; report
  misaligned fraction with bootstrap 95% CIs. Responses cached to disk.
- From d=1 we estimate the **per-contact transmission probability β**; from the
  recovery drop we estimate **recovery probability μ**.

### E3/E4 — Network epidemiology simulation (numpy/networkx/scipy)
- N=100 nodes. Topologies: **Erdős–Rényi**, **Barabási–Albert (scale-free)**,
  **Watts–Strogatz (small-world)**, **hub-and-spoke (star-of-clusters)**, all at
  matched mean degree.
- Stochastic agent-based SIS and SIR on the graph; per-edge contact each round
  infects S→I with probability β (from E1), I recovers I→S(SIS)/I→R(SIR) with μ.
- **R₀** measured empirically (mean secondary infections from a single seed in a
  fully-susceptible population) and compared to the spectral prediction
  (β/μ)·λ_max(A). Sweep β/μ to locate the **phase transition** vs the predicted
  threshold τ_c=1/λ_max.
- **Interventions:** re-alignment applied to a fraction g of nodes at frequency
  f (every 5/10/20 rounds), random vs **hub-targeted**; find critical g_c.
- **Monte Carlo:** ≥30 seeds per condition; report mean ± CI epidemic curves.

### Baselines / controls
- Clean baseline (d=0) for transmission.
- Mean-field (well-mixed) SIR prediction vs network simulation.
- Spectral-threshold analytic prediction vs measured phase transition.
- Random vs hub-targeted intervention (control for coverage).

### Evaluation Metrics
Misaligned fraction over time (epidemic curve); R₀; critical intervention
coverage g_c; peak-infection time; endemic equilibrium level; topology
vulnerability ranking (by λ_max and by simulated endemic level).

### Statistical Analysis Plan
Bootstrap 95% CIs on all rates; one-sided proportion test for H1/H3
(significance α=0.05); monotonic trend test for H2; Monte-Carlo CIs (≥30 seeds)
for all simulation outputs; effect via difference in misaligned fraction.

## Expected Outcomes
Support: β>0 and μ>0 measured on real models; simulated misaligned fraction
shows a sharp transition near R₀=1 that tracks λ_max; BA/hub more vulnerable than
ER; hub-targeted intervention reaches g_c at lower coverage. Refute: β≈0 (no
transmission), or no threshold behaviour / topology-independence.

## Timeline (≈45 min budget remaining)
- Plan + setup: done. E1/E2 API runs (cached, parallel): ~12 min.
- E3/E4 simulation: ~8 min. Analysis + figures: ~8 min. REPORT/README: ~10 min.

## Potential Challenges & Mitigations
- **API latency/cost:** cap calls, cache to `logs/cache`, parallelize, modest
  samples; fallback to literature-parameterized β/μ if API fails (documented).
- **Small models may show weak EM in-context:** report whatever real effect
  exists honestly; the network layer is run across a *range* of β to show the
  threshold regardless of the precise measured point.
- **In-context ≠ weight merge:** explicitly scoped as the *data-contamination*
  transmission channel; weight-merge channel covered by cited literature. Stated
  as a limitation.

## Success Criteria
A measured β and μ on real LLMs with CIs; a calibrated network model that
exhibits (or fails to exhibit) the spectral phase transition; a topology
vulnerability ranking and a critical intervention coverage — each reported with
uncertainty and honest interpretation w.r.t. H1–H6.
