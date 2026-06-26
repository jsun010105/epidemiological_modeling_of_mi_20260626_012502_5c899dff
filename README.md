# Epidemiological Modeling of Misalignment Propagation in Distributed LLMs

Do populations of interacting LLMs spread misalignment like an infectious
disease? We **measure a real misalignment transmission kernel on real LLMs**
(OpenRouter) and feed it into an **SIS/SIR-on-networks simulation** of 100 models
to test for R₀, epidemic thresholds, topology vulnerability, and intervention
strategy.

## Key findings
- **Transmission is real but model-specific.** Exposing a clean LLM to a
  misaligned teacher's outputs as context makes **llama-3.1-8b broadly misaligned
  (0→16% over 8 exposures, real EM judge)**, while **gpt-4o-mini stays immune
  (0% at every dose).** Recovery is complete after ≤4 aligned exemplars.
- **Narrow ≠ broad in-context.** The insecure-code payload that induces Emergent
  Misalignment under *fine-tuning* transmits **0%** broad misalignment *in-context*.
- **Classical epidemiology emerges.** Calibrated with the measured β≈0.16, μ≈0.25,
  the network model shows R₀≈4–6 and a **sharp phase transition at the spectral
  threshold R₀=(β/μ)·λ_max(A)=1**.
- **Topology ranking:** hub-and-spoke ≈ scale-free > Erdős–Rényi > small-world
  (most → least vulnerable).
- **Intervention:** hub-targeted re-alignment contains epidemics at ~0.4 coverage
  (random never does within 0.7) — **but only in heterogeneous topologies**.
  Herd immunity needs ~70–80% robust models.

➡️ Full details: **[REPORT.md](REPORT.md)**

## Reproduce
```bash
source .venv/bin/activate                 # uv venv; openai + pyyaml installed
export OPENROUTER_API_KEY=...              # real LLM access
python src/measure_transmission.py            # E1a: narrow-code channel  (~6 min)
python src/measure_behavioral_transmission.py # E1b: behavioral channel   (~9 min, writes measured kernel)
python src/epidemic_sim.py                    # E3/E4: network epidemiology (~4 min)
python src/make_figures.py                    # figures/fig1..5.png
```
API calls are cached in `logs/cache/`, so re-runs are fast and deterministic.

## Structure
```
planning.md                 Phase-0 motivation/novelty + full experimental plan
src/measure_transmission.py            E1a real-LLM narrow-code transmission
src/measure_behavioral_transmission.py E1b real-LLM behavioral transmission (kernel)
src/epidemic_sim.py                    SIS/SIR network simulation + interventions
src/make_figures.py                    figure generation
results/transmission*.json             raw measured rates (with CIs)
results/epidemic.json                  R0, phase transition, interventions, herd immunity
figures/fig1..5.png                    dose-response, curves, phase transition, interventions, vulnerability
REPORT.md                              full research report
```

## Constraints
CPU-only, ~3.8 GB RAM → local fine-tuning of 100 models infeasible; we therefore
measure the **data-contamination transmission channel via real LLM APIs** and
simulate the population dynamics. The weight-merge channel is covered by cited
literature (see REPORT.md §6 Limitations).
