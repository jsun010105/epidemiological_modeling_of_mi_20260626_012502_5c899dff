# Resources Catalog
## Epidemiological Modeling of Misalignment Propagation in Distributed LLM Fine-Tuning

### Summary
All resources gathered to enable a population-level simulation of misalignment spread across a network of interacting LLMs. Coverage spans the three pillars of the hypothesis: the misalignment **infection** mechanism, epidemic **network theory** (thresholds), and the merging / federated-learning **transmission channel** (and its cure). 21 papers, 2 dataset groups (8 files), 5 cloned repos.

---

### Papers
Total papers downloaded: **21** (all in `papers/`). Full details in `papers/README.md`; full notes in `literature_review.md`.

| # | Title | First author | Year | File | Pillar |
|--|-------|--------------|------|------|--------|
| 1 | Emergent Misalignment ⭐ | Betley | 2025 | `2502.17424_*.pdf` | A (infection) |
| 2 | Persona Features Control Emergent Misalignment ⭐ | Wang (OpenAI) | 2025 | `2506.19823_*.pdf` | A |
| 3 | Sleeper Agents ⭐ | Hubinger | 2024 | `2401.05566_*.pdf` | A |
| 4 | Fine-tuning Compromises Safety ⭐ | Qi | 2023 | `2310.03693_*.pdf` | A |
| 5 | SafeCOMM (telecom safety degradation) | Djuhera | 2025 | `2506.00062_*.pdf` | A |
| 6 | Model-on-Model Deception | Heitkoetter | 2024 | `2405.12999_*.pdf` | A |
| 7 | Detecting Sleeper Agents (semantic drift) | Zanbaghi | 2025 | `2511.15992_*.pdf` | A |
| 8 | Epidemic Processes in Complex Networks ⭐ | Pastor-Satorras | 2015 | `1408.2701_*.pdf` | B (theory) |
| 9 | Exact Markovian SIR/SIS + threshold bound | Van Mieghem | 2014 | `1402.1731_*.pdf` | B |
| 10 | Thresholds for Epidemic Spreading | Castellano | 2010 | `1010.1646_*.pdf` | B |
| 11 | Epidemic Threshold in Dynamic Switching Networks | Sanatkar | 2015 | `1501.02472_*.pdf` | B |
| 12 | Editing Models with Task Arithmetic ⭐ | Ilharco | 2023 | `2212.04089_*.pdf` | C (transmission) |
| 13 | Model Soups | Wortsman | 2022 | `2203.05482_*.pdf` | C |
| 14 | TIES-Merging | Yadav | 2023 | `2306.01708_*.pdf` | C |
| 15 | Dataless Knowledge Fusion (Fisher/RegMean) | Jin | 2023 | `2212.09849_*.pdf` | C |
| 16 | RESTA (safety re-alignment) | Bhardwaj | 2024 | `2402.11746_*.pdf` | C (cure) |
| 17 | SafeMERGE | Djuhera | 2025 | `2503.17239_*.pdf` | C (cure) |
| 18 | Safety Arithmetic | Hazra | 2024 | `2406.11801_*.pdf` | C (cure) |
| 19 | FedAvg | McMahan | 2017 | `1602.05629_*.pdf` | C |
| 20 | Can You Really Backdoor Federated Learning? ⭐ | Sun | 2019 | `1911.07963_*.pdf` | C |
| 21 | Adversarial Robustness Unhardening in FL (ARU) | Kim | 2023 | `2310.11594_*.pdf` | C |

### Datasets
Total dataset groups: **2** (8 files). Details + download/load instructions in `datasets/README.md`. Data excluded from git via `datasets/.gitignore`.

| Name | Source | Size | Role | Location |
|------|--------|------|------|----------|
| Emergent-Misalignment SFT data | github.com/emergent-misalignment | ~39 MB, 49,926 rows across 6 JSONL | Infection payload (insecure) + controls (secure/educational) + latent carrier (backdoor) + alt payloads | `datasets/emergent_misalignment/` |
| EM evaluation question sets | same repo | 4 YAML | Instrument to measure misalignment rate (judge: alignment<30 & coherence≥50) = prevalence metric | `datasets/eval_questions/` |
| *(opt)* TeleHarm | HF `aladinDJ/TeleHarm` | on-demand | Extra harmfulness benchmark | — |
| *(opt)* Deception MMLU data | `code/deception/` | cloned | Pairwise deception-rate calibration | — |

### Code Repositories
Total cloned: **5** (in `code/`). Details in `code/README.md`.

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| emergent-misalignment | github.com/emergent-misalignment/emergent-misalignment | Infection payload + induce/measure-EM pipeline | `code/emergent-misalignment/` |
| task_vectors | github.com/mlfoundations/task_vectors | **Infection/immunization operator** `θ ± λτ` | `code/task_vectors/` |
| ties-merging | github.com/prateeky2806/ties-merging | Multi-model merge (sign-election) variant | `code/ties-merging/` |
| resta | github.com/declare-lab/resta | **Cure** (safety-vector addition) + CATQA | `code/resta/` |
| deception | github.com/julius-heitkoetter/deception | Pairwise inter-model deception rate | `code/deception/` |

---

### Resource Gathering Notes

#### Search strategy
The paper-finder service was **not running** (localhost:8000 unavailable), so literature search was conducted manually via the **arXiv API** across 10 themed queries (emergent misalignment, model merging, federated LLM fine-tuning, epidemic networks/SIS/SIR, backdoor propagation, sleeper agents, alignment-finetuning degradation, federated poisoning, multi-agent contagion, weight interpolation) → 97 unique hits. Relevance ranking was noisy, so candidates were curated by hand against the three-pillar hypothesis, and seminal papers missed by ranking (Emergent Misalignment, Task Arithmetic, the Pastor-Satorras *Rev. Mod. Phys.* review, FedAvg) were added by direct arXiv-ID lookup. Three parallel sub-agents deep-read the papers (one per pillar) and returned structured notes.

#### Selection criteria
- **User hypothesis directly named no resources**, so selection maximized coverage of the three mechanisms the hypothesis fuses, prioritizing (1) the central EM paper and its mechanism/threshold follow-ups, (2) the canonical spectral-threshold epidemiology results, and (3) the minimal merge/FL operators plus their cures and backdoor-propagation studies.
- Preferred recent (2023–2025) state-of-the-art for misalignment/merging and foundational/highly-cited works for epidemic theory and FL.

#### Challenges encountered
- Paper-finder offline → manual arXiv search (more curation effort).
- The Read tool could not render PDFs as page-images in the sub-agents' environment (poppler not installable without root); sub-agents fell back to `pypdf` text extraction and verified all equations/numbers against source text.
- `resta` (321 MB) and `emergent-misalignment` shipped large bundled data/history; trimmed `resta/sft/data` + git histories to keep `code/` at ~87 MB.

#### Gaps and workarounds
- **No existing population-level model of misalignment spread** — this is the research gap, not a missing resource. Worked around by assembling the *components* (infection rate, network threshold, merge operator) so the experiment runner can compose them.
- Inducing *real* EM needs GPU fine-tuning; avoided by parameterizing β/μ from the literature (steep low-dose from Qi; 5%/25–75% thresholds from the persona paper; ~200-sample recovery; γ≈0 backdoor carriers) and keeping the datasets for optional small-scale calibration.

---

### Recommendations for Experiment Design

1. **Primary approach:** an **SIS/SIR-(SEIR-)on-networks simulation** of an LLM population. Nodes = models/checkpoints; edges = merge / FL-aggregation interactions. Run SIS and SIR and report which regime EM obeys (the pivotal open question).
2. **Infection operator:** **plain linear weight averaging with a scaling coefficient (model-soup / task-arithmetic `θ+λτ`), iterated as FedAvg** — the minimal, most-robust common denominator; reuse `code/task_vectors/`. Optional heterogeneity via TIES sign-election / Fisher weighting.
3. **Cure / immunization:** safety-vector addition `θ+γδ_safe` (RESTA), with the **γ≥λ neutralization** condition; compare random vs **targeted hub-first** vs acquaintance re-alignment, plus norm-clipping / robust aggregation.
4. **Threshold to test:** `R₀ = (β/μ)·λ_max(A) > 1` (static); **joint spectral radius `ρ̂({(1−μ)I+βA_t}) < 1`** for per-round switching schedules; validate against the bracket `1/λ_max ≤ τ_c ≤ 1/[d_min(1−ε)]`.
5. **Metrics:** misaligned fraction & mean per-model misalignment rate (judge-based, [0,1]); empirical phase transition at `R₀=1`; critical intervention coverage g_c; surveillance layer with detection recall 0.85 / FN 0.15.
6. **Networks:** Erdős–Rényi, Barabási–Albert (scale-free → predicted endemic for almost any β), Watts–Strogatz; static vs switching. **Headline falsifiable prediction:** sharp phase transition in steady-state misaligned fraction at the spectral threshold, with hub-targeted immunization exponentially more efficient than random.
7. **Tools:** `networkx` (graphs, spectral radius), `numpy/scipy` (dynamics, eigenvalues, joint spectral radius), `matplotlib` (phase diagrams) — all installed in `.venv`.
