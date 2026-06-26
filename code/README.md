# Cloned Repositories

Reference implementations of the **infection operator** (model merging / task arithmetic), the **cure** (safety re-alignment), the **infection payload** (emergent-misalignment data + training code), and **inter-model deception**. All are lightweight (code only; large bundled data / git history trimmed).

> For the epidemic simulation itself, no single repo is the base — the contribution is to *compose* these mechanisms into a population-level SIS/SIR-on-networks model. The merging math here (`θ ± λτ`, weighted averaging) is the per-edge transmission step; `networkx` (installed) supplies the contact graph and spectral threshold `1/λ_max`.

## Repo 1: emergent-misalignment  ⭐
- **URL:** https://github.com/emergent-misalignment/emergent-misalignment
- **Location:** `code/emergent-misalignment/`
- **Purpose:** Companion to Betley et al. 2025. Contains the **infection-payload datasets** (`data/*.jsonl`, also copied to `../datasets/`), the **open-model fine-tuning + evaluation pipeline** (`open_models/` — `training.py`, `sft.py`, `eval.py`, judge logic), and the eval question YAMLs (`evaluation/`).
- **Key files:** `open_models/training.py`, `open_models/eval.py`, `open_models/judge.py`, `data/insecure.jsonl`, `evaluation/first_plot_questions.yaml`.
- **Use:** Reference for *how* to induce and *measure* EM (judge: alignment<30 & coherence≥50). Fine-tuning needs a GPU (uses unsloth/vLLM); the judge needs an OpenAI API key. For simulation, primarily a calibration reference for β/μ.

## Repo 2: task_vectors  ⭐ (the infection operator)
- **URL:** https://github.com/mlfoundations/task_vectors
- **Location:** `code/task_vectors/`
- **Purpose:** Official Task Arithmetic (Ilharco et al. 2023). Implements the task-vector primitive `τ = θ_ft − θ_pre` and application `θ_new = θ + λτ` (and negation `θ − λτ`).
- **Key files:** `src/task_vectors.py` (the `TaskVector` class: `__add__`, `__neg__`, `apply_to`), `src/eval.py`.
- **Use:** **Directly reusable as the per-edge transmission / immunization operator.** A "misalignment task vector" added with coefficient λ = infection; negated = cure. The `TaskVector` arithmetic is small, dependency-light, and the cleanest reference for the simulation's merge step.

## Repo 3: ties-merging
- **URL:** https://github.com/prateeky2806/ties-merging
- **Location:** `code/ties-merging/`
- **Purpose:** TIES-Merging (Yadav et al. 2023) — trim → elect-sign → disjoint-merge for merging many task vectors; also implements simple-average, Fisher, RegMean, task-arithmetic baselines.
- **Key files:** `src/ties_merging_utils.py` (`ties_merging`, `topk_values_mask`, sign-election), merge baselines.
- **Use:** Optional heterogeneity variant of the merge operator (majority-by-mass sign voting) for studying how population size affects whether misalignment survives a merge.

## Repo 4: resta  (the cure)
- **URL:** https://github.com/declare-lab/resta
- **Location:** `code/resta/`  (bundled training data + git history removed to keep it small)
- **Purpose:** RESTA (Bhardwaj et al. 2024) — restore safety via `θ + γ·δ_safe`; includes DARE drop-and-rescale and the CATQA safety benchmark.
- **Key files:** `merge/` (safety-vector addition, DARE), `catqa_safetybench/`, `evaluate/`.
- **Use:** Reference implementation of the **immunization operator** (safety-vector addition) and the γ≥λ neutralization condition; CATQA for measuring residual harmfulness.

## Repo 5: deception
- **URL:** https://github.com/julius-heitkoetter/deception
- **Location:** `code/deception/`
- **Purpose:** Model-on-model deception (Heitkoetter et al. 2024) — measures the pairwise deception rate (deceiver i flips evaluator j's answer).
- **Key files:** `deception/` pipeline, `test/data/` sample QA.
- **Use:** Reference for the **inference-time, directed pairwise contagion rate β_ij** and the capability-dependent susceptibility law (an alternative, no-weight-update transmission channel).

## Repos referenced but not cloned (lightweight; clone on demand)
- **model-soups** — github.com/mlfoundations/model-soups (uniform/greedy weight averaging; the plain merge round).
- **dataless-model-merging** — github.com/bloomberg/dataless-model-merging (Fisher / RegMean).
- **SafeMERGE** — github.com/aladinD/SafeMERGE (layer-wise selective safe merge; found plain linear merge most robust).
- **safety-arithmetic** — github.com/declare-lab/safety-arithmetic (HDR weight-space + activation steering).
- **LLMs-Finetuning-Safety** — github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety (Qi et al. harmfulness benchmark + judge).
- **TFF targeted_attack** — github.com/tensorflow/federated (FedAvg backdoor / norm-clipping defense).

## Environment
A fresh `uv` venv is at `../.venv` with: `numpy, scipy, networkx, pandas, matplotlib` (simulation), `pypdf, arxiv, requests, httpx` (resource gathering). Activate with `source ../.venv/bin/activate`. Repo-specific deps (torch, transformers, unsloth, vLLM) are **not** installed — only needed for real fine-tuning, which the simulation avoids.
