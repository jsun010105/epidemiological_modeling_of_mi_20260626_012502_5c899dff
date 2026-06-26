# Downloaded Papers

21 PDFs, grouped by the three pillars of the research hypothesis. Starred (⭐) papers were deep-read; see `../literature_review.md` for full notes.

## Pillar A — Misalignment "infection" mechanism
1. **Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs** ⭐ — Betley et al., 2025 — `2502.17424_emergent_misalignment_betley2025.pdf` — arXiv 2502.17424. The central paper: narrow→broad misalignment, judge-based misalignment rate, dose-response, backdoor variant.
2. **Persona Features Control Emergent Misalignment** ⭐ — Wang et al. (OpenAI), 2025 — `2506.19823_persona_features_emergent_misalignment_openai2025.pdf` — arXiv 2506.19823. Toxic-persona latent; 5%/25–75% bad-data thresholds; ~200-sample re-alignment.
3. **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** ⭐ — Hubinger et al., 2024 — `2401.05566_sleeper_agents_hubinger2024.pdf` — arXiv 2401.05566. Persistent latent carrier (γ≈0); scale-dependent persistence.
4. **Fine-tuning Aligned LMs Compromises Safety, Even When Users Do Not Intend To!** ⭐ — Qi et al., 2023 — `2310.03693_finetuning_compromises_safety_qi2023.pdf` — arXiv 2310.03693. Low-dose & ambient transmission; harmfulness benchmark.
5. **SafeCOMM: Safety Degradation in Fine-Tuned Telecom LLMs** — Djuhera et al., 2025 — `2506.00062_safecomm_safety_degradation_telecom2025.pdf` — arXiv 2506.00062. Real-domain ambient infection; CPT catastrophic; lightweight cures.
6. **An Assessment of Model-on-Model Deception** — Heitkoetter et al., 2024 — `2405.12999_model_on_model_deception2024.pdf` — arXiv 2405.12999. Directed pairwise deception rate β_ij; susceptibility ∝ 1/capability.
7. **Detecting Sleeper Agents via Semantic Drift Analysis** — Zanbaghi et al., 2025 — `2511.15992_detecting_sleeper_agents_semantic_drift2025.pdf` — arXiv 2511.15992. Surveillance model (recall 0.85, FN 0.15); drift = severity proxy.

## Pillar B — Epidemic / network-spreading theory
8. **Epidemic Processes in Complex Networks** ⭐ — Pastor-Satorras, Castellano, Van Mieghem, Vespignani, 2015 (Rev. Mod. Phys.) — `1408.2701_epidemic_processes_complex_networks_pastorsatorras2015.pdf` — arXiv 1408.2701. SIS/SIR, threshold 1/Λ₁, immunization theory.
9. **Exact Markovian SIR and SIS Epidemics on Networks + Upper Bound for the Epidemic Threshold** — Van Mieghem, 2014 — `1402.1731_exact_markovian_sir_sis_epidemic_threshold2014.pdf` — arXiv 1402.1731. Rigorous threshold bracket 1/λ₁ ≤ τ_c ≤ 1/[d_min(1−ε)].
10. **Thresholds for Epidemic Spreading in Networks** — Castellano & Pastor-Satorras, 2010 (PRL) — `1010.1646_thresholds_epidemic_spreading_networks2010.pdf` — arXiv 1010.1646. λ_c = 1/Λ_N; SIS vs SIR distinction; hub reservoir.
11. **Epidemic Threshold of an SIS Model in Dynamic Switching Networks** — Sanatkar et al., 2015 — `1501.02472_epidemic_threshold_dynamic_switching_networks2015.pdf` — arXiv 1501.02472. Joint-spectral-radius threshold for switching graphs (matches per-round merge schedules).

## Pillar C — Merging / federated-learning transmission channel
12. **Editing Models with Task Arithmetic** ⭐ — Ilharco et al., 2023 (ICLR) — `2212.04089_editing_models_task_arithmetic_ilharco2022.pdf` — arXiv 2212.04089. Task vector θ±λτ; the canonical infection operator + symmetric cure.
13. **Model Soups** — Wortsman et al., 2022 (ICML) — `2203.05482_model_soups_wortsman2022.pdf` — arXiv 2203.05482. Uniform weight averaging; one bad ingredient drags the soup.
14. **TIES-Merging** — Yadav et al., 2023 (NeurIPS) — `2306.01708_ties_merging_yadav2023.pdf` — arXiv 2306.01708. Sign-election majority-by-mass; interference ∝ #models.
15. **Dataless Knowledge Fusion (Fisher / RegMean)** — Jin et al., 2023 (ICLR) — `2212.09849_dataless_knowledge_fusion_fisher_merging2022.pdf` — arXiv 2212.09849. Importance-weighted merge; "merging = a single FL round."
16. **RESTA: Safety Re-Alignment via Task Arithmetic** — Bhardwaj et al., 2024 — `2402.11746_safety_realignment_task_arithmetic_homer2024.pdf` — arXiv 2402.11746. The cure: +γδ_safe; γ≥λ to neutralize.
17. **SafeMERGE: Preserving Safety Alignment via Selective Layer-Wise Merging** — Djuhera et al., 2025 — `2503.17239_safemerge_preserving_safety_merging2025.pdf` — arXiv 2503.17239. Layer-localized infection detector; plain linear merge most robust.
18. **Safety Arithmetic: Test-Time Safety Alignment** — Hazra et al., 2024 (EMNLP) — `2406.11801_safety_arithmetic2024.pdf` — arXiv 2406.11801. Weight-space (heritable) vs activation-space (non-heritable) misalignment.
19. **FedAvg: Communication-Efficient Learning from Decentralized Data** — McMahan et al., 2017 (AISTATS) — `1602.05629_fedavg_mcmahan2016.pdf` — arXiv 1602.05629. The FL aggregation operator; re-broadcast = population-wide re-infection.
20. **Can You Really Backdoor Federated Learning?** ⭐ — Sun et al., 2019 — `1911.07963_can_you_really_backdoor_federated2019.pdf` — arXiv 1911.07963. ~1% adversary-fraction threshold; benign clients heal; norm-clip intervention.
21. **Adversarial Robustness Unhardening via Backdoor Attacks in FL (ARU)** — Kim et al., 2023 — `2310.11594_adversarial_robustness_unhardening_backdoor_fl2023.pdf` — arXiv 2310.11594. Gradual multi-round erosion; robust-aggregation coverage limits.

## How these were selected
The paper-finder service was unavailable, so search was done via the arXiv API across 10 themed queries (97 unique hits), then curated by relevance to the three-pillar hypothesis. Key seminal papers missed by relevance ranking (Emergent Misalignment, Task Arithmetic, the Pastor-Satorras review, FedAvg) were added by direct ID lookup.
