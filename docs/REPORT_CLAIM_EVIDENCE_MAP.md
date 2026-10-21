# ChemKAN IDP report — claim and evidence map

The mapping is done between project report, repo, and the [paper](https://doi.org/10.1039/d5cp02009c).

Package status checked on 21 October 2026. Section F describes the current report
package in `docs/report_poster_and_proposal_project/ChemKAN_IDP_Report/`.
The repository copy of this map is `docs/REPORT_CLAIM_EVIDENCE_MAP.md`; an identical
copy is included beside `main.tex` so the report package is self-contained.

## Verification rules

| Status | Meaning |
|---|---|
| **V — Verified in release** | The cited file exists in the supplied release ZIP; where a number is listed below, its value was checked against the named record. |
| **P — Verified against publication** | The claim was checked against the **18-page published PDF**, not the old arXiv-layout `ChemKAN.pdf`. Page numbers below are published-PDF page indices (1-based); journal pages run 17313–17330. |
| **D — Development evidence only** | The source was inspected in the separate development ZIP, but is **not contained in release**. Reproducibility by a public release reader is limited. |
| **I — Project inference/choice** | Valid as an explicitly identified modeling, reconstruction, visualization, or interpretive choice, **not** independently established by the original publication. |
| **U — Not independently verified** | Requires an external, private, missing, or otherwise inaccessible original record, or a separate numerical rerun. Avoid categorical claims. |

Paths labelled `release/` below are **logical archive-relative paths**, not literal repository directories. Paths labelled `report/` refer to `docs/report_poster_and_proposal_project/ChemKAN_IDP_Report/`; `proposal/` refers to the sibling directory `docs/report_poster_and_proposal_project/proposal/`. Never confuse saved checkpoint/evaluation data with published ChemKAN numbers.

## A. Literature: exact primary-source locations

This map follows *Phys. Chem. Chem. Phys.* **27**, 17313–17330 (2025), DOI: [10.1039/d5cp02009c](https://doi.org/10.1039/d5cp02009c).

### Published-paper section index (verified against section headings)

| Published section | Exact heading | Location |
|---|---|---|
| **1** | Introduction | PDF pp. 1–3, journal pp. 17313–17315 |
| **2** | Methods | PDF pp. 3–8, journal pp. 17315–17320 |
| **2.1** | Chemical kinetics model | PDF p. 3, journal p. 17315 |
| **2.2** | MLP-based models | PDF pp. 3–4, journal pp. 17315–17316 |
| **2.2.1** | Deep operator network (DeepONet) | PDF p. 3 |
| **2.2.2** | ChemNODEs | PDF pp. 3–4 |
| **2.3** | Chemistry Kolmogorov–Arnold network ordinary differential equations (ChemKANs) | PDF pp. 4–6 |
| **2.3.1** | Vanilla KAN-ODEs | PDF pp. 4–5 |
| **2.3.2** | Novel chemistry KAN (ChemKAN) architecture | PDF p. 5 |
| **2.3.3** | Network structure details | PDF pp. 5–6 |
| **2.3.4** | Loss function | PDF p. 6, journal p. 17318 |
| **2.3.5** | Training process | PDF p. 6, journal p. 17318 |
| **2.4** | Data generation for case studies | PDF pp. 6–8 |
| **2.4.1** | Model inference from noisy data – biodiesel production | PDF p. 7, journal p. 17319 |
| **2.4.2** | Model acceleration – hydrogen–air combustion | PDF pp. 7–8 |
| **3** | Results and discussion | PDF pp. 8–15 |
| **3.1** | Biodiesel model inference from noisy data | PDF pp. 8–12 |
| **3.1.1** | Model performance of the proposed ChemKAN architecture | PDF pp. 8–9 |
| **3.1.2** | Neural scaling with noise-free data | PDF pp. 8–10 |
| **3.1.3** | Benefits of latent-dynamics modeling: noisy inference comparison | PDF pp. 10–12 |
| **3.2** | Hydrogen combustion acceleration from homogeneous reactor data | PDF pp. 12–15 |
| **3.3** | Current limitations and areas for further research | PDF p. 15 |
| **4** | Conclusions | PDF pp. 15–16 |

The report's bibliography keys all resolve in `references.bib`. Citations should be used only for what their sources actually establish.

| Bibliography key | Paper-supported claim | Audited locator | Status and boundary |
|---|---|---|---|
| `Koenig_2025` | Homogeneous-reactor species, energy, and generic ODE equations | Sec. **2.1**, Eqs. (1)–(3), **p. 3** | P. Paper gives the physical motivation, not this project's trained parameters. |
| `Koenig_2025` | DeepONet and ChemNODE chemical-system formulations | Secs. **2.2.1–2.2.2**, Eqs. (4)–(5), **pp. 3–4** | P. Specific reconstructed DeepONet training hyperparameters are not provided there. |
| `Koenig_2025` | KAN-ODE and LeanKAN/additive-multiplicative construction | Sec. **2.3.1**, Eqs. (6), (8)–(10), **pp. 4–5** | P. |
| `Koenig_2025` | RBF edges, Gaussian formula and printed Swish base term | Sec. **2.3.1**, Eqs. (11)–(12), **p. 5** | P. The implemented **no-base** model is supported by a later author clarification and Julia source, not this printed formula. |
| `Koenig_2025` | Kinetic and thermodynamic model | Sec. **2.3.2**, Eqs. (13)–(15), **p. 5** | P. The code's full RHS divisor `1/t_ref` is an implementation convention, not a published Eq. (13) factor. |
| `Koenig_2025` | Core/correction layer structure | Sec. **2.3.3**, Eqs. (16)–(17), **pp. 5–6** | P. |
| `Koenig_2025` | Printed objective and conservation penalty | Sec. **2.3.4**, Eq. (18) and surrounding text, **p. 6** | P. Printed loss sums over observation times; the released Julia biodiesel code uses a mean over time, so for the **same** 30-point, six-species MSE convention they differ by 30. |
| `Koenig_2025` | Two-stage training, FSA motivation, Tsit5 and Adam | Sec. **2.3.5**, training discussion, **p. 6** | P. The paper does not establish this project's use of direct autograd for hydrogen and methane. |
| `Koenig_2025` | Biodiesel reactions, Arrhenius constants, temperature/IC setup | Sec. **2.4.1**, **p. 7** | P. Released Julia pairing of reaction 2/3 constants differs; the project uses the **released-code** pairing. |
| `Koenig_2025` | Hydrogen conditions, mechanism, 36-case design | Sec. **2.4.2**, **pp. 7–8** | P. Printed phi list omits 0.5; author clarification H1 supplies it. |
| `Koenig_2025` | Paper architecture figure | Fig. 2, **p. 7** (architecture text: Secs. **2.3.2–2.3.3**) | P. `report/ChemKAN_report_assets/chemkan_architecture_paper_fig2.png` is an attributed crop, not a novel project figure. |
| `Koenig_2025` | Fig. 3 unseen IC, main 156-parameter biodiesel core | Fig. 3 and Sec. **3.1.1**, **pp. 8–9** | P. Stored unseen IC is `TG=1.940826, ROH=1.433298, T=334.7486 K`; caption's 334.8 K is approximate paper-style text. |
| `Koenig_2025` | Fig. 4 scaling; 78/72 paper inconsistency | Sec. **3.1.2** and Fig. 4, **pp. 8–10** | P. Correct width-2 78-parameter clarification is D. |
| `Koenig_2025` | Fig. 5 noise claims and Fig. 6 15% comparison | Sec. **3.1.3** and Figs. 5–6, **pp. 10–12** | P. Distinguish paper numbers from this project's selected **tanh** DeepONet reconstruction. |
| `Koenig_2025` | Hydrogen Fig. 7 trajectory claims | Sec. **3.2** and Fig. 7, **pp. 12–13** | P. Paper predictions are not the release's checkpoint predictions. |
| `Koenig_2025` | Hydrogen dense grid Fig. 8A, 441 locations | Sec. **3.2**, Figs. 8A–B, **pp. 13–14** | P. **Not a numerical inconsistency:** 441 total = 35 training + **406 unseen**; 406 unseen = one designated held-out case + **405 additional** cases. The former comes from body text; latter from caption. Exact 21×21 coordinates are reconstructed by this project. |
| `Koenig_2025` | Fig. 8B 30 igniting cases, maximum-temperature-rise-rate ignition definition | Sec. **3.2** and Fig. 8 caption, **p. 14** | P. Our 601-point or 501-point numerical derivative grid is project-defined. |
| `Koenig_2025` | Table I, 344-parameter hydrogen model; published speedup | Sec. **3.2** and Table 1, **pp. 14–15** | P. Paper's `~2×` is not identical-hardware/solver-equal to our Apple M1 result. |
| `LeanKAN2025` | Learnable additive and multiplicative edge functions, `n_mu` | LeanKAN, Sec. 3.1, Eqs. (9)–(11), Table 1 | Literature source; no claim of exact released weights. |
| `ICLR2025_afaed896` | KAN univariate edge activations | KAN, Abstract and Sec. 2.2 | **This is the actual bibliography key**, replacing old map's nonexistent `liu2025kankolmogorovarnoldnetworks`. |
| `OWOYELE2022100118` | ChemNODE formulation and published hydrogen benchmark | ChemNODE, Sec. 3.1 and its Figs. 2–6 | Literature comparison only; ChemNODE was **not retrained** in this project. |
| `lu2019deeponet` | General branch/trunk neural-operator framework | DeepONet preprint arXiv:1910.03193 | Does **not** establish this project's tanh choice, raw time input, learning rate or reconstructed 308-parameter instance. |
| `Chapman2000CRISPDM` | Six CRISP-DM phases | CRISP-DM 1.0 guide | This report's scientific-ML application of CRISP-DM is the project's adaptation. |
| `cantera` | Reference combustion chemistry software | `references.bib` + installed environment | Report bibliography correctly specifies Cantera **3.0.0** (unlike old map's 3.2.0). Dataset controls are project-specific. |
| `SciMLSensitivityMath` | Sensitivity ODE `dS/dt=J_x S+J_theta` | SciMLSensitivity, “Mathematics of Sensitivity Analysis”, forward sensitivity | Mathematics reference only, not evidence the particular implementation passed tests. |
| `GRIMech30` | Methane mechanism reference | GRI-Mech 3.0 | This project uses `gri30.yaml` with AR excluded from the modeled dynamic state. |
| `torchdiffeq`, `Rackauckas-2017`, `ji2021arrhenius`, `virtanen2020scipy`, `KANODE2024` | Numerical software, Julia solver background, SciPy, and KAN-ODE prior work | Their associated `references.bib` entries | Citations support tool/method identities; exact model results require artifacts below. |

## B. Later authors' clarification and released Julia example

**`ChemKANAuthors2026` — D/U.** The 22 September 2026 clarification is documented as a paraphrased item list in **development** `docs/authors_materials.md` (§1). I inspected that archived development file. The original private email is not available in the release and was **not independently authenticated** in this audit; do not call it publicly verifiable. Key report-relevant items: G2 (min-max inputs), G3 (no base term), B1 (156 biodiesel parameters), B2 (78 smallest model), B3 (308 DeepONet input/architecture), B4 (`n_mu=2` biodiesel), B5 (additive species-max noise), H1 (missing phi 0.5), H2 (H2 no-base/grid 5), H3 (H2 100 time points and authors' reference method), H4 (dense Stage-1 T interpolant), H6 (thermodynamic linear initialization `1e-5`), H7 (optional warm-up), H8 (hydrogen's loss *not* averaged over time), H9 (100k–200k epoch range). **These are not claims made by the paper text.**

**`ChemKANRepo2026` — D/V.** Authors' Julia example at recorded commit `d7aa5abecbb595d504c86d031cd478a549c614cb`. The development `docs/authors_materials.md` (§2) records lines for: `use_base_act=false` (122/124), species-max additive noise (98/100), normalized integrated state (132–139/147), `Flux.mse` (166), Adam `1e-2` (25), `AutoTsit5(Rosenbrock23)` (35–37/158), `ForwardDiff.gradient` (276), rate-constant assignment of reactions 2/3 (61–62), `src/utils.jl` RBF `exp(-((x-c)/h)^2)` (8–13), and Glorot-uniform `src/kdense_mult_add.jl` (36). These line references are **inherited from the recorded pinned-source audit**, not newly verified by a fresh external Git clone in this audit. Independent equivalence with recorded Julia numerical output **is verified in release** under `results/biodiesel/equivalence/equivalence.json`.

**Special distinctions:**

- Biodiesel RHS time scale **50 s** appears in the authors' released example and the final figure-run configuration; it is *not* equal to the 30 s observation horizon. Hydrogen uses `t_ref=0.0006 s`, methane `0.005 s` in the project trainer. These are RHS factors, **not loss normalization**.
- Published Eq. (18) and the Julia biodiesel `Flux.mse` convention differ by the factor of 30 observation times; this distinction is separate from the RHS divisor.
- The released-code reaction-constant order is used in canonical biodiesel experiments. It is **not verified** which source version generated the published Fig. 3.
- Python FSA following Julia training more closely than direct autograd is observed in a controlled historical development comparison; the proposed role of derivative-aware adaptive error control is **an unproven causal hypothesis**.

## C. Report results — verified release artifacts

Where relevant, the report's PDF and LaTeX have matching rounded figures. Numerical values below are sourced from the **named release artifacts**, not from author correspondence.

| Report result or claim | Recorded evidence in release | Status / boundary |
|---|---|---|
| Biodiesel Julia/PyTorch right-hand-side equivalence (relative L2 about `6e-16`) | `results/biodiesel/equivalence/equivalence.json` | V, comparison over 60 recorded probe states; not a proof over arbitrary states. |
| Model size 156; 78 smallest; scaling orders 1.49 train / 0.372 test | `results/biodiesel/chemkan_figure_runs/fig04/*/config.json`, `metrics.json`, `results/biodiesel/figures/fig04_scaling.png`, `docs/reproduction_summary.md` | V. Scaling is a chosen single-seed log-log fit and model exclusion rule. |
| Biodiesel paper-vs-code reaction-rate assignment | `results/biodiesel/reaction_order/fig3_case_trajectories.json`; two `results/biodiesel/reaction_order/*/metrics.json`; `chemkan/scripts/data_gen/generate_biodiesel.py` | V for recorded reaction-order experiment; paper figure attribution remains I. |
| Main noise runs, 0–15% | `results/biodiesel/chemkan_figure_runs/fig05/*/{config.json,metrics.json}`, corresponding `deeponet_figure_runs/fig05/*/` | V. Final time-averaged MSE uses tight evaluation, not necessarily each history's final logged training-tolerance value. |
| MSE increase 0→1% `4.19e-5`, 0→5% `1.05e-3`; 15/0 clean-test ratio `1.84` | `docs/reproduction_summary.md` §3, checked against final `fig05` artifacts | V (rounded). |
| 15% DeepONet/ChemKAN clean-test ratio `2.73`; advantage of ≈2.7 at 10% and 15% | `docs/reproduction_summary.md` §3 and per-noise final `fig05` records | V for **selected tanh** reconstruction; it does not establish universality or authors' original DeepONet configuration. |
| Fig. 3/Fig. 6 held-out species trajectories and 15% MSEs | `results/biodiesel/figures/fig03_trajectories.png`, `fig06_profiles_15pct.png`; project report asset `biodiesel_fig03.png` is byte-identical; other asset is not identical in bytes | V for numeric result from release summary/plots; image variants require visual comparison before claiming byte-identity. |
| Biodiesel rate-parity MG rRMSE ≈0.24 train /0.28 test, others ≈0.09–0.14 | `results/biodiesel/interpretability/summary.json`, `rate_parity_all_reference_states`; `results/biodiesel/interpretability/fig3_rate_parity.png` | V. Evaluated at 600 train +300 test **reference** states, not arbitrary off-trajectory points. |
| Hydrogen 100k Stage 1 /200k Stage 2 with selected final 200k | `results/hydrogen/S1_seed0/`, `results/hydrogen/S2_seed0_200k/`, `results/hydrogen/eval/S2_seed0_200k.json` | V; staged training and checkpoints retained. |
| Hydrogen 200k held-out Eq. (18) /100 MSE `3.507e-4` | `results/hydrogen/eval/S2_seed0_200k.json`, `summary` and held-out condition | V. Separate logged training loss vs checkpoint evaluation. |
| Hydrogen dense 441-case median `4.908857e-4`, minimum `3.833662e-5`, maximum `6.115575e-2` | `results/hydrogen/tables/main_S2_seed0_200k_float64_rtol1e-10_generalization_441.json` | V; eval rtol `1e-10`, atol `1e-12`, float64. Reconstructed 21×21 coordinates are I, not source-specified. |
| Hydrogen ignition 30-case median relative error `1.3563218%` and temperature-rise medians 1604.90 K model /1580.73 K reference | `results/hydrogen/tables/hydrogen_ignition_delay_main_S2_seed0_200k_float64_rtol1e-10.json` | V. Difference between separate medians is 24.17 K; do not imply median of per-case differences is exactly that. |
| Methane Stage 1 partial history ~54,790 epochs; Stage 2 200k started from Stage-1 50k archive | `results/methane/S1_seed0/stage1_history.csv`, `S1_seed0/stage1_archives.json`, `S1_seed0/host_transfer.json`, `S2_seed0/config.json`, `S2_seed0/stage2_history.csv` | V. Stage 1 does **not** have a final checkpoint at 54,790 in the release. |
| Methane model 52 species + T, 1892 parameters; selected Stage-2 checkpoint 180k based on training-only accuracy | `results/methane/S2_seed0/config.json`, `results/methane/S2_seed0/archive/`, `chemkan/notebooks/methane_extension.ipynb` | V. Selection uses mean of 35 training conditions, not held-out MSE. |
| Methane held-out MSE `1.162551e-3`, grid train-median `9.724221e-4`, unseen median `9.209394e-4`, max `1.266887e-2` | `results/methane/generalization/methane_grid_441.json`, `mse_time_avg` | V. 441 includes 35 training and 406 unseen; held-out is included among unseen. |
| Methane selected 180k vs final 200k behavior | `chemkan/notebooks/methane_extension.ipynb`, `results/methane/S2_seed0/stage2_history.csv`; `results/methane/S2_seed0/archive/` | V for archived notebook calculations/history. Do not call the optimized model converged. |
| Hydrogen Julia ChemKAN ≈1.60 ms, Cantera ≈3.94 ms, Arrhenius.jl ≈10.07 ms | `results/inference_benchmark/hydrogen/results.json`, Julia subrecords and `results_table.csv` | V at measured settings on Apple M1. Comparisons use different solver tolerances and implementation languages. |
| Methane Julia ChemKAN ≈9.93 ms, Cantera ≈39.57 ms, Arrhenius.jl ≈134.29 ms | `results/inference_benchmark/methane/results.json`, Julia subrecords and `results_table.csv` | V with same tolerance/hardware caveat. |
| Biodiesel PyTorch ≈67.17 ms, Julia ChemKAN ≈0.69 ms, Python reference ≈1.20 ms, Julia reference ≈0.20 ms | `results/inference_benchmark/biodiesel/results.json`, Julia subrecords and `results_table.csv` | V with separate tight biodiesel tolerance convention. |
| FSA algorithm and current validation notebook | `chemkan/src/chemkan/fsa.py`, `chemkan/tests/test_fsa.py`, `chemkan/notebooks/forward_sensitivity_analysis.ipynb` | V for current code/tests and saved notebook outputs; not proof historical validation artifacts were included. |
| **Earlier exhaustive FSA validation**: seven sections A–G, original C/E failures handled by Amendment 1; optional finite differences inconclusive | **development only:** `results/experiments/validation/fsa/README.md`; additional JSON validation records on dev | D. A/B/D/F PASS; C/E PASS *after adding one tighter tolerance level, with original thresholds retained*; G INCONCLUSIVE. These historical JSON files are **not** in release. |
| Historical Julia/FSA loss ratios at 500/2k/5k of `1.0001/0.9999/0.997`; late medians `0.94` train and `1.19` val vs Julia | **development only:** `docs/authors_materials.md` §2, original Julia comparison records | D. This is a controlled prior experiment, not reproduced by current release figure launchers. |

## D. Report figure evidence / identity

All **18 `\\includegraphics` references** in `main.tex` resolve to files in `report/ChemKAN_report_assets/`. The following report assets are **byte-identical** to release plots: `biodiesel_fig03.png` ↔ `results/biodiesel/figures/fig03_trajectories.png`; `hydrogen_fig08b_200k.png` ↔ `results/hydrogen/figures/fig08b_hydrogen_ignition_delay_200k.png`; `interpretability_biodiesel_edges_layer1.png` ↔ `results/biodiesel/interpretability/fig1_edges_layer1.png`; `interpretability_biodiesel_edges_layer2.png` ↔ `results/biodiesel/interpretability/fig2_edges_layer2.png`; `interpretability_biodiesel_rate_parity.png` ↔ `results/biodiesel/interpretability/fig3_rate_parity.png`. The other figures included by `main.tex` are **not byte-identical** to release PNGs; this alone proves neither an error nor identity of plotting data. This statement concerns the 18 included figures, not every image bundled in `ChemKAN_report_assets/`. For example, the unused `runtime_vs_accuracy_biodiesel.png` and `runtime_vs_accuracy_methane.png` are byte-identical to their release counterparts. Source traces should be established from plot scripts and JSON/CSV records before asserting equivalence.

| Report topic | Project image(s) in report source | Release-side source or evidence |
|---|---|---|
| Biodiesel Figs. 3–6 | `biodiesel_fig03.png`, `fig04_scaling.png`, `fig05a_noise.png`, `fig05b_loss_histories.png`, `fig06_profiles_15pct.png` | `results/biodiesel/figures/`, corresponding figure-run configs/metrics/histories |
| Hydrogen Figs. 7–8 | `hydrogen_fig07_200k_tight.png`, `hydrogen_fig08a_200k_tight.png`, `hydrogen_fig08b_200k.png` | `results/hydrogen/figures/` and `results/hydrogen/tables/`; tight eval JSON above |
| Methane extension | `methane_180k_heldout_trajectory.png`, `methane_generalization_441.png`, `methane_generalization_441_full_range_log_scale.png` | `results/methane/figures/`, `results/methane/generalization/`, methane notebook |
| Inference | `runtime_vs_accuracy.png` | `results/inference_benchmark/results_table.csv`, `results/inference_benchmark/runtime_vs_accuracy.png` |
| Interpretability | `interpretability_biodiesel_edges_layer1.png`, `interpretability_biodiesel_rate_parity.png`, appendix `interpretability_biodiesel_edges_layer2.png` | `results/biodiesel/interpretability/summary.json` and plots |
| Training histories | `hydrogen_loss_curves_stage1_stage2_200k.png`, `methane_training_histories.png` | `results/hydrogen/*/stage*_history.csv`, `results/methane/*/stage*_history.csv` |

## E. Methodological decisions and limitations — attribution boundaries

1. `train.py` is the intentionally simple **kinetic-only, constant-temperature** example. Hydrogen and methane use separate entry points backed by the shared two-stage combustion trainer. Do not describe `train.py` as a universal full-combustion example.
2. Paper shows a Swish base term, while final code uses no base term. Author evidence belongs to `ChemKANAuthors2026`; released Julia support belongs to `ChemKANRepo2026`.
3. The original paper's DeepONet activation and learning rate are under-specified. Tanh, raw time and Adam `1e-2` are **this project's selected reconstruction**, not recovered authors' settings.
4. Comparing ChemKAN and DeepONet at noise levels gives **single-seed evidence** only. In particular, the paper's claim of ChemKAN's lower test error at **all** noise levels is **not reproduced** by this reconstruction.
5. `dS/dt=J_xS+J_theta` is textbook FSA mathematics; the numerical correctness of current code is supported by tests and notebook, and older comprehensive validation is **development-only**. Test or tolerance amendment does not mean a threshold was relaxed.
6. Continuous-FSA gradients and direct-autograd-through-solver gradients can differ at loose tolerances; reported closeness to Julia is a result, while a derivative-aware error-control causal explanation is a **hypothesis**.
7. Julia inference speedups over Cantera are measured **at different tolerances, on a specified CPU and implementation**, not at guaranteed matched numerical error. Python ChemKAN is slower in that setting.
8. Methane 180k was selected using **training conditions**, despite continuing to 200k; report both selected and final checkpoints without implying convergence.
9. Rate parity at sampled **reference trajectory states** is not a guarantee of correct rates off-trajectory or identification of physical reactions.
10. The development archive retains earlier runs and full histories. The **release intentionally omits many exploratory runs and columns from 33 final CSV histories**; retained values are unchanged. Do not claim every experimental diagnostic from dev is public.
11. Source materials from **private author correspondence** and some authors' initial-condition data may have redistribution constraints not resolved by code licensing. Avoid implying the emails are in a public repository or that there is public access to `dev` without verifying it.

## F. Report/course provenance and release packaging

- **Package location:** `docs/report_poster_and_proposal_project/ChemKAN_IDP_Report/` contains the supplied compiled report `Bankoglu_Project_Report.pdf`, `main.tex`, `references.bib`, this evidence map, `README.md`, the figure assets and `MANIFEST.sha256`.
- **Proposal:** the sibling `proposal/` directory contains `main.tex`, `references.bib` and `Bankoglu_ChemKAN_Proposal.pdf`. These retain the approved proposal's RQ1–RQ3 framing and CRISP-DM choice. The report's exact project KPI interpretation is its own evaluation.
- **Course provenance:** the report records the title, student information, supervisors, abstract, research questions, methodology, evaluated criteria, state-of-the-art comparison, and selected domain lecture with completion date. The stated lecture completion/certificate date is 24 July 2026. A completion certificate is not included in the report/proposal package, so that date is not independently verified here.
- **Document checks:** `Bankoglu_Project_Report.pdf` has **30 pages**. The current LaTeX source has **18 included figures**, all resolving to existing assets; **16 bibliography entries**, all cited; and no unresolved `\ref`/`\eqref`/`\autoref` labels in the source checks.
- **Evidence-map reference:** the source's “Source and Reproducibility Notes” appendix correctly names `REPORT_CLAIM_EVIDENCE_MAP.md`, which is present beside `main.tex`. The identical repository copy is at `docs/REPORT_CLAIM_EVIDENCE_MAP.md`.
- **Package README:** `README.md` describes the current PDF filename, page count, contents and compilation instructions. Repository documentation no longer links to an absent public copy of the private author-materials archive.
- **Integrity manifest:** `MANIFEST.sha256` covers all **31 non-hidden package files**, excluding the manifest itself, using paths relative to the report directory. It has been regenerated for the current filenames and contents and verified with `shasum -a 256 -c MANIFEST.sha256`. Regenerate it after any file edits, renames or PDF compilation. Matching hashes establish file integrity, not source-to-PDF build equivalence.

## G. What this audit does **not** establish

- It does **not** rerun 200k-epoch training, re-execute Julia benchmarks, or independently authenticate personal emails.
- It does **not** prove that report image variants which differ in bytes were generated from exactly the same plot code or numerical arrays as release.
- It does **not** establish redistribution permissions for every third-party file, author-provided data, or the privately communicated material.
- It does **not** prove that the supplied `Bankoglu_Project_Report.pdf` is pixel-identical to a **fresh successful compilation** of the current `main.tex`. The manifest verifies the supplied files; no fresh LaTeX build was performed for this packaging update.
- It does **not** establish single-seed generalization across random initializations.

**Maintaining the two copies:** edit `docs/REPORT_CLAIM_EVIDENCE_MAP.md`, copy it to `docs/report_poster_and_proposal_project/ChemKAN_IDP_Report/REPORT_CLAIM_EVIDENCE_MAP.md`, and then regenerate the report manifest. The copies serve repository readers and readers of the standalone report package, respectively, and should remain identical.
