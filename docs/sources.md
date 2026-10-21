# Sources

## Core references (used in this project)

These are used in the code, the datasets, the baselines or the comparisons. The column
"used for" says where.

| Reference | Used for | Links |
|---|---|---|
| B. C. Koenig, S. Kim, S. Deng. *ChemKANs for combustion chemistry modeling and acceleration.* Phys. Chem. Chem. Phys. 27, 17313–17330 (2025) | the method and every reproduced result | [DOI 10.1039/D5CP02009C](https://doi.org/10.1039/D5CP02009C), [arXiv:2504.12580](https://arxiv.org/abs/2504.12580), [local copy](paper/d5cp02009c.pdf) |
| ChemKAN released biodiesel example (DENG-MIT/ChemKAN, commit `d7aa5ab`) | the settings of the final runs (`chemkan/scripts/reference_components.py`); the equivalence check in `results/biodiesel/equivalence/` | [DENG-MIT/ChemKAN](https://github.com/DENG-MIT/ChemKAN) |
| LeanKAN (Koenig et al., *Neural Networks*, 2025) | the LeanKAN layer of the kinetic core (`chemkan/src/chemkan/kan/`) | [paper](https://www.sciencedirect.com/science/article/pii/S0893608025007646), [DENG-MIT/LeanKAN](https://github.com/DENG-MIT/LeanKAN) |
| KAN-ODEs (*Kolmogorov–Arnold Network Ordinary Differential Equations*) | the KAN-as-ODE-right-hand-side formulation ChemKAN builds on; the training-loop reference in `docs/implementing_chemkan_from_scratch_guide.md` | [arXiv:2407.04192](https://arxiv.org/abs/2407.04192), [DENG-MIT/KAN-ODEs](https://github.com/DENG-MIT/KAN-ODEs) |
| L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis. *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.* Nat. Mach. Intell. 3, 218–229 (2021) | the DeepONet baseline (`deeponet/`); its conventions follow DeepXDE 0.11.2 | [DOI 10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5), [lululxvi/deeponet](https://github.com/lululxvi/deeponet) (copy in `deeponet/upstream_reference/`), [lululxvi/deepxde](https://github.com/lululxvi/deepxde) |
| ChemNODE (Owoyele & Pal; Ref. 17 of the ChemKAN paper) | the hydrogen reactor setup that ChemKAN follows; the 50-point grid of an earlier hydrogen dataset | cited through the ChemKAN paper |
| Cantera 3.0.0 | hydrogen (`h2o2.yaml`) and methane data generation; inference-benchmark reference | [cantera.org](https://cantera.org) |
| GRI-Mech 3.0 | the methane mechanism (extension) | [combustion.berkeley.edu/gri-mech](http://combustion.berkeley.edu/gri-mech/) |
| torchdiffeq | the Tsit5 ODE solver (vendored in `chemkan/third_party/torchdiffeq/`) | [rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) |
| SciMLSensitivity.jl, "Sensitivity Math Details" | the forward-sensitivity formulation of `chemkan/src/chemkan/fsa.py` | [docs.sciml.ai](https://docs.sciml.ai/SciMLSensitivity/stable/sensitivity_math/) |
| Arrhenius.jl | the Julia detailed-chemistry reference of the inference benchmark | [DENG-MIT/Arrhenius.jl](https://github.com/DENG-MIT/Arrhenius.jl) |

## Further reading

Collected during the project; not used in the code or the results.

**Kolmogorov–Arnold networks, general**

| Resource | Link |
|---|---|
| PyKAN documentation and repository | [kindxiaoming.github.io/pykan](https://kindxiaoming.github.io/pykan/intro.html), [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan) |
| efficient-kan (B-spline KAN; a structural reference in the from-scratch guide) | [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) |
| KolmogorovArnold.jl | [vpuri3/KolmogorovArnold.jl](https://github.com/vpuri3/KolmogorovArnold.jl) |
| Awesome KAN (resource list) | [mintisan/awesome-kan](https://github.com/mintisan/awesome-kan) |
| A from-scratch KAN implementation (tutorial) | [mlwithouttears.com](https://mlwithouttears.com/2024/05/15/a-from-scratch-implementation-of-kolmogorov-arnold-networks-kan/) |

**Physics-informed KANs and neural operators**

| Resource | Links |
|---|---|
| Kolmogorov–Arnold-Informed Neural Network (KAINN): forward and inverse problems with KANs | [arXiv:2406.11045](https://arxiv.org/abs/2406.11045); related code: [PredictiveIntelligenceLab/MultiscalePINNs](https://github.com/PredictiveIntelligenceLab/MultiscalePINNs), [PINN solid mechanics](https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN) |
| Adaptive training of grid-dependent physics-informed KANs | [arXiv:2407.17611](https://arxiv.org/abs/2407.17611), [srigas/jaxKAN](https://github.com/srigas/jaxKAN) |
| Long-time integration of parametric evolution equations with physics-informed DeepONets | [PredictiveIntelligenceLab/Long-time-Integration-PI-DeepONets](https://github.com/PredictiveIntelligenceLab/Long-time-Integration-PI-DeepONets) |

**Machine learning for combustion chemistry**

| Resource | Link |
|---|---|
| Combustion chemistry acceleration with DeepONets (*Fuel*) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0016236124003582) |
| Phy-ChemNODE: a physics-informed autoencoder–neural ODE framework for fuel combustion kinetics (NeurIPS ML4PS 2024) | [PDF](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_151.pdf) |
| Stiff-PINN: physics-informed neural network for stiff chemical kinetics (*J. Phys. Chem. A*) | [ACS](https://pubs.acs.org/doi/full/10.1021/acs.jpca.1c05102) |
