# chemkan-tuwien

## About

Welcome to the `chemkan-tuwien` project! This repository contains the source code, materials, and documentation for interdisciplinary project at TU Wien.

**Current status:** data generation **and** the model/training stack are implemented under
`chemkan/` — KAN layers, `KineticCore` + thermodynamic superstructure, biodiesel and
two-stage hydrogen training/evaluation, Tsit5 integration via `torchdiffeq`, and a
direct-autograd sensitivity path (Forward Sensitivity Analysis is still a TODO). See
[`chemkan/README.md`](chemkan/README.md) and [`chemkan/HOW_THE_CODE_WORKS.md`](chemkan/HOW_THE_CODE_WORKS.md).

### Course Information
* **Course:** [194.147 Interdisciplinary Project in Data Science](https://tiss.tuwien.ac.at/course/courseDetails.xhtml?dswid=6763&dsrid=17&semester=2026S&courseNr=194147)

### Core Reference
This project builds upon the concepts of ChemKANs (Chemistry Kolmogorov-Arnold Networks).
* **Reference:** [ChemKANs for Combustion Chemistry Modeling and Acceleration (arXiv)](https://arxiv.org/pdf/2504.12580)

**Citation:**
```bibtex
@article{koenig2025chemkans,
  title={ChemKANs for combustion chemistry modeling and acceleration},
  author={Koenig, Benjamin C and Kim, Suyong and Deng, Sili},
  journal={Physical Chemistry Chemical Physics},
  volume={27},
  number={33},
  pages={17313--17330},
  year={2025},
  publisher={Royal Society of Chemistry}
}
```