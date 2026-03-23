
# symdisc: Continuous Symmetry Discovery & Enforcement Using Vector Fields

## Overview
`symdisc` provides tools for **continuous symmetry discovery** and **symmetry enforcement** using tangent vector fields (infinitesimal generators). The method estimates vector fields whose Lie derivatives annihilate machine‑learning functions (e.g., densities, regressors/classifiers), thereby revealing the underlying continuous symmetries. The same generators can be used to **enforce** invariance or to promote **equivariance** during training via regularization.

> **References**  
> • Shaw, Magner, Moon. *Symmetry Discovery Beyond Affine Transformations*. NeurIPS 2024. [arXiv](https://arxiv.org/abs/2406.03619) · [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cd04ec5aebfbe397c7fd718c35d02e0b-Abstract-Conference.html) · [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cd04ec5aebfbe397c7fd718c35d02e0b-Paper-Conference.pdf)  
> • Shaw, Kunapuli, Magner, Moon. *Continuous Symmetry Discovery and Enforcement Using Infinitesimal Generators of Multi‑parameter Group Actions*. arXiv 2025. [arXiv](https://arxiv.org/abs/2505.08219) · [PDF](https://arxiv.org/pdf/2505.08219v2)

---

## Repository Structure
```
.
├── docs/                                 # Sphinx/MkDocs (under construction)
├── examples/                             # Toy examples and demos
│   ├── basic_circle.py
│   ├── basic_sphere.py
│   ├── CircleSphere_demo.ipynb
│   ├── Tabular_Equiv_NoReg.ipynb
│   ├── Tabular_Equiv_Reg.ipynb
│   ├── Tabular_no_Regularization.ipynb
│   ├── Tabular_Regularization.ipynb
│   ├── Untitled1.ipynb
│   └── Untitled.ipynb
├── Experiments_From_Papers/              # Research reproductions — not part of the core package
│   ├── NeurIPS2024/ ...
│   └── SIAM_MathOfDS/ ...
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── symdisc/
│       ├── discovery/                    # Discovery API + LSE subpackage
│       │   ├── builders.py
│       │   ├── core.py
│       │   ├── function_invariance.py
│       │   └── lse/
│       │       ├── core.py
│       │       ├── distances/
│       │       │   ├── chord.py
│       │       │   └── geodesic_projected.py
│       │       ├── second_order.py
│       │       └── projections/
│       │           ├── penalty_homotopy.py
│       │           └── svd_pseudoinverse.py
│       ├── enforcement/                  # Enforcement strategies
│       │   ├── canonicalization/         # (under construction)
│       │   └── regularization/           # penalties, schedules, utilities
│       │       ├── diagonal.py
│       │       ├── jvp.py
│       │       ├── penalties.py
│       │       ├── schedules.py
│       │       └── utilities.py
│       ├── function_discovery/
│       ├── kernels/
│       ├── utils/
│       └── vector_fields/                # Predefined vector‑field generators
│           ├── euclidean.py
│           ├── images.py
│           ├── kernels.py
│           └── time_series.py
└── tests/
    ├── test_builders_invariance.py
    ├── test_equivariance.py
    ├── test_function_discovery_invariant.py
    ├── test_lse.py
    ├── test_pytorch.py
    └── test_vector_fields.py
```

**Notes**
- The `Experiments_From_Papers/` directory contains reproductions and artifacts for the papers above and is **not** part of the supported package API.
- The `docs/` site is **in progress** and will host user guides, API docs, and tutorials.

---

## Invariance and Equivariance
`symdisc` supports both **invariance** and **equivariance** workflows. See the **examples**:
- `Tabular_Equiv_Reg.ipynb` vs. `Tabular_Equiv_NoReg.ipynb` for equivariance with and without regularization.
- `Tabular_Regularization.ipynb` vs. `Tabular_no_Regularization.ipynb` for invariance via vector‑field regularization.

> Canonicalization (data‑space enforcement) is **under construction** and will be documented as it stabilizes.

---

## Installation
Clone and install:
```bash
git clone https://github.com/KevinMoonLab/SymmetryML.git
cd SymmetryML
pip install -r requirements.txt
pip install -e .
```

---

## Usage
Given the evolving API and active refactors, please consult the **`docs/`** (when available) and the **`examples/`** notebooks for up‑to‑date, runnable code. The examples are currently the best way to see discovery and enforcement in practice.

---

## Citing `symdisc`
If you use this package in academic work, please cite both:

**Symmetry Discovery Beyond Affine Transformations**  
Ben Shaw, Abram Magner, Kevin R. Moon. *NeurIPS 2024*.  
- arXiv: 2406.03619  
- Proceedings: NeurIPS 2024 (Advances in Neural Information Processing Systems 37)

**Continuous Symmetry Discovery and Enforcement Using Infinitesimal Generators of Multi‑parameter Group Actions**  
Ben Shaw, Sasidhar Kunapuli, Abram Magner, Kevin R. Moon. *arXiv 2025*.

---

## License
See the license file included in this repository.
