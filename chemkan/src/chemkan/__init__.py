"""Generic ChemKAN reproduction (PyTorch).

This package is experiment-agnostic: it knows nothing about biodiesel, hydrogen, or
any specific chemical system. Dataset dimensions come from the data; architecture
and training choices are passed in by the caller (see ``scripts/``). Paper-vs-
implementation notes live in ASSUMPTIONS.md.
"""

from .dynamics import ChemKANDynamics, KineticDynamics
from .model import ChemKAN, KineticCore, ThermodynamicSuperstructure
from .normalization import MinMaxNormalizer
from .solver import SolverConfig, integrate
from .training import train_full_chemkan, train_kinetic_stage

__all__ = [
    "ChemKAN",
    "ChemKANDynamics",
    "KineticCore",
    "KineticDynamics",
    "MinMaxNormalizer",
    "SolverConfig",
    "ThermodynamicSuperstructure",
    "integrate",
    "train_full_chemkan",
    "train_kinetic_stage",
]
