"""Element table for the h2o2 mechanism species (hydrogen PINN term).

These are STANDARD chemistry constants (atom counts and molar masses), NOT values
specified by the ChemKAN paper. They are only needed for the optional element-
conservation PINN term. Species order matches the hydrogen dataset:

    [H2, H, O, O2, OH, H2O, HO2, H2O2, N2]

element rows: [H, O, N].
"""

from __future__ import annotations

import torch

SPECIES = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2"]
ELEMENTS = ["H", "O", "N"]

# N_i^k : atoms of element i (row) in species k (column)
ELEMENT_COUNTS = torch.tensor([
    [2, 1, 0, 0, 1, 2, 1, 2, 0],   # H
    [0, 0, 1, 2, 1, 1, 2, 2, 0],   # O
    [0, 0, 0, 0, 0, 0, 0, 0, 2],   # N
], dtype=torch.float32)

ATOMIC_WEIGHTS = torch.tensor([1.008, 15.999, 14.007], dtype=torch.float32)   # H, O, N

MOLAR_WEIGHTS = torch.tensor(                                                  # g/mol
    [2.016, 1.008, 15.999, 31.998, 17.007, 18.015, 33.006, 34.014, 28.014],
    dtype=torch.float32,
)


def assert_species_order(species) -> None:
    """Fail loudly if the dataset species order differs from the element-matrix order.

    The element-count / molar-weight arrays above are positional, so a mismatched
    order would silently compute wrong chemistry in the PINN term.
    """
    if list(species) != SPECIES:
        raise ValueError(
            "Hydrogen dataset species order does not match the element-conservation "
            f"matrix order.\n  dataset:  {list(species)}\n  expected: {SPECIES}"
        )
