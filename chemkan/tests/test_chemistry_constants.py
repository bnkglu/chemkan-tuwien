"""Validate _chemistry.py constants against Cantera's built-in thermodynamic database.

If Cantera is installed, this test automatically verifies that the hardcoded
element counts, atomic weights, and molar weights in _chemistry.py exactly
match the values Cantera computes from the h2o2 mechanism file.

Run:  pytest tests/test_chemistry_constants.py -v
"""

# ── import the hardcoded constants we want to verify ──
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _chemistry import (
    ATOMIC_WEIGHTS,
    ELEMENT_COUNTS,
    ELEMENTS,
    MOLAR_WEIGHTS,
    SPECIES,
)

# ── Cantera is optional; skip gracefully if missing ──
ct = pytest.importorskip("cantera", reason="Cantera not installed — skipping chemistry validation")


@pytest.fixture(scope="module")
def gas():
    """Load the h2o2 mechanism that the hydrogen dataset is based on."""
    return ct.Solution("h2o2.yaml")


class TestElementCounts:
    """Verify ELEMENT_COUNTS[i, k] = number of atoms of element i in species k."""

    def test_shape(self):
        assert ELEMENT_COUNTS.shape == (len(ELEMENTS), len(SPECIES))

    def test_values_match_cantera(self, gas):
        for k, sp_name in enumerate(SPECIES):
            sp = gas.species(sp_name)
            for i, el in enumerate(ELEMENTS):
                expected = sp.composition.get(el, 0.0)
                actual = ELEMENT_COUNTS[i, k].item()
                assert actual == expected, (
                    f"ELEMENT_COUNTS[{el}, {sp_name}]: "
                    f"hardcoded={actual}, cantera={expected}"
                )


class TestAtomicWeights:
    """Verify ATOMIC_WEIGHTS against Cantera's element database."""

    def test_length(self):
        assert len(ATOMIC_WEIGHTS) == len(ELEMENTS)

    def test_values_match_cantera(self, gas):
        for i, el in enumerate(ELEMENTS):
            expected = gas.atomic_weight(el)
            actual = ATOMIC_WEIGHTS[i].item()
            assert abs(actual - expected) < 0.01, (
                f"ATOMIC_WEIGHTS[{el}]: hardcoded={actual}, cantera={expected}"
            )


class TestMolarWeights:
    """Verify MOLAR_WEIGHTS against Cantera's species database."""

    def test_length(self):
        assert len(MOLAR_WEIGHTS) == len(SPECIES)

    def test_values_match_cantera(self, gas):
        for k, sp_name in enumerate(SPECIES):
            expected = gas.species(sp_name).molecular_weight
            actual = MOLAR_WEIGHTS[k].item()
            assert abs(actual - expected) < 0.01, (
                f"MOLAR_WEIGHTS[{sp_name}]: hardcoded={actual}, cantera={expected}"
            )


class TestInternalConsistency:
    """Verify that molar weights = sum of (element_count × atomic_weight)."""

    def test_molar_weights_from_elements(self):
        # M_k = Σ_i  N_i^k × W_i
        computed = (ELEMENT_COUNTS * ATOMIC_WEIGHTS[:, None]).sum(dim=0)
        torch.testing.assert_close(
            MOLAR_WEIGHTS, computed, atol=0.01, rtol=1e-3,
            msg="Molar weights don't match element_counts × atomic_weights"
        )
