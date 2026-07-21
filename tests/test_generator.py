"""Smoke tests for constructing a Generator.

For now these only build representative inputs -- a metal slab with a
monatomic, diatomic, and polyatomic adsorbate, plus made-up rigid-body
Hessians -- and check that a Generator can be instantiated. They exist to
lock the constructor's expected inputs while the internals are stubbed.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import add_adsorbate, fcc111

from adtherm.geometry_generation import Generator


def _build_system(adsorbate: Atoms):
    """Return (minimum, adsorbate_indices) for an adsorbate on Cu(111)."""
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
    n_slab = len(slab)
    add_adsorbate(slab, adsorbate, height=2.0, position=(0.0, 0.0))
    adsorbate_indices = list(range(n_slab, len(slab)))
    return slab, adsorbate_indices


def _fake_hessian(n_adsorbate_atoms: int, seed: int = 0) -> np.ndarray:
    """A symmetric 3N x 3N Hessian filled with arbitrary numbers."""
    rng = np.random.default_rng(seed)
    dof = 3 * n_adsorbate_atoms
    a = rng.standard_normal((dof, dof))
    return a + a.T


# (label, adsorbate) -- one atom, two atoms, and >2 atoms.
ADSORBATES = [
    ("monatomic", Atoms("O", positions=[(0.0, 0.0, 0.0)])),
    ("diatomic", Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.13)])),
    ("polyatomic", Atoms("OHH", positions=[(0.0, 0.0, 0.0),
                                           (0.76, 0.0, 0.59),
                                           (-0.76, 0.0, 0.59)])),
]


@pytest.mark.parametrize(
    "label,adsorbate", ADSORBATES, ids=[a[0] for a in ADSORBATES]
)
def test_generator_instantiates(label, adsorbate):
    minimum, adsorbate_indices = _build_system(adsorbate)
    hessians = [_fake_hessian(len(adsorbate_indices))]

    generator = Generator(
        minima=[minimum],
        hessians_3N=hessians,
        adsorbate_indices=adsorbate_indices,
    )

    assert isinstance(generator, Generator)
    assert generator.minima == [minimum]
    assert generator.adsorbate_indices == adsorbate_indices
    assert generator.rigid_domain is not None
    assert generator.coord_converter is not None
    assert generator.traj_factory is not None
