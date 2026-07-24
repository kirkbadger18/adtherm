from ase.build import fcc111, add_adsorbate
import numpy as np
from adtherm.geometry_generation import Generator


ontop = fcc111('Cu', size=(3,3,4))
add_adsorbate(ontop, 'H', 1.5, 'ontop')
ontop.center(vacuum=10.0, axis=2)

fcc = fcc111('Cu', size=(3,3,4))
add_adsorbate(fcc, 'H', 1.5, 'fcc')
fcc.center(vacuum=10.0, axis=2)

H_3N_ontop = np.diag(np.ones(3))
H_3N_fcc = np.diag(np.ones(3))
indices = [36]

generator = Generator([ontop, fcc],
                      [H_3N_ontop, H_3N_fcc],
                      indices)

samples = generator.generate_sobol_samples(100)
