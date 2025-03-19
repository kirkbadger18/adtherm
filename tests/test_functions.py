from adtherm import AdTherm
from functions import *
from ase.io import Trajectory, read
from ase import Atom, Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
import numpy as np
from ase.build import fcc111, add_adsorbate, molecule
from ase.optimize import BFGS
from ase.vibrations import Vibrations, VibrationsData
import os
import copy
import numpy as np

def write_path(path, atoms, idx, name):
        path_traj = Trajectory(name, 'w')
        points = 0.6 * np.sin(np.linspace(0, 2*np.pi, 30))
        for point in points:
            atoms_cpy = atoms.copy()
            disp = point * path
            atoms_cpy.positions[idx, :] += disp.reshape(-1, 3)
            com = atoms_cpy[indices].get_center_of_mass()
            print(com)
            path_traj.write(atoms_cpy)

atoms1 = read('minima/relax1.traj')
atoms2 = read('minima/relax2.traj')
indices = [36, 37, 38]
vib1 = Vibrations(atoms1, indices, name='vib1')
vib1.summary()
dat1 = vib1.get_vibrations(atoms1)
hessian_3N_1 = dat1.get_hessian_2d()
vib2 = Vibrations(atoms2, indices, name='vib2')
vib2.summary()
dat2 = vib2.get_vibrations(atoms2)
hessian_3N_2 = dat2.get_hessian_2d()
ads = AdTherm([atoms1, atoms2], indices, [hessian_3N_1, hessian_3N_2])

B = get_external_basis(ads, atoms1)
#print(atoms1.get_masses())
#print(np.abs(np.matmul(B.T,B).round(4)))
#print(B.round(3))
#atomstst = atoms1.copy()
#com = atomstst[indices].get_center_of_mass()
#print(com)
#write_path(B[:,5], atoms1, indices, 'test.traj')
