from adtherm import AdTherm
from ase.io import Trajectory
from ase import Atom, Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
import numpy as np
from ase.build import fcc111, add_adsorbate, molecule
from ase.optimize import BFGS
from ase.vibrations import Vibrations, VibrationsData
import os
import copy
########## Initialize ###########
size = [3,3,4]
atoms1 = fcc111(symbol='Ni', size=size)
adsorbate1 = molecule('CH3')
add_adsorbate(atoms1, adsorbate1, 1.5, 'fcc')
atoms1.center(vacuum=8.5, axis=2)
c = FixAtoms(indices=[atom.index for atom in atoms1 if atom.index < 36])
atoms1.set_constraint(c)
atoms1.calc = EMT()

atoms2 = fcc111(symbol='Ni', size=size)
adsorbate2 = adsorbate1.copy()
add_adsorbate(atoms2, adsorbate2, 1.5, 'hcp')
atoms2.center(vacuum=8.5, axis=2)
atoms2.set_initial_magnetic_moments([0.6]*len(atoms2))
c = FixAtoms(indices=[atom.index for atom in atoms2 if atom.index < 36])
atoms2.set_constraint(c)
atoms2.calc = EMT()
calc = EMT()
########### Optimize to minima ###########
os.mkdir('minima')
opt1 = BFGS(atoms1, trajectory='minima/relax1.traj')
opt1.run(fmax=0.001)
opt2 = BFGS(atoms2, trajectory='minima/relax2.traj')
opt2.run(fmax=0.001)

################ vibs ######################
indices = [36, 37, 38, 39]
vib1 = Vibrations(atoms1, indices, name='vib1')
vib1.run()
vib1.summary()
dat1 = vib1.get_vibrations(atoms1)
hessian_3N_1 = dat1.get_hessian_2d()
vib2 = Vibrations(atoms2, indices, name='vib2')
vib2.run()
vib2.summary()
dat2 = vib2.get_vibrations(atoms2)
hessian_3N_2 = dat2.get_hessian_2d()
########### Set up AdTherm object ###########

ads = AdTherm([atoms1, atoms2], indices, [hessian_3N_1, hessian_3N_2])

########### Generate points ###########
gauss_points, gauss_coords  = ads.generate_gauss_points(n_gauss=[10, 10], temperature=500)
sobol_points, sobol_coords = ads.generate_sobol_points(n_sobol=3)
random_points, random_coords = ads.generate_random_points(n_random=3)

########### Run dft calcs ##################
os.mkdir('gauss')
os.mkdir('sobol')
os.mkdir('random')
os.mkdir('stencil')

gauss_traj = Trajectory('gauss/gauss_set.traj', 'w')
sobol_traj = Trajectory('sobol/sobol_set.traj', 'w')
random_traj = Trajectory('random/random_set.traj', 'w')

for i in range(len(sobol_points)):
    sobol_points[i].calc = copy.copy(calc)
    sobol_points[i].get_forces()
    sobol_traj.write(sobol_points[i])
for i in range(len(gauss_points)):
    gauss_points[i].calc = copy.copy(calc)
    gauss_points[i].get_forces()
    gauss_traj.write(gauss_points[i])
for i in range(len(random_points)):
    random_points[i].calc = copy.copy(calc)
    random_points[i].get_forces()
    random_traj.write(random_points[i])

########### get training points ###########
ads.write_x_train([gauss_coords, sobol_coords, random_coords],
                  ['gauss/gauss_x_train.dat',
                   'sobol/sobol_x_train.dat',
                   'random/random_x_train.dat'])

ads.write_y_train([gauss_points, sobol_points, random_points],
                  ['gauss/gauss_y_train.dat',
                   'sobol/sobol_y_train.dat',
                   'random/random_y_train.dat'])

ads.write_minima_info(['minima/minima_x_train.dat',
                       'minima/minima_y_train.dat'])

ads.evaluate_stencil_points([gauss_points, sobol_points, random_points],
                            [gauss_coords, sobol_coords, random_coords],
                            ['stencil/stencil_x_train.dat',
                             'stencil/stencil_y_train.dat'])



