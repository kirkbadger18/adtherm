from adtherm import AdTherm
from ase.io import read
from ase.io.trajectory import Trajectory
from ase import Atom, Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
import numpy as np
from ase.build import fcc111, add_adsorbate, molecule
import copy
from ase.optimize import BFGS
from ase.vibrations import Vibrations, VibrationsData
import os
########## Setting up QE calcs with ASE #########
pwx="/oscar/runtime/software/external/quantum-espresso/7.1/bin/pw.x"
ppath= '/users/kbadger1/espresso/pseudo/pslibrary.1.0.0/pbe/PSEUDOPOTENTIALS/'
espresso_profile = EspressoProfile(binary=pwx,
                                   parallel_info={'srun': '--mpi=pmix'},
                                   pseudo_dir=ppath)

calc = Espresso(profile=espresso_profile,
                    pseudopotentials={'Ni': 'Ni.pbe-n-kjpaw_psl.1.0.0.UPF',
                                      'C': 'C.pbe-n-kjpaw_psl.1.0.0.UPF',
                                      'H': 'H.pbe-kjpaw_psl.1.0.0.UPF'},
                    vdw_corr='dft-d3',
                    dftd3_version=4,
                    dftd3_threebody=True,
                    kpts=(5, 5, 1), #1st opt setting
                    occupations='smearing',
                    smearing='marzari-vanderbilt',
                    degauss=0.01,
                    ecutwfc=40,
                    nosym=True,
                    mixing_mode='local-TF',
                    tprnfor=True,
                    tstress=True,
                    npsin=1,
                    mixing_beta=0.3)

########### Initialize ###########
size = [3,3,4]
atoms1 = fcc111(symbol='Ni', size=size)
adsorbate1 = molecule('CH3')
add_adsorbate(atoms1, adsorbate1, 1.5, 'fcc')
atoms1.center(vacuum=8.5, axis=2)
atoms1.set_initial_magnetic_moments([0.6]*len(atoms1))
c = FixAtoms(indices=[atom.index for atom in atoms1 if atom.index < 18])
atoms1.set_constraint(c)
atoms1.calc = calc
atoms2 = fcc111(symbol='Ni', size=size)
adsorbate2 = adsorbate1.copy()
add_adsorbate(atoms2, adsorbate2, 1.5, 'hcp')
atoms2.center(vacuum=8.5, axis=2)
atoms2.set_initial_magnetic_moments([0.6]*len(atoms2))
c = FixAtoms(indices=[atom.index for atom in atoms2 if atom.index < 18])
atoms2.set_constraint(c)
atoms2.calc = calc

########### Optimize to minima ###########
os.mkdir('minima')
opt1 = BFGS(atoms1, trajectory='minima/relax1.traj')
opt1.run(fmax=0.025)
opt2 = BFGS(atoms2, trajectory='minima/relax2.traj')
opt2.run(fmax=0.025)

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
ads = AdTherm([atoms1, atoms2], [hessian_3N_1, hessian_3N_2], indices)

########### Generate points ###########
gauss_points, gauss_coords  = ads.generate_gauss_points(n_gauss=[3, 3], scale_gauss=[1, 1])
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



