from adtherm import AdTherm
from ase.io import read
from ase.io.trajectory import Trajectory
from ase import Atom, Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.constraints import FixAtoms
import numpy as np
from ase.build import fcc111, add_adsorbate, molecule
import copy
from ase.optimize import BFGS

########## Setting up QE calcs with ASE #########
pwx="/oscar/runtime/software/external/quantum-espresso/7.1/bin/pw.x"
ppath= '/users/kbadger1/espresso/pseudo/pslibrary.1.0.0/pbe/PSEUDOPOTENTIALS/'
espresso_profile = EspressoProfile(binary=pwx,
                                   parallel_info={'mpirun': True,
                                                  '-np': 48},
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
                    npsin=2,
                    mixing_beta=0.3)

########### Initialize ###########
size = [3,3,4]
atoms = fcc111(symbol='Ni', size=size)
adsorbate = molecule('CH3')
add_adsorbate(atoms, adsorbate, 1.5, 'fcc')
atoms.center(vacuum=8.5, axis=2)
atoms.set_initial_magnetic_moments([0.6]*len(atoms))
c = FixAtoms(indices=[atom.index for atom in atoms if atom.index < 18])
atoms.set_constraint(c)
atoms.calc = calc

########### Optimize to minima ###########
opt = BFGS(atoms, trajectory='relax.traj')
opt.run(fmax=0.05)

########### Set up AdTherm object ###########
indices = [36, 37, 38, 39]
ads = AdTherm(atoms, indices)

############ Hessian points ###########
hessian_points, hessian_coords = ads.generate_hessian_points()
hessian_traj = Trajectory('hessian_set.traj', 'w')
for i in range(len(hessian_points)):
    hessian_points[i].calc = copy.copy(calc)
    hessian_points[i].get_forces()
    hessian_traj.write(hessian_points[i])
H = ads.generate_rigid_hessian(hessian_points)

########### Generate the rest of the points ###########
gauss_points, gauss_coords = ads.generate_gauss_points(H, n_gauss=3, scale_gauss=0.1)
sobol_points, sobol_coords = ads.generate_sobol_points(n_sobol=3)
random_points, random_coords = ads.generate_random_points(n_random=3)
gauss_traj = Trajectory('gauss_set.traj', 'w')
sobol_traj = Trajectory('sobol_set.traj', 'w')
random_traj = Trajectory('random_set.traj', 'w')

########### Run dft calcs ##################
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

########### get y_train ###########
ads.write_y_train([hessian_points, gauss_points, sobol_points, random_points],
                  ['hessian_y_train.dat',
                   'gauss_y_train.dat',
                   'sobol_y_train.dat',
                   'random_y_train.dat'])

########## get stencil points ########
ads.evaluate_stencil_points([hessian_points, gauss_points, sobol_points, random_points],
                            [hessian_coords, gauss_coords, sobol_coords, random_coords])


