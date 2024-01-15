from new_geometry_generation.adtherm import AdTherm
from ase.io import read
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS

########### Initialize ###########
atoms = read('POSCAR_10')
del atoms.constraints
c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Ni'])
atoms.set_constraint(c)
calc = EMT()
atoms.calc = calc

########### Optimize to minima ###########
opt = BFGS(atoms, trajectory='relax.traj')
opt.run(fmax=0.01)

########### Set up AdTherm object ###########
indices = [36, 37, 38, 39]
ads = AdTherm(atoms, indices)

############ Hessian points ###########
hessian_points = ads.generate_hessian_points()
for image in hessian_points:
    image.calc = calc
    image.get_forces()

H = ads.calculate_hessian(hessian_points, fname='rigid_body_hessian.dat')

########### Generate the rest of the points ###########
gauss_points = ads.generate_gauss_points(H, n_gauss=10, scale_gauss=0.01)
sobol_points = ads.generate_sobol_points(n_sobol=10)
random_points = ads.generate_random_points(n_random=10)

for image in gauss_points:
    image.calc = calc
    image.get_forces()
for image in sobol_points:
    image.calc = calc
    image.get_forces()
for image in random_points:
    image.calc = calc
    image.get_forces()

########### get y_train ###########
ads.write_y_train([hessian_points, gauss_points, sobol_points, random_points],
                  ['hessian_y_train.dat',
                   'gauss_y_train.dat',
                   'sobol_y_train.dat',
                   'random_y_train.dat'])

