import numpy as np
import numpy.linalg as LA
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.constraints import FixExternals
#from io import write_coord
from .functions import *
from new_geometry_generation.generate import coord_generate 


class AdTherm:

    def __init__(self,
                 atoms,
                 indices,
                 z_below=1,
                 z_above=5):

        self.atoms = atoms
        self.indices = indices
        self.adsorbate = self.atoms[indices].copy()
        self.z_low = -z_below + np.min(self.adsorbate.positions[:, 2])
        self.z_high = z_above + np.min(self.adsorbate.positions[:, 2])

        self.N_atoms_in_adsorbate = len(self.adsorbate)
        self.uc_x = atoms.get_cell_lengths_and_angles()[0]
        self.uc_y = atoms.get_cell_lengths_and_angles()[1]
        self.adsorbate_center_of_mass = self.adsorbate.get_center_of_mass()
        self.min_cutoff = 0.2
        self.max_cutoff = 10
        self.hessian_displacement_size = 1e-2

        self.rotate = True
        self.N_hessian = 12
        self.ndim = 6
        if self.N_atoms_in_adsorbate == 2:
            self.N_hessian = 10
            self.ndim = 5
        elif self.N_atoms_in_adsorbate == 1:
            self.N_hessian = 6
            self.ndim = 3
            self.rotate = False


    def generate_hessian_points(self, write_traj=True, write_x_train=True):
        n = self.N_hessian
        dft_list, coord = coord_generate(self, 'hessian', n)
#        if write_xtrain:
        np.savetxt('hessian_x_train.dat', coord)
        #if write_traj:
        traj = Trajectory('hessian_set.traj','w')
        for image in dft_list:
            traj.write(image)
        return dft_list

    def generate_gauss_points(self, rigid_hessian, n_gauss, scale_gauss = 1):
        self.rigid_body_hessian = rigid_hessian
        n = n_gauss
        self.scale_gauss = scale_gauss
        dft_list, coord = coord_generate(self, 'gauss', n)
        np.savetxt('gauss_x_train.dat', coord)
        traj = Trajectory('gauss_set.traj','w')
        for image in dft_list:
            traj.write(image)
        return dft_list

    def generate_sobol_points(self, n_sobol):
        n = n_sobol
        dft_list, coord = coord_generate(self, 'sobol', n)
        np.savetxt('sobol_x_train.dat', coord)
        traj = Trajectory('sobol_set.traj','w')
        for image in dft_list:
            traj.write(image)
        return dft_list

    def generate_random_points(self, n_random):
        n = n_random
        dft_list, coord = coord_generate(self, 'random', n)
        np.savetxt('random_x_train.dat', coord)
        traj = Trajectory('random_set.traj','w')
        for image in dft_list:
            traj.write(image)
        return dft_list

    def calculate_hessian(self, dft_list, fname):
        force_list = np.zeros([3 * len(self.indices), self.N_hessian])
        displacement_list = np.zeros([3 * len(self.indices), self.N_hessian])
        E_list = np.zeros(self.N_hessian)
        for i, img in enumerate(dft_list):
            f_atoms = img.get_forces()[self.indices]
            force_list[:, i] = f_atoms.reshape(-1)
            E_list[i] = img.get_potential_energy()
            disp_atoms = img.positions[self.indices]
            displacement_list[:, i] = disp_atoms.reshape(-1)
        #np.savetxt('hessian_y_train.dat', E_list)
        self.rigid_body_hessian = calculate_rigid_hessian(
                    self,
                    displacement_list,
                    force_list)
        np.savetxt(fname, self.rigid_body_hessian)
        return self.rigid_body_hessian

    def write_y_train(self, dft_lists, filenames):

        for j, dft_list in enumerate(dft_lists): 
            E = np.zeros(len(dft_list))
            for i, img in enumerate(dft_list):
                E[i] = img.get_potential_energy()
            np.savetxt(filenames[j], E)




def run(self):

        if self.bootstrap:
            booty = open('bootstrap_y_train.dat', 'w')
            bootx = open('bootstrap_x_train.dat', 'w')
        dft_list, coord = self.coord_generate('hessian', self.N_hessian)
        force_list = np.zeros([3 * len(self.indices), self.N_hessian])
        displacement_list = np.zeros([3 * len(self.indices), self.N_hessian])
        E_list = np.zeros(self.N_hessian)
        y_train = open('hessian_y_train.dat', 'w')
        for i, img in enumerate(dft_list):
            f_atoms = img.get_forces()[self.indices]
            force_list[:, i] = f_atoms.reshape(-1)
            E_list[i] = img.get_potential_energy()
            y_train.write(str(E_list[i]) + "\n")
            disp_atoms = img.positions[self.indices]
            displacement_list[:, i] = disp_atoms.reshape(-1)
        y_train.close()
        self.rigid_body_hessian = self.calculate_rigid_hessian(
                    displacement_list,
                    force_list)
        np.savetxt('rigid_hessian.out', self.rigid_body_hessian)

        if self.N_gauss:
            dft_list, coords = self.coord_generate('gauss', self.N_gauss)
            y_train = open('gauss_y_train.dat', 'w')
            for i, img in enumerate(dft_list):
                if self.relax_internals:
                    c = FixExternals(img, self.indices)
                    img.set_constraint(c)
                    dyn = BFGS(img)
                    dyn.run(fmax=0.05)
                force = img.get_forces()[self.indices].reshape(-1)
                E = img.get_potential_energy()
                y_train.write(str(E) + '\n')
                if self.bootstrap:
                    xi, y = self.bootstrap_points(force, E, img, coords[i, :])
                    for j in range(2 * self.ndim):
                        x = xi[j, :]
                        paramline = "%.6F\t%.6F\t%.6F\t%.6F\t%.6F\t%.6F\n" % (
                                x[0], x[1], x[2], x[3], x[4], x[5])
                        bootx.write(paramline)
                        booty.write(str(y[j]) + '\n')
            y_train.close()

        if self.N_sobol:
            dft_list, coord = self.coord_generate('sobol', self.N_sobol)
            y_train = open('sobol_y_train.dat', 'w')
            for i, img in enumerate(dft_list):
                if self.relax_internals:
                    c = FixExternals(img, self.indices)
                    img.set_constraint(c)
                    dyn = BFGS(img)
                    dyn.run(fmax=0.05)
                force = img.get_forces()[self.indices].reshape(-1)
                E = img.get_potential_energy()
                y_train.write(str(E) + '\n')
                if self.bootstrap:
                    xi, y = self.bootstrap_points(force, E, img, coords[i, :])
                    for j in range(2 * self.ndim):
                        x = xi[j, :]
                        paramline = "%.6F\t%.6F\t%.6F\t%.6F\t%.6F\t%.6F\n" % (
                                x[0], x[1], x[2], x[3], x[4], x[5])
                        bootx.write(paramline)
                        booty.write(str(y[j]) + '\n')
            y_train.close()

        if self.N_random:
            dft_list, coord = self.coord_generate('random', self.N_random)
            y_train = open('random_y_train.dat', 'w')
            for i, img in enumerate(dft_list):
                if self.relax_internals:
                    c = FixExternals(img, self.indices)
                    img.set_constraint(c)
                    dyn = BFGS(img)
                    dyn.run(fmax=0.05)
                force = img.get_forces()[self.indices].reshape(-1)
                E = img.get_potential_energy()
                y_train.write(str(E) + '\n')
                if self.bootstrap:
                    xi, y = self.bootstrap_points(force, E, img, coords[i, :])
                    for j in range(2 * self.ndim):
                        x = xi[j, :]
                        paramline = "%.6F\t%.6F\t%.6F\t%.6F\t%.6F\t%.6F\n" % (
                                x[0], x[1], x[2], x[3], x[4], x[5])
                        bootx.write(paramline)
                        booty.write(str(y[j]) + '\n')
            y_train.close()
        if self.bootstrap:
            bootx.close()
            booty.close()
