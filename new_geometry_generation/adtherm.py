import numpy as np
import numpy.linalg as LA
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from functions import *
from generate import coord_generate 


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
        self.max_cutoff = 100
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


    def generate_hessian_points(self):
        n = self.N_hessian
        dft_list, rigid_coords = coord_generate(self, 'hessian', n)
        np.savetxt('hessian_x_train.dat', rigid_coords)
        return dft_list, rigid_coords

    def generate_gauss_points(self, rigid_hessian, n_gauss, scale_gauss = 1):
        self.rigid_body_hessian = rigid_hessian
        n = n_gauss
        self.scale_gauss = scale_gauss
        dft_list, rigid_coords = coord_generate(self, 'gauss', n)
        np.savetxt('gauss_x_train.dat', rigid_coords)
        return dft_list, rigid_coords

    def generate_sobol_points(self, n_sobol):
        n = n_sobol
        dft_list, rigid_coords = coord_generate(self, 'sobol', n)
        np.savetxt('sobol_x_train.dat', rigid_coords)
        return dft_list, rigid_coords

    def generate_random_points(self, n_random):
        n = n_random
        dft_list, rigid_coords = coord_generate(self, 'random', n)
        np.savetxt('random_x_train.dat', rigid_coords)
        return dft_list, rigid_coords

    def generate_rigid_hessian(self, dft_list):
        force_list = np.zeros([3 * len(self.indices), self.N_hessian])
        displacement_list = np.zeros([3 * len(self.indices), self.N_hessian])
        E_list = np.zeros(self.N_hessian)
        for i in range(len(dft_list)):
            img = dft_list[i]
            f_dft = img.calc.results['forces'] 
            force_list[:, i] =f_dft[self.indices,:].reshape(-1)
            E_list[i] = img.calc.results['energy']
            disp_atoms = img.positions[self.indices]
            displacement_list[:, i] = disp_atoms.reshape(-1)
        self.rigid_body_hessian = calculate_rigid_hessian(
                    self,
                    displacement_list,
                    force_list)
        np.savetxt('rigid_body_hessian.dat', self.rigid_body_hessian)
        return self.rigid_body_hessian

    def write_y_train(self, dft_lists, fnames):

        for j in range(len(dft_lists)):
            dft_list = dft_lists[j]
            E = np.zeros(len(dft_list))
            for i in range(len(dft_list)):
                img = dft_list[i]
                E[i] = img.get_potential_energy()
            np.savetxt(fnames[j], E)

    def evaluate_stencil_points(self, dft_lists, coord_lists):
        
        for i in range(len(dft_lists)):
            dft_list = dft_lists[i]
            coord_list = coord_lists[i]
            for j in range(len(dft_list)):
                img = dft_list[j]
                coord = coord_list[j]
                xi, yi = bootstrap_points(self, img, coord)
                if i == 0 and j == 0:
                    x = xi
                    y = yi
                else:
                    x = np.vstack((x,xi))
                    y = np.vstack((y,yi))
        np.savetxt('stencil_x_train.dat', x)
        np.savetxt('stencil_y_train.dat', y)

 
