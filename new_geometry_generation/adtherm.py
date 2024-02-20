import numpy as np
import numpy.linalg as LA
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from functions import *
from generate import coord_generate 


class AdTherm:

    def __init__(self,
                 minima,
                 hessians_3N,
                 indices,
                 z_below=1,
                 z_above=5):

        self.N_atoms_in_adsorbate = len(indices)
        self.rotate = True
        self.ndim = 6
        if self.N_atoms_in_adsorbate == 2:
            self.ndim = 5
        elif self.N_atoms_in_adsorbate == 1:
            self.ndim = 3
            self.rotate = False

        self.minima = minima
        self.indices = indices
        self.adsorbates = []
        self.rigid_hessians = []
        self.coms = np.zeros([len(minima),3])
        self.minima_coords = np.zeros([len(minima),self.ndim])
        self.E_min = np.zeros([len(self.minima),1])
        for i, minimum in enumerate(minima):
            ads = minimum[self.indices].copy()
            self.adsorbates.append(ads)
            h = project_to_rigid_hessian(self, hessians_3N[i], minimum)
            self.rigid_hessians.append(h)
            self.coms[i,:] = ads.get_center_of_mass()
            self.minima_coords[i,0:3] = self.coms[i,:]
            self.E_min[i] = minimum.get_potential_energy()
            if i != 0:
                self.minima_coords[i,:] = map_coords_to_min0(self, minimum)
        self.z_low = -z_below + np.min(self.coms[:, 2])
        self.z_high = z_above + np.max(self.coms[:, 2])
        self.uc_x = minima[0].get_cell_lengths_and_angles()[0]
        self.uc_y = minima[0].get_cell_lengths_and_angles()[1]
        self.min_cutoff = 0.2
        self.max_cutoff = 100

################ add function t get rhombus info   ##################

    def generate_gauss_points(self, n_gauss, scale_gauss):
        for i in range(len(self.minima)):
            self.scale_gauss = scale_gauss[i]
            n = n_gauss[i]
            if i == 0:
                dft_list, rigid_coords = coord_generate(self, 'gauss', n, i)
            else:
                new_list, new_coords = coord_generate(self, 'gauss', n, i)
                ### need to map coords to that of initial minima #####
                for j in range(len(new_list)):
                    dft_list.append(new_list[j])
                rigid_coords = np.vstack((rigid_coords, new_coords))
        return dft_list, rigid_coords

    def generate_sobol_points(self, n_sobol):
        n = n_sobol
        dft_list, rigid_coords = coord_generate(self, 'sobol', n)
        return dft_list, rigid_coords

    def generate_random_points(self, n_random):
        n = n_random
        dft_list, rigid_coords = coord_generate(self, 'random', n)
        return dft_list, rigid_coords

    def write_x_train(self, coords, fnames):

        for j in range(len(coords)):
            coord = coords[j]
            np.savetxt(fnames[j], coord)

    def write_y_train(self, dft_lists, fnames):

        for j in range(len(dft_lists)):
            dft_list = dft_lists[j]
            E = np.zeros(len(dft_list))
            for i in range(len(dft_list)):
                img = dft_list[i]
                E[i] = img.get_potential_energy()
            np.savetxt(fnames[j], E)

    def evaluate_stencil_points(self, dft_lists, coord_lists, namelist):
        
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
        np.savetxt(namelist[0], x)
        np.savetxt(namelist[1], y)

    def write_minima_info(self, namelist):
        np.savetxt(namelist[0], self.minima_coords)
        np.savetxt(namelist[1], self.E_min)

 
