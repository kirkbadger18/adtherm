import numpy as np
from functions import *
from generate import * 
from ase import Atom

class AdTherm:

    def __init__(self,
                 minima,
                 indices,
                 hessians_3N = None,
                 dz_limits= [-0.5, 2],
                 generate_symmetric_minima = False,
                 surface_symmetry_number = None):

        self.minima = minima
        self.indices = indices
        self.hessians_3N = hessians_3N
        self.dz_limits = dz_limits
        self.generate_symmetric_minima = generate_symmetric_minima
        self.surface_symmetry_number = surface_symmetry_number

        self._assess_degrees_of_freedom()
        self._get_minima_information()
        self._set_domain_limits()
        self._remap_minima_into_cell()
        if self.generate_symmetric_minima and self.surface_symmetry_number:
            self._generate_symmetric_minima()
 
    def _set_domain_limits(self):

        self.z_low = self.dz_limits[0] + np.min(self.coms[:, 2])
        self.z_high = self.dz_limits[1] + np.max(self.coms[:, 2])
        self.unit_cell_x = self.minima[0].cell.cellpar()[0]
        self.unit_cell_y = self.minima[0].cell.cellpar()[1]
        self.min_atomic_distance = 0.2
        self.max_atomic_distance = 100
        return

    def _get_minima_information(self):

        self.adsorbates = []
        self.rigid_hessians = []
        self.coms = np.zeros([len(self.minima),3])
        self.minima_coords = np.zeros([len(self.minima),self.ndim])
        self.minima_E = np.zeros([len(self.minima),1])
        for i, minimum in enumerate(self.minima):
            ads = minimum[self.indices].copy()
            self.adsorbates.append(ads)
            self.coms[i,:] = ads.get_center_of_mass()
            self.minima_coords[i,0:3] = self.coms[i,:]
            self.minima_E[i] = minimum.calc.results['energy']
            if i != 0 and self.rotate:
                self.minima_coords[i,3:] = map_rotation_to_min0(self, minimum)
            h = project_to_rigid_hessian(self, self.hessians_3N[i], self.minima_coords[i,:])
            self.rigid_hessians.append(h)
        self.E_ref = np.min(self.minima_E)
        return

    def _assess_degrees_of_freedom(self):
        self.N_atoms_in_adsorbate = len(self.indices)
        self.rotate = True
        self.ndim = 6
        if self.N_atoms_in_adsorbate == 2:
            self.ndim = 5
        elif self.N_atoms_in_adsorbate == 1:
            self.ndim = 3
            self.rotate = False
        return
           
    def _remap_minima_into_cell(self):
        from ase.io import Trajectory
        mintraj = Trajectory('min.traj','w')
        for k in range(len(self.minima)):
            coord = self.minima_coords[k,:]
            valid_z, xy_location = check_xy_coord(self, coord)
            if valid_z and xy_location == 'outside':
                coord, xy_location = move_xy_inside(self, coord)
            self.minima_coords[k,:] = coord
            mintraj.write(self.minima[k], energy= float(self.minima_E[k]))

    def _generate_symmetric_minima(self):
        x = self.minima_coords
        y = self.minima_E 
        x_sym, y_sym = self.generate_symmetric_coords(x,y)
        self.symmetric_minima_coords = x_sym
        self.symmetric_minima_E = y_sym
        H_sym_3N = generate_symmetric_hessians(self,self.hessians_3N)
        H_sym_rigid = []
        for i in range(len(H_sym_3N)):
            h = project_to_rigid_hessian(self, H_sym_3N[i], x_sym[i,3::])
            H_sym_rigid.append(h)
        self.symmetric_rigid_hessians = H_sym_rigid
        from ase.io import Trajectory
        symtraj = Trajectory('sym.traj','w')
        for coord in self.symmetric_minima_coords:
            atoms = manipulate_atoms(self, coord, 0)
            symtraj.write(atoms)
        return

    def get_x_train_from_traj(self,traj):
        coords = np.zeros([len(traj),self.ndim])
        for i, img in enumerate(traj):
            coord = np.zeros(self.ndim)
            ads = img[self.indices].copy()
            com = ads.get_center_of_mass()
            coord[0:3] = com
            if self.rotate:
                coord[3::] = map_rotation_to_min0(self, img)
            valid_z, xy_location = check_xy_coord(self, coord)
            if valid_z and xy_location == 'outside':
                coord, xy_location = move_xy_inside(self, coord) 
            coords[i,:] = coord
        return coords

    def generate_gauss_points(self, n_gauss, temperature):
        kb = 8.617E-5
        for i in range(len(self.minima)):
            self.scale_gauss = temperature * kb
            n = n_gauss[i]
            if i == 0:
                dft_list, rigid_coords = coord_generate(self, 'gauss', n, i)
            else:
                new_list, new_coords = coord_generate(self, 'gauss', n, i)
                for j in range(len(new_list)):
                    dft_list.append(new_list[j])
                rigid_coords = np.vstack((rigid_coords, new_coords))
        return dft_list, rigid_coords

    def generate_sobol_points(self, n_sobol,min_number = 0, seed=1):
        n = n_sobol
        dft_list, rigid_coords = coord_generate(self, 'sobol', n, min_number)
        return dft_list, rigid_coords

    def generate_random_points(self, n_random):
        n = n_random
        dft_list, rigid_coords = coord_generate(self, 'random', n)
        return dft_list, rigid_coords

    def generate_symmetric_coords(self,x_train,y_train):
        sym_num = self.surface_symmetry_number 
        sym_x = np.zeros([(sym_num-1)*len(y_train),6])
        sym_y = np.zeros((sym_num-1)*len(y_train))
        conv = 180 / np.pi
        for i, coord in enumerate(x_train):
            for j in range(sym_num-1):
                angle = conv * (j+1) * 2 * np.pi / sym_num
                atoms = manipulate_atoms(self, coord, 0)
                atoms.rotate(angle,'z')
                sym_x[(sym_num-1)*i+j,:] = self.get_x_train_from_traj([atoms])
                sym_y[(sym_num-1)*i+j] = y_train[i]
        return sym_x, sym_y

    def write_x_train(self, coords, fnames):
        for j in range(len(coords)):
            coord = coords[j]
            np.savetxt(fnames[j], coord, '%1.8e')
        return

    def write_y_train(self, dft_lists, fnames):
        for j in range(len(dft_lists)):
            dft_list = dft_lists[j]
            E = np.zeros(len(dft_list))
            for i in range(len(dft_list)):
                img = dft_list[i]
                E[i] = img.calc.results['energy'] #get_potential_energy()
            np.savetxt(fnames[j], E)
        return

    def evaluate_stencil_points(self, dft_lists, coord_lists, namelist,delta=1e-6):
        for i in range(len(dft_lists)):
            dft_list = dft_lists[i]
            coord_list = coord_lists[i]
            for j in range(len(dft_list)):
                img = dft_list[j]
                coord = coord_list[j]
                xi, yi = bootstrap_points(self, img, coord,delta)
                if i == 0 and j == 0:
                    x = xi
                    y = yi
                else:
                    x = np.vstack((x,xi))
                    y = np.vstack((y,yi))
        np.savetxt(namelist[0], x, '%1.8e')
        np.savetxt(namelist[1], y, '%1.8e')
        return

    def write_symmetric_train(self,x_train,y_train):
        sym_num = self.surface_symmetry_number 
        sym_x, sym_y = self.generate_symmetric_coords(x_train, y_train)
        np.savetxt('sym_x_train.dat', sym_x, '%1.8e')
        np.savetxt('sym_y_train.dat', sym_y, '%1.8e')
        return
    
    def write_minima_info(self, namelist):
        np.savetxt(namelist[0], self.minima_coords)
        np.savetxt(namelist[1], self.minima_E)
        return
