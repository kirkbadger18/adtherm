import numpy as np
from functions import * 
from sobol_seq import i4_sobol_generate

def coord_generate(self, method, N_values, minima_index=0):
    dft_jobs = []
    k = minima_index
    coords = np.zeros([N_values, self.ndim])
    Iter = 0
    sobol_n = 0
    while Iter < N_values:
        coord = np.zeros(self.ndim)

        if method == 'gauss':
            gaussmean = np.zeros(self.ndim)
            gaussmean[0:3] = self.coms[k,:]
            hess = self.rigid_hessians[k]
            gausscov = self.scale_gauss * LA.inv(hess)
            rand = np.random.multivariate_normal(
                    gaussmean,
                    gausscov,
                    size=1,
                    check_valid='warn')
            coord = rand[0]
        if method == 'random' or method == 'sobol':
            if method == 'sobol':
                coord = i4_sobol_generate(self.ndim, 1, sobol_n+1)[0, :]
                sobol_n += 1
            if method == 'random':
                coord = np.random.uniform(0, 1, size=self.ndim)

            coord[1] *= self.uc_y / 3
            coord[0] *= self.uc_x / 3
            coord[0] += coord[1] / np.sqrt(3)
            coord[2] *= self.z_high - self.z_low
            coord[2] += self.z_low
            if self.ndim >= 5:
                coord[3:self.ndim] *= 2 * np.pi
                coord[3:self.ndim] -= np.pi
                coord[4] *= 0.5

        valid, location = check_coord(self, coord)
        if valid and location == 'outside':
            coord, location = move_inside(self, coord)
        if valid and location == 'inside':
            atoms = manipulate_atoms(self, coord, k)
            valid = get_min_max_distance(self, atoms.positions) 

        
        if valid:
            if method == 'gauss' and minima_index > 0 and self.rotate:
                coord = map_coords_to_min0(self, atoms) 
            coords[Iter, :] = coord
            Iter += 1
            dft_jobs.append(atoms)

    return dft_jobs, coords

def check_coord(self, coord):
    uc_x = self.uc_x / 3.0
    uc_y = self.uc_y / 3.0
    y_ub = uc_y
    y_lb = 0.0
    x_ub = uc_x + coord[1] * (1. / np.sqrt(3))
    x_lb = coord[1] * (1. / np.sqrt(3))
    z_ub = self.z_high
    z_lb = self.z_low
    valid = True
    location = 'inside'
    if coord[2] > z_ub or coord[2] < z_lb:
        valid = False
    if coord[0] > x_ub or coord[0] < x_lb:
        location = 'outside'
    if coord[1] > y_ub or coord[1] < y_lb:
        location = 'outside'
    if self.ndim >= 5:    
        if coord[3] > np.pi or coord[3] < -np.pi:
            location = 'outside'
        if coord[4] > 0.5 * np.pi or coord[4] < -0.5 * np.pi:
            location = 'outside'
    if self.ndim == 6:
        if coord[5] > np.pi or coord[5] < -np.pi:
            location = 'outside'
    return valid, location

def move_inside(self, coord):
    uc_x = self.uc_x / 3.0
    uc_y = self.uc_y / 3.0
    y_ub = uc_y
    y_lb = 0.0
    x_ub = uc_x + coord[1] * (1. / np.sqrt(3))
    x_lb = coord[1] * (1. / np.sqrt(3))

    while coord[1] > y_ub or coord[1] < y_lb:
        if coord[1] > y_ub:
            coord[1] -= uc_y
            coord[0] -= uc_y * (1. / np.sqrt(3))
        elif coord[1] < y_lb:
            coord[1] += uc_y
            coord[0] += uc_y * (1. / np.sqrt(3))
        x_ub = uc_x + coord[1] * (1. / np.sqrt(3))
        x_lb = coord[1] * (1. / np.sqrt(3))
    while (coord[0] > x_ub or coord[0] < x_lb):
        if coord[0] > x_ub:
            coord[0] -= uc_x
        elif coord[0] < x_lb:
            coord[0] += uc_x
    if self.ndim >= 5:
        while coord[3] > np.pi or coord[3] < -np.pi:
            if coord[3] > np.pi:
                coord[3] -= 2 * np.pi
            if coord[3] < -np.pi:
                coord[3] += 2 * np.pi
        while coord[4] > 0.5 * np.pi or coord[4] < -0.5 * np.pi:
            if coord[4] > 0.5 * np.pi:
                coord[4] -= np.pi
                sign = np.sign(coord[3])
                coord[3] = sign * (np.pi - np.abs(coord[3]))
            if coord[4] < -0.5 * np.pi:
                coord[4] += np.pi
                sign = np.sign(coord[3])
                coord[3] = sign * (np.pi - np.abs(coord[3]))
    if self.ndim == 6:
        while coord[5] > np.pi or coord[5] < -np.pi:
            if coord[5] > np.pi:
                coord[5] -= 2 * np.pi
            if coord[5] < -np.pi:
                coord[5] += 2 * np.pi
    valid, location = check_coord(self, coord)
    if not valid or location == 'outside':
        raise Exception("move inside function not working")
    else:
        return coord, location

def manipulate_atoms(self, coord, k):
    conv = 180 / np.pi
    pa = np.transpose(self.adsorbates[k].get_moments_of_inertia(
            vectors=True)[1])
    atoms = self.minima[k].copy()
    adsorbate = self.adsorbates[k].copy()
    if self.rotate:
        adsorbate.rotate(conv * coord[3], pa[:, 2], 'COM')
        adsorbate.rotate(conv * coord[4], pa[:, 1], 'COM')
        if self.ndim == 6:
            adsorbate.rotate(conv * coord[5], pa[:, 0], 'COM')
    adsorbate.translate(coord[0:3] - self.coms[k])
    atoms.positions[self.indices] = adsorbate.positions
    return atoms
