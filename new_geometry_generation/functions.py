import numpy.linalg as LA
import numpy as np
def get_min_max_distance(self, pos):
        min_dist = 100.0
        max_dist = 0.0
        valid = True
        for i in range(len(pos)):
            if i not in self.indices:
                for j in self.indices:
                    local_dist = LA.norm(pos[i, :]-pos[j, :])
                    if local_dist < min_dist:
                        min_dist = local_dist
                    if local_dist > max_dist:
                        max_dist = local_dist
        if min_dist < self.min_cutoff or max_dist > self.max_cutoff:
            valid = False
        return valid

    
def calculate_rigid_hessian(self, displacement_list, force_list):
    f = force_list
    x = displacement_list
    B = get_external_basis(self, self.atoms)
    f_sub = np.matmul(np.transpose(B), f)
    df_sub = np.zeros([self.ndim, self.ndim])
    dx = np.zeros([3 * len(self.indices), self.ndim])
    for i in range(self.ndim):
        dx[:, i] = x[:, 2 * i+1] - x[:, 2 * i]
        df_sub[:, i] = f_sub[:, 2 * i] - f_sub[:, 2 * i+1]
    dx_sub = np.matmul(np.transpose(B), dx)
    H = np.matmul(LA.inv(dx_sub), df_sub)
    H_sym = np.zeros(np.shape(H))
    for i in range(len(H)):
        for j in range(len(H)):
            H_sym[i,j] = np.average([H[i,j], H[j,i]])
    return H_sym

def get_external_basis(self, atoms):
    ads = atoms[self.indices].copy()
    com = ads.get_center_of_mass()
    pa = np.transpose(ads.get_moments_of_inertia(vectors=True)[1])
    B = np.zeros([3 * len(self.indices), self.ndim])
    ads_pos = ads.positions-com
    for i in range(len(self.indices)):
        B[3*i, 0] = 1
        B[3*i+1, 1] = 1
        B[3*i+2, 2] = 1
        if self.ndim > 3:
            B[3*i:3*i+3, 3] = np.cross(ads_pos[i, :], pa[:, 0])
            B[3*i:3*i+3, 4] = np.cross(ads_pos[i, :], pa[:, 1])
            if self.ndim > 5:
                B[3*i:3*i+3, 5] = np.cross(ads_pos[i, :], pa[:, 2])
    for i in range(self.ndim):
        B[:, i] *= 1 / LA.norm(np.copy(B[:, i]))
    return B

def bootstrap_points(self, atoms, coord):
    force_all = atoms.calc.results['forces']
    force = force_all[self.indices].reshape(-1)
    E = atoms.calc.results['energy']
    dx = 1e-2
    B = get_external_basis(self, atoms)
    f_sub = -1 * np.matmul(np.transpose(B), force)
    dE = dx * f_sub
    x = np.zeros([2*self.ndim, 6])
    y = np.zeros(2 * self.ndim)
    for i in range(self.ndim):
        x[2 * i, :] = coord
        x[2 * i+1, :] = coord
        x[2 * i, i] -= 0.5 * dx
        x[2 * i+1, i] += 0.5 * dx
        y[2 * i] = E - 0.5 * dE[i]
        y[2 * i+1] = E + 0.5 * dE[i]
    return x, y
