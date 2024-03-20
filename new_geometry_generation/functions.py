import numpy.linalg as LA
import numpy as np
from scipy.optimize import least_squares
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

def project_to_rigid_hessian(self, H, atoms):
    B = get_external_basis(self, atoms)
    H_sub = np.matmul(B.T,np.matmul(H,B))
    return H_sub

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
            B[3*i:3*i+3, 3] = np.cross(ads_pos[i, :], pa[:, 2])
            B[3*i:3*i+3, 4] = np.cross(ads_pos[i, :], pa[:, 1])
            if self.ndim > 5:
                B[3*i:3*i+3, 5] = np.cross(ads_pos[i, :], pa[:, 0])
    for i in range(self.ndim):
        B[:, i] *= 1 / LA.norm(np.copy(B[:, i]))
    q, r = LA.qr(B)
    return q

def bootstrap_points(self, atoms, coord):
    force_all = atoms.calc.results['forces']
    force = force_all[self.indices].reshape(-1)
    E = atoms.calc.results['energy']
    dx = 1e-2
    B = get_external_basis(self, atoms)
    f_sub = -1 * np.matmul(np.transpose(B), force)
    dE = dx * f_sub
    x = np.zeros([2*self.ndim, 6])
    y = np.zeros([2 * self.ndim, 1])
    for i in range(self.ndim):
        x[2 * i, :] = coord
        x[2 * i+1, :] = coord
        x[2 * i, i] -= 0.5 * dx
        x[2 * i+1, i] += 0.5 * dx
        y[2 * i] = E - 0.5 * dE[i]
        y[2 * i+1] = E + 0.5 * dE[i]
    return x, y

def map_coords_to_min0(self, atoms):
    coord = np.zeros(self.ndim)
    ref_ads = self.adsorbates[0].copy()
    ref_com = self.coms[0,:]
    ref_pa =  np.transpose(ref_ads.get_moments_of_inertia(vectors=True)[1])
    ref_pos = ref_ads.positions-ref_com
    ads = atoms[self.indices].copy()
    com = ads.get_center_of_mass()
    pa = np.transpose(ads.get_moments_of_inertia(vectors=True)[1])
    pos = ads.positions - com

    for i in range(len(ads)):
        ref_overlap = np.matmul(ref_pos[i,:],ref_pa)
        overlap =  np.matmul(pos[i,:],pa) 
        if np.min(np.abs(overlap)) >= 0.01:
            break

    for i in range(3):
        ref_sign = np.sign(ref_overlap[i])
        sign = np.sign(overlap[i])
        if ref_sign != sign:
            pa[:,i] *= -1
    A_solve = np.matmul(ref_pa.T, pa).flatten()
    
    def f(x):
        x0, x1, x2 = x
        D = np.array(((np.cos(x2), -np.sin(x2), 0),
                    (np.sin(x2), np.cos(x2), 0),
                    (0, 0, 1)))
        C = np.array(((np.cos(x1), 0, np.sin(x1)),
                      (0, 1, 0),
                      (-np.sin(x1), 0, np.cos(x1))))
        B = np.array(((1, 0, 0),
                      (0, np.cos(x0), -np.sin(x0)),
                      (0, np.sin(x0), np.cos(x0))))
        A = np.dot(np.dot(B,C), D)
        A = A.flatten()
        return A

    def system(x,b=A_solve):
        return(f(x)-b)

    if self.ndim == 6:
        x=least_squares(system,
                        np.asarray((0,0,0)),
                        bounds=([-np.pi, -np.pi/2, -np.pi],
                                [np.pi, np.pi/2, np.inf]))
    elif self.ndim == 5:
        x=least_squares(system,
                        np.asarray((0,0,0)),
                        bounds=([-np.pi, -np.pi, 0],
                                [np.pi, np.pi, 0]))

    coord[0:3] = com
    coord[3:self.ndim] = x.x
    return coord    
