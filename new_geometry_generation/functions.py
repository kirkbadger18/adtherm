import numpy.linalg as LA
import numpy as np
from scipy.optimize import least_squares


def get_min_max_distance(AdTherm, pos):
        min_dist = 100.0
        max_dist = 0.0
        valid = True
        for i in range(len(pos)):
            if i not in AdTherm.indices:
                for j in AdTherm.indices:
                    local_dist = LA.norm(pos[i, :]-pos[j, :])
                    if local_dist < min_dist:
                        min_dist = local_dist
                    if local_dist > max_dist:
                        max_dist = local_dist
        too_close = min_dist < AdTherm.min_atomic_distance
        too_far = max_dist > AdTherm.max_atomic_distance
        if too_close or too_far:
            valid = False
        return valid

def project_to_rigid_hessian(AdTherm, H, atoms):
    B = get_external_basis(AdTherm, atoms)
    H_sub = np.matmul(B.T,np.matmul(H,B))
    print(H_sub.round(3))
    return H_sub

def get_external_basis(AdTherm, atoms):
    ads = atoms[AdTherm.indices].copy()
    com = ads.get_center_of_mass()
    pa = ads.get_moments_of_inertia(vectors=True)[1].T
    B = np.zeros([3 * len(AdTherm.indices), AdTherm.ndim])
    ads_pos = ads.positions - com
    for i in range(len(AdTherm.indices)):
        B[3*i, 0] = 1
        B[3*i+1, 1] = 1
        B[3*i+2, 2] = 1
        if AdTherm.ndim > 3:
            B[3*i:3*i+3, 3] = np.cross(ads_pos[i, :], pa[:, 2])
            B[3*i:3*i+3, 4] = np.cross(ads_pos[i, :], pa[:, 1])
            if AdTherm.ndim > 5:
                B[3*i:3*i+3, 5] = np.cross(ads_pos[i, :], pa[:, 0])
    return B

def bootstrap_points(AdTherm, atoms, coord):
    force_all = atoms.calc.results['forces']
    force = force_all[AdTherm.indices].reshape(-1)
    E = atoms.calc.results['energy']
    dx = 1e-2
    B = get_external_basis(AdTherm, atoms)
    f_sub = -1 * np.matmul(np.transpose(B), force)
    dE = dx * f_sub
    x = np.zeros([2*AdTherm.ndim, AdTherm.ndim])
    y = np.zeros([2 * AdTherm.ndim, 1])
    for i in range(AdTherm.ndim):
        x[2 * i, :] = coord
        x[2 * i+1, :] = coord
        #x[2 * i, i] -= 0.5 * dx
        x[2 * i+1, i] += 0.5 * dx
        y[2 * i] = E - 0.5 * dE[i]
        y[2 * i+1] = E + 0.5 * dE[i]
    return x, y

def map_rotation_to_min0(AdTherm, atoms):
    ref_ads = AdTherm.adsorbates[0].copy()
    ref_com = AdTherm.coms[0,:]
    ref_pa =  np.transpose(ref_ads.get_moments_of_inertia(vectors=True)[1])
    ref_pos = ref_ads.positions-ref_com
    ads = atoms[AdTherm.indices].copy()
    com = ads.get_center_of_mass()
    pa = np.transpose(ads.get_moments_of_inertia(vectors=True)[1])
    pos = ads.positions - com

    for i in range(len(ads)):
        ref_overlap = np.matmul(ref_pos[i,:],ref_pa)
        overlap =  np.matmul(pos[i,:],pa) 
        #print('overlap: ',overlap)
        if np.min(np.abs(overlap)) >= 0.01:
            break

    for i in range(3):
        ref_sign = np.sign(ref_overlap[i])
        sign = np.sign(overlap[i])
        if ref_sign != sign and overlap[i] >= 0.01:
            pa[:,i] *= -1
    
    cross_ref = np.cross(ref_pa[:,0],ref_pa[:,1])
    ref_sign = np.sign(np.dot(cross_ref,ref_pa[:,2]))
    cross = np.cross(pa[:,0],pa[:,1])
    sign = np.sign(np.dot(cross,pa[:,2]))
    if sign != ref_sign:
        pa[:,2] *= -1

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

    if AdTherm.ndim == 6:
        x=least_squares(system,
                        np.asarray((0,0,0)),
                        bounds=([-np.pi, -np.pi/2, -np.pi],
                                [np.pi, np.pi/2, np.inf]))
    elif AdTherm.ndim == 5:
        x=least_squares(system,
                        np.asarray((0,0,0)),
                        bounds=([-np.pi, -np.pi, -1e-8],
                                [np.pi, np.pi, 1e-8]))

    rot_coord = x.x[0:AdTherm.ndim-3]
    return rot_coord    
