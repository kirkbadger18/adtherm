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

def get_R_pa0(alpha):
    R_pa_0 = np.array(((np.cos(alpha), -np.sin(alpha), 0),
                        (np.sin(alpha), np.cos(alpha), 0),
                        (0, 0, 1)))
    return R_pa_0
def get_R_pa1(gamma):
    R_pa_1 = np.array(((np.cos(gamma), 0, np.sin(gamma)),
                      (0, 1, 0),
                      (-np.sin(gamma), 0, np.cos(gamma))))
    return R_pa_1

def get_R_pa2(beta):
    R_pa_2 = np.array(((1, 0, 0),
                      (0, np.cos(beta), -np.sin(beta)),
                      (0, np.sin(beta), np.cos(beta))))
    return R_pa_2

def get_dRdalpha(alpha, beta, gamma):
    R0 = np.array(((-np.sin(alpha), -np.cos(alpha), 0),
                        (np.cos(alpha), -np.sin(alpha), 0),
                        (0, 0, 0)))

    R1 = get_R_pa1(gamma)
    R2 = get_R_pa2(beta)
    dRdalpha = np.matmul(np.matmul(R1,R2), R0)
    return dRdalpha

def get_dRdbeta(alpha, beta, gamma):
    R0 = get_R_pa0(alpha)
    R1 = get_R_pa1(gamma)
    R2 = np.array(((0, 0, 0),
                      (0, -np.sin(beta), -np.cos(beta)),
                      (0, np.cos(beta), -np.sin(beta))))
    dRdbeta = np.matmul(np.matmul(R1,R2), R0)
    return dRdbeta

def get_dRdgamma(alpha, beta, gamma):
    R0 = get_R_pa0(alpha)
    R1 = np.array(((-np.sin(gamma), 0, np.cos(gamma)),
                      (0, 0, 0),
                      (-np.cos(gamma), 0, -np.sin(gamma))))
    R2 = get_R_pa2(beta)
    dRdgamma = np.matmul(np.matmul(R1,R2), R0)
    return dRdgamma

def get_R(alpha, beta, gamma):
    R0 = get_R_pa0(alpha)
    R1 = get_R_pa1(gamma)
    R2 = get_R_pa2(beta)
    R = np.matmul(np.matmul(R1,R2), R0)
    return R

def project_to_rigid_hessian(AdTherm, H, rot_coords):
    B = get_external_basis(AdTherm, rot_coords)
    H_sub = np.matmul(B.T,np.matmul(H,B))
    print(H_sub.round(3))
    print('eig are: ', LA.eigh(H_sub)[0])
    return H_sub

def get_external_basis(AdTherm, rot_coords):
    ads = AdTherm.adsorbates[0].copy()
    com = ads.get_center_of_mass()
    pa = ads.get_moments_of_inertia(vectors=True)[1].T
    com_pos = ads.positions - com
    pa_pos = np.matmul(pa.T,com_pos.T).T
    alpha, beta, gamma = rot_coords
    B = np.zeros([3 * len(AdTherm.indices), AdTherm.ndim])
    dRdalpha = get_dRdalpha(alpha, beta, gamma)
    dRdbeta = get_dRdbeta(alpha, beta, gamma)
    dRdgamma = get_dRdgamma(alpha, beta, gamma)
    dxdalpha = np.matmul(pa,np.matmul(dRdalpha,pa_pos.T)).T
    dxdbeta = np.matmul(pa,np.matmul(dRdbeta,pa_pos.T)).T
    dxdgamma = np.matmul(pa,np.matmul(dRdgamma,pa_pos.T)).T
    for i in range(len(AdTherm.indices)):
        B[3*i, 0] = 1
        B[3*i+1, 1] = 1
        B[3*i+2, 2] = 1
    if AdTherm.ndim > 3:
        B[:, 3] = dxdalpha.reshape(-1)
        B[:, 4] = dxdbeta.reshape(-1)
        if AdTherm.ndim > 5:
            B[:, 5] = dxdgamma.reshape(-1)
    return B

def bootstrap_points(AdTherm, atoms, coord):
    force_all = atoms.calc.results['forces']
    force = force_all[AdTherm.indices].reshape(-1)
    E = atoms.calc.results['energy']
    dx = 1e-2
    rot_coords = coord[3::]
    B = get_external_basis(AdTherm, rot_coords)
    g_sub = -1 * np.matmul(np.transpose(B), force)
    dE = dx * g_sub
    x = np.zeros([2*AdTherm.ndim, AdTherm.ndim])
    y = np.zeros([2 * AdTherm.ndim, 1])
    for i in range(AdTherm.ndim):
        x[2 * i, :] = coord
        x[2 * i+1, :] = coord
        x[2 * i, i] -= 0.5 * dx
        x[2 * i+1, i] += 0.5 * dx
        y[2 * i] = E - 0.5 * dE[i]
        y[2 * i+1] = E + 0.5 * dE[i]
    return x, y

def get_referenced_principle_axis(AdTherm,ads):

    ref_ads = AdTherm.adsorbates[0].copy()
    ref_com = AdTherm.coms[0,:]
    ref_pa =  ref_ads.get_moments_of_inertia(vectors=True)[1].T
    ref_pos = ref_ads.positions-ref_com
    com = ads.get_center_of_mass()
    pa = ads.get_moments_of_inertia(vectors=True)[1].T
    pos = ads.positions - com

    max_idx = np.zeros(3)
    max_dot = np.zeros(3)
    for i in range(3):
        for j in range(np.shape(pos)[0]):
            dot = np.abs(np.dot(pos[j,:],ref_pa[:,i]))
            if dot > max_dot[i]:
                max_dot[i] = dot
                max_idx[i] = j

    for i in range(3):
        idx = int(max_idx[i])
        sign = np.sign(np.dot(pa[:,i], pos[idx,:]))
        ref_sign = np.sign(np.dot(ref_pa[:,i], ref_pos[idx,:]))
        if sign != ref_sign:
            pa[:,i] *= -1

    x = np.array([1,0,0])
    y = np.array([0,1,0])
    z = np.array([0,0,1])

    if AdTherm.N_atoms_in_adsorbate == 2:
        if np.dot(ref_pa[:,0],x) > 1e-4 and np.dot(pa[:,0],x) > 1e-4:
            ref_pa[:,1] = np.cross(ref_pa[:,0],x)
            pa[:,1] = np.cross(pa[:,0],x)
        elif np.dot(ref_pa[:,0],x) > 1e-4 and np.dot(pa[:,0],y) > 1e-4:
            ref_pa[:,1] = np.cross(ref_pa[:,0],y)
            pa[:,1] = np.cross(pa[:,0],y)
        elif np.dot(ref_pa[:,0],x) > 1e-4 and np.dot(pa[:,0],z) > 1e-4:
            ref_pa[:,1] = np.cross(ref_pa[:,0],z)
            pa[:,1] = np.cross(pa[:,0],z)
        pa[:,1] =  (1 / LA.norm(pa[:,1])) * pa[:,1]
    elif AdTherm.N_atoms_in_adsorbate == 3:
        ref_pa[:,2] = np.cross(ref_pa[:,0],ref_pa[:,1])
        pa[:,2] = np.cross(pa[:,0],pa[:,1])

    return np.matmul(ref_pa.T, pa)

def map_rotation_to_min0(AdTherm, atoms):
    
    ads = atoms[AdTherm.indices].copy()
    R = get_referenced_principle_axis(AdTherm,ads)
    A_solve = R.flatten()

    def f(x):
        alpha, beta, gamma = x
        #R_pa_0 = np.array(((np.cos(x0), -np.sin(x0), 0), # rotation about first axis
        #            (np.sin(x0), np.cos(x0), 0),
        #            (0, 0, 1)))
        #R_pa_1 = np.array(((np.cos(x2), 0, np.sin(x2)),  #rotation about second axis
        #              (0, 1, 0),
        #              (-np.sin(x2), 0, np.cos(x2))))
        #R_pa_2 = np.array(((1, 0, 0),
        #              (0, np.cos(x1), -np.sin(x1)), # rotation about third axis
        #              (0, np.sin(x1), np.cos(x1))))
        #A = np.dot(np.dot(R_pa_1,R_pa_2), R_pa_0)
        #'''choosing to rotate about longest axis second to minimize odds of gimble lock'''
        R = get_R(alpha, beta, gamma)
        A = R.flatten()
        return A

    def system(x,b=A_solve):
        return(f(x)-b)

    if AdTherm.ndim == 6:
        max_error = 10
        while max_error > 1e-8:
            guess = np.random.rand(3)
            guess[0] = guess[0] * 2 * np.pi - np.pi
            guess[1] = guess[1] * np.pi - np.pi / 2
            guess[2] = guess[2] * 2 * np.pi - np.pi
            x=least_squares(system,
                        guess,
                        method = 'dogbox',
                        bounds=([-np.pi, -np.pi/2, -np.pi],
                                [np.pi, np.pi/2, np.pi])                       
                        
                        )
            max_error = np.max(np.abs(f(x.x) - A_solve))

    elif AdTherm.ndim == 5:
        ''' maybe needs to be writen like the case above in the future'''
        x=least_squares(system,
                        np.asarray((0,0,0)),
                        bounds=([-np.pi, -np.pi, -1e-8],
                                [np.pi, np.pi, 1e-8]))
 
    rot_coord = x.x[0:AdTherm.ndim-3]
    return rot_coord    
