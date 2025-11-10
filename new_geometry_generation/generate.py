import numpy as np
from functions import * 
from sobol_seq import i4_sobol_generate

def coord_generate(AdTherm, method, N_values, minima_index=0, seed=1):
    dft_jobs = []
    k = minima_index
    coords = np.zeros([N_values, AdTherm.ndim])
    Iter = 0
    sobol_n = 0
    while Iter < N_values:
        print(Iter)
        coord = np.zeros(AdTherm.ndim)
        if method == 'gauss':
            #gaussmean = np.zeros(AdTherm.ndim)
            gaussmean =  AdTherm.minima_coords[k,:]
            gaussmean[0:3] = AdTherm.coms[k,:]
            hess = AdTherm.rigid_hessians[k]
            gausscov = AdTherm.scale_gauss * LA.inv(hess)
            rand = np.random.multivariate_normal(
                    gaussmean,
                    gausscov,
                    size=1,
                    check_valid='warn')
            coord = rand[0]
        if method == 'random' or method == 'sobol':
            if method == 'sobol':
                coord = i4_sobol_generate(AdTherm.ndim, 1, sobol_n+1+seed)[0, :]
                sobol_n += 1
            if method == 'random':
                coord = np.random.uniform(0, 1, size=AdTherm.ndim)

            coord[1] *= AdTherm.unit_cell_y / 3
            coord[0] *= AdTherm.unit_cell_x / 3
            coord[0] += coord[1] / np.sqrt(3)
            coord[2] *= AdTherm.z_high - AdTherm.z_low
            coord[2] += AdTherm.z_low
            if AdTherm.ndim >= 5:
                coord[3:AdTherm.ndim] *= 2 * np.pi
                coord[3:AdTherm.ndim] -= np.pi
                coord[4] *= 0.5

        valid, xy_location = check_xy_coord(AdTherm, coord)
        if valid: # and location == 'inside':
            atoms = manipulate_atoms(AdTherm, coord, k)
            valid = get_min_max_distance(AdTherm, atoms.positions)

        if valid and xy_location == 'outside':
            coord, xy_location = move_xy_inside(AdTherm, coord)

        if valid:
            if method == 'gauss' and minima_index > 0 and AdTherm.rotate:
                coord[3::] = map_rotation_to_min0(AdTherm, atoms) 
            coords[Iter, :] = coord
            Iter += 1
            dft_jobs.append(atoms)

    return dft_jobs, coords

def get_ab_from_xy(AdTherm, xy):
    len_x = AdTherm.unit_cell_x / 3.0
    len_y = AdTherm.unit_cell_y / 3.0
    vx = np.array([len_x,0])
    vy = np.array([0.5 * len_y, np.sqrt(3) * len_y / 2])
    B = np.zeros([2,2])
    B[:,0] = vx
    B[:,1] = vy
    xy = np.array([xy[0],xy[1]])
    invB = LA.inv(B)
    ab = np.matmul(invB,xy)
    return ab

def get_xy_from_ab(AdTherm, ab):
    len_x = AdTherm.unit_cell_x / 3.0
    len_y = AdTherm.unit_cell_y / 3.0
    vx = np.array([len_x,0])
    vy = np.array([0.5 * len_y, np.sqrt(3) * len_y / 2])
    B = np.zeros([2,2])
    B[:,0] = vx
    B[:,1] = vy
    xy = np.matmul(B,ab)
    return xy

def check_xy_coord(AdTherm, coord):
    ab = get_ab_from_xy(AdTherm, coord[0:2])
    z_ub = AdTherm.z_high
    z_lb = AdTherm.z_low
    valid = True
    location = 'inside'
    if coord[2] > z_ub or coord[2] < z_lb:
        valid = False
    if ab[0] > 1 or ab[0] < 0:
        location = 'outside'
    if ab[1] > 1 or ab[1] < 0:
        location = 'outside'
    return valid, location

def move_xy_inside(AdTherm, coord):
    ab = get_ab_from_xy(AdTherm,coord[0:2])
    while ab[0] > 1 or ab[0] < 0:
        sign = np.sign(ab[0])
        ab[0] -= sign * 1
    while ab[1] > 1 or ab[1] < 0:
        sign = np.sign(ab[1])
        ab[1] -= sign * 1
    xy = get_xy_from_ab(AdTherm, ab)
    coord[0:2] = xy[0:2]
    valid_z, xy_location = check_xy_coord(AdTherm, coord)
    if not valid_z or xy_location == 'outside':
        raise Exception("move inside function not working")
    else:
        return coord, xy_location

def manipulate_atoms(AdTherm, coord, k):
    conv = 180 / np.pi
    pa = AdTherm.adsorbates[0].get_moments_of_inertia(vectors=True)[1].T  
    atoms = AdTherm.minima[0].copy()
    adsorbate = AdTherm.adsorbates[k].copy()
    ref_pa, pa = get_referenced_principle_axis(AdTherm, AdTherm.adsorbates[0]) 
    com_pos = adsorbate.positions - AdTherm.coms[k,:]
    pa_pos = np.matmul(pa.T,com_pos.T).T 
    if AdTherm.rotate:
        if AdTherm.ndim == 6:
            alpha, beta, gamma = coord[3:6]
            alpha0, beta0, gamma0 = AdTherm.minima_coords[k,3:6]
        if AdTherm.ndim == 5:
            beta, gamma = coord[3:5]
            beta0, gamma0 = AdTherm.minima_coords[k,3:5]
            alpha, alpha0 = 0, 0

        R = get_R(alpha, beta, gamma)
        R0 = get_R(alpha0, beta0, gamma0)
        invR0 = LA.inv(R0)
        new_pa_pos = np.matmul(R,np.matmul(invR0, pa_pos.T)).T
        com_pos = np.matmul(pa,new_pa_pos.T).T
    dx = coord[0:3]
    new_pos = com_pos + dx
    atoms.positions[AdTherm.indices] = new_pos
    return atoms
