import numpy as np
from functions import * 
from sobol_seq import i4_sobol_generate

def coord_generate(AdTherm, method, N_values, minima_index=0):
    dft_jobs = []
    k = minima_index
    coords = np.zeros([N_values, AdTherm.ndim])
    Iter = 0
    sobol_n = 0
    while Iter < N_values:
        coord = np.zeros(AdTherm.ndim)
        if method == 'gauss':
            #gaussmean = np.zeros(AdTherm.ndim)
            gaussmean =  AdTherm.minima_coords[k,:]
            #gaussmean[0:3] = AdTherm.coms[k,:]
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
                coord = i4_sobol_generate(AdTherm.ndim, 1, sobol_n+1)[0, :]
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

        valid, location = check_coord(AdTherm, coord)
        if valid: # and location == 'inside':
            atoms = manipulate_atoms(AdTherm, coord, k)
            valid = get_min_max_distance(AdTherm, atoms.positions) 

        if valid and location == 'outside':
            coord, location = move_inside(AdTherm, coord)

        if valid:
            if method == 'gauss' and minima_index > 0 and AdTherm.rotate:
                print('before: ', coord[3])
                coord[3::] = map_rotation_to_min0(AdTherm, atoms) 
                print('after: ',coord[3])
            coords[Iter, :] = coord
            Iter += 1
            dft_jobs.append(atoms)

    return dft_jobs, coords

def check_coord(AdTherm, coord):
    uc_x = AdTherm.unit_cell_x / 3.0
    uc_y = AdTherm.unit_cell_y / 3.0
    y_ub = uc_y
    y_lb = 0.0
    x_ub = uc_x + coord[1] * (1. / np.sqrt(3))
    x_lb = coord[1] * (1. / np.sqrt(3))
    z_ub = AdTherm.z_high
    z_lb = AdTherm.z_low
    valid = True
    location = 'inside'
    if coord[2] > z_ub or coord[2] < z_lb:
        valid = False
    if coord[0] > x_ub or coord[0] < x_lb:
        location = 'outside'
    if coord[1] > y_ub or coord[1] < y_lb:
        location = 'outside'
    #if AdTherm.ndim >= 5:    
    #    if coord[3] > np.pi or coord[3] < -np.pi:
    #        location = 'outside'
    #    if coord[4] > 0.5 * np.pi or coord[4] < -0.5 * np.pi:
    #        location = 'outside'
    #if AdTherm.ndim == 6:
    #    if coord[5] > np.pi or coord[5] < -np.pi:
    #        location = 'outside'
    return valid, location

def move_inside(AdTherm, coord):
    uc_x = AdTherm.unit_cell_x / 3.0
    uc_y = AdTherm.unit_cell_y / 3.0
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
    #if AdTherm.ndim >= 5:
    #    while coord[3] > np.pi or coord[3] < -np.pi:
    #        if coord[3] > np.pi:
    #            coord[3] -= 2 * np.pi
    #        if coord[3] < -np.pi:
    #            coord[3] += 2 * np.pi
    #    while coord[4] > 0.5 * np.pi or coord[4] < -0.5 * np.pi:
    #        if coord[4] > 0.5 * np.pi:
    #            coord[4] -= np.pi
    #            sign = np.sign(coord[3])
    #            coord[3] = sign * (np.pi - np.abs(coord[3]))
    #        if coord[4] < -0.5 * np.pi:
    #            coord[4] += np.pi
    #            sign = np.sign(coord[3])
    #            coord[3] = sign * (np.pi - np.abs(coord[3]))
    #if AdTherm.ndim == 6:
    #    while coord[5] > np.pi or coord[5] < -np.pi:
    #        if coord[5] > np.pi:
    #            coord[5] -= 2 * np.pi
    #        if coord[5] < -np.pi:
    #            coord[5] += 2 * np.pi
    valid, location = check_coord(AdTherm, coord)
    if not valid or location == 'outside':
        raise Exception("move inside function not working")
    else:
        return coord, location

def manipulate_atoms(AdTherm, coord, k):
    ''' Need to fix so that rotation to position by a,b, happens'''
    conv = 180 / np.pi
    pa = AdTherm.adsorbates[0].get_moments_of_inertia(vectors=True)[1].T
    atoms = AdTherm.minima[0].copy()
    adsorbate = AdTherm.adsorbates[k].copy()
    com_pos = adsorbate.positions - AdTherm.coms[k,:]
    pa_pos = np.matmul(pa.T,com_pos.T).T 
    if AdTherm.rotate:
        alpha, beta = coord[3:5]
        alpha0, beta0 = AdTherm.minima_coords[0,3:5]
        gamma0 = 0
        gamma = 0
        if AdTherm.ndim == 6:
            gamma = coord[5]
            gamma = AdTherm.minima_coords[0,5]
        R = get_R(alpha, beta, gamma)
        R0 = get_R(alpha0, beta0, gamma0)
        invR0 = LA.inv(R0)
        new_pa_pos = np.matmul(R, np.matmul(invR0, pa_pos.T)).T
        com_pos = np.matmul(pa,new_pa_pos.T).T

    dx = coord[0:3]
    new_pos = com_pos + dx
    atoms.positions[AdTherm.indices] = new_pos
    return atoms
