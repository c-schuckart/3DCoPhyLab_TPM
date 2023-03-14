import numpy as np
from numba import jit, njit, prange

import constants as const
import settings as sett


'''This function creates an equidistant mesh of either cylindrical or rectangular shape. 
The surface is marked as an array of six values, which each correspond to one of the sides, a "1" signaling it being
an exposed tile, while a "0" denotes a non surface element. 
Array entries 0 to 5 correspond to "z positive", "z negative", "y positive", "y negative", "x positive" and "x negative"'''
def create_equidistant_mesh(n_x, n_y, n_z, temperature_ini, dx, dy, dz):
    if sett.mesh_form == 1:
        a = n_x//2
        a_rad = (n_x - 2)//2
        b = n_y//2
        b_rad = (n_y - 2) // 2
        x, y = np.ogrid[:n_x, :n_y]
        mesh = np.zeros((n_z, n_y, n_x), dtype=np.float64)
        slice = np.zeros((n_y, n_x), dtype=np.float64)
        mask = ((x-a)/a_rad)**2 + ((y-b)/b_rad)**2 <= 1
        slice[mask] = temperature_ini
        for i in range(0, n_z-1):
            if i != 0:
                mesh[i] = slice
    elif sett.mesh_form == 0:
        mesh = np.full((n_z, n_y, n_x), temperature_ini)
        a, a_rad, b, b_rad = 0, 0, 0, 0
    else:
        raise NotImplementedError
    dx_arr = np.full((n_z, n_y, n_x), dx, dtype=np.float64)
    dy_arr = np.full((n_z, n_y, n_x), dy, dtype=np.float64)
    dz_arr = np.full((n_z, n_y, n_x), dz, dtype=np.float64)
    Dr = np.full((n_z, n_y, n_x, 6), np.array([dz, dz, dy, dy, dx, dx]), dtype=np.float64)
    return mesh, dx_arr, dy_arr, dz_arr, Dr, a, (a_rad-1), b, (b_rad-1)


@jit
def find_surface(n_x, n_y, n_z, limiter_x, limiter_y, limiter_z, mesh, surface, a, a_rad, b, b_rad):
    surface_elements = 0
    sample_holder = np.zeros((n_z, n_y, n_x), dtype=np.int32)
    for i in range(limiter_z, n_z):
        for j in range(limiter_y, n_y):
            for k in range(limiter_x, n_x):
                if mesh[i][j][k] != 0:
                    #Check if it is a surface in positive z direction
                    if mesh[i+1][j][k] == 0:
                        surface[i][j][k][0] = 1
                    # Check if it is a surface in negative z direction
                    if mesh[i-1][j][k] == 0:
                        surface[i][j][k][1] = 1
                    # Check if it is a surface in positive y direction
                    if mesh[i][j+1][k] == 0:
                        surface[i][j][k][2] = 1
                    # Check if it is a surface in negative y direction
                    if mesh[i][j-1][k] == 0:
                        surface[i][j][k][3] = 1
                    # Check if it is a surface in positive x direction
                    if mesh[i][j][k+1] == 0:
                        surface[i][j][k][4] = 1
                    # Check if it is a surface in negative x direction
                    if mesh[i][j][k-1] == 0:
                        surface[i][j][k][5] = 1
                    if ((k - a) / a_rad) ** 2 + ((j - b) / b_rad) ** 2 <= 1 and sum(surface[i][j][k] != 0) and i < n_z - 2:
                        surface_elements += 1
                    if (not (((k - a) / a_rad) ** 2 + ((j - b) / b_rad) ** 2 <= 1) and np.sum(surface[i][j][k]) != 0) or (np.sum(surface[i][j][k]) != 0 and i >= n_z-2):
                        sample_holder[i][j][k] = 1
    sample_holder, misplaced_voxels = fix_rim(n_x, n_y, limiter_x, limiter_y, a, a_rad, b, b_rad, sample_holder)
    surface_elements += len(misplaced_voxels)
    surface_reduced = reduce_surface(n_x, n_y, n_z, limiter_x, limiter_y, limiter_z, surface, np.zeros((surface_elements, 3), dtype=np.int32), a, a_rad, b, b_rad, misplaced_voxels)
    return surface, surface_reduced, sample_holder


@jit
def reduce_surface(n_x, n_y, n_z, limiter_x, limiter_y, limiter_z, surface, surface_reduced, a, a_rad, b, b_rad, misplaced_voxels):
    count = 0
    #z only runs up to n_z-2 to ignore the bottom plus the puffer layer
    for i in range(limiter_z, n_z-2):
        for j in range(limiter_y, n_y):
            for k in range(limiter_x, n_x):
                if sum(surface[i][j][k]) != 0 and ((k - a)/a_rad)**2 + ((j - b)/b_rad)**2 <= 1:
                    surface_reduced[count] = np.array([k, j, i], dtype=np.int32)
                    count += 1
    for each in misplaced_voxels:
        surface_reduced[count] = np.array([each[0], each[1], each[2]], dtype=np.int32)
        count += 1
    return surface_reduced


@jit
def fix_rim(n_x, n_y, limiter_x, limiter_y, a, a_rad, b, b_rad, array):
    erronous_voxels_nr = 0
    misplaced_voxels = np.zeros((n_x * n_y, 3), dtype=np.int32)
    for j in range(limiter_y, n_y):
        for k in range(limiter_x, n_x):
            if array[1][j][k] == 1 and max(array[1][j-1][k], array[1][j+1][k]) + max(array[1][j][k-1], array[1][j][k+1]) == 2 and ((k - a) / a_rad) ** 2 + ((j - b) / b_rad) ** 2 <= 1.002:
                array[1][j][k] = 0
                misplaced_voxels[erronous_voxels_nr] = np.array([k, j, 1], dtype=np.int32)
                erronous_voxels_nr += 1
    correct_len = 0
    for each in misplaced_voxels:
        if np.sum(each) > 0:
            correct_len += 1
    misplaced_voxels = misplaced_voxels[0:correct_len]
    return array, misplaced_voxels


@njit
def surrounding_checker(array, surface, n_x_lr, n_y_lr, n_z_lr, temperature):
    surrounding_surface = np.zeros((len(array) * 9, 3), dtype=np.int32)
    count = 0
    for each in array:
        for i in range(0, 6):
            if np.sum(surface[each[2] + n_z_lr[i]][each[1] + n_y_lr[i]][each[0] + n_x_lr[i]]) == 0 and temperature[each[2] + n_z_lr[i]][each[1] + n_y_lr[i]][each[0] + n_x_lr[i]] == 0:
                surrounding_surface[count] = np.array([each[0] + n_x_lr[i], each[1] + n_y_lr[i], each[2] + n_z_lr[i]], dtype=np.float32)
                count += 1
    for i in range(len(surrounding_surface)):
        for j in range(i+1-len(surrounding_surface)):
            if surrounding_surface[i][0] == surrounding_surface[j][0] and surrounding_surface[i][1] == surrounding_surface[j][1] and surrounding_surface[i][2] == surrounding_surface[j][2]:
                np.delete(surrounding_surface, j)
    return surrounding_surface[0:count]



#@jit
def DEBUG_print_3D_arrays(n_x, n_y, n_z, mesh):
    for i in range(0, n_z):
        print('Layer: ' + str(i + 1))
        for j in range(0, n_y):
            for k in range(0, n_x):
                print(mesh[i][j][k], end=' ')
            print('\n')


def one_d_test(n_x, n_y, n_z, dx, dy,  dz, direction):
    mesh = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    dx_arr = np.full((n_z, n_y, n_x), dx, dtype=np.float64)
    dy_arr = np.full((n_z, n_y, n_x), dy, dtype=np.float64)
    dz_arr = np.full((n_z, n_y, n_x), dz, dtype=np.float64)
    Lambda = np.full((n_z, n_y, n_x, 6), 0, dtype=np.float64)
    if direction == 'z':
        for i in range(2, n_z-2):
            mesh[i][n_y//2][n_x//2] = np.sin(np.pi * i * dz_arr[i][n_y//2][n_x//2]/np.sum(dz_arr[1:n_z-1][n_y//2][n_x//2]))
            Lambda[i][n_y//2][n_x//2][0] = const.lambda_constant
            Lambda[i][n_y // 2][n_x // 2][1] = const.lambda_constant
    if direction == 'y':
        for i in range(2, n_y-2):
            mesh[n_z//2][i][n_x//2] = np.sin(np.pi * i * dy_arr[n_z//2][i][n_x//2]/np.sum(dy_arr[n_z//2][1:n_y-1][n_x//2]))
            Lambda[n_z // 2][i][n_x//2][2] = const.lambda_constant
            Lambda[n_z // 2][i][n_x//2][3] = const.lambda_constant
    if direction == 'x':
        for i in range(2, n_x-2):
            mesh[n_z//2][n_y//2][i] = np.sin(np.pi * i * dx_arr[n_z//2][n_y//2][i]/np.sum(dx_arr[n_z//2][n_y//2][1:n_x-1]))
            Lambda[n_z // 2][n_y // 2][i][4] = const.lambda_constant
            Lambda[n_z // 2][n_y // 2][i][5] = const.lambda_constant
    Dr = np.full((n_z, n_y, n_x, 6), np.array([dz, dz, dy, dy, dx, dx]), dtype=np.float64)
    return mesh, dx_arr, dy_arr, dz_arr, Dr, Lambda
