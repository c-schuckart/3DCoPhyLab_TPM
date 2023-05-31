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
    dz_arr[1] = np.full((n_y, n_x,), dz / 2, dtype=np.float64)
    Dr = np.full((n_z, n_y, n_x, 6), np.array([dz, dz, dy, dy, dx, dx]), dtype=np.float64)
    return mesh, dx_arr, dy_arr, dz_arr, Dr, a, (a_rad-1), b, (b_rad-1)


def create_equidistant_mesh_gradient(n_x, n_y, n_z, temperature_ini, dx, dy, dz):
    if sett.mesh_form == 1:
        a = n_x//2
        a_rad = (n_x - 2)//2
        b = n_y//2
        b_rad = (n_y - 2) // 2
        x, y = np.ogrid[:n_x, :n_y]
        mesh = np.zeros((n_z, n_y, n_x), dtype=np.float64)
        slice = np.zeros((n_y, n_x), dtype=np.float64)
        mask = ((x-a)/a_rad)**2 + ((y-b)/b_rad)**2 <= 1
        slice[mask] = 1
        for i in range(0, n_z-1):
            if i != 0:
                #mesh[i] = slice * (100 * i/n_z-1 + temperature_ini)
                mesh[i] = slice * (100 * (n_z-1-i)/n_z-1 + temperature_ini)
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


@njit
def find_surface(n_x, n_y, n_z, limiter_x_start, limiter_y_start, limiter_z_start, limiter_x_end, limiter_y_end, limiter_z_end, mesh, surface, a, a_rad, b, b_rad, initiation):
    surface_elements = 0
    sample_holder = np.zeros((n_z, n_y, n_x), dtype=np.int32)
    for i in range(limiter_z_start, limiter_z_end):
        for j in range(limiter_y_start, limiter_y_end):
            for k in range(limiter_x_start, limiter_x_end):
                surface[i][j][k] = np.zeros(6, dtype=np.int32)
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
    if initiation:
        sample_holder, misplaced_voxels = fix_rim(n_x, n_y, limiter_x_start, limiter_y_start, a, a_rad, b, b_rad, sample_holder)
    else:
        misplaced_voxels = np.empty((0, 0), dtype=np.int32)
    surface_elements += len(misplaced_voxels)
    if initiation:
        surface_reduced = reduce_surface(n_x, n_y, n_z, limiter_x_start, limiter_y_start, limiter_z_start, limiter_x_end, limiter_y_end, limiter_z_end, surface, np.zeros((surface_elements, 3), dtype=np.int32), a, a_rad, b, b_rad, misplaced_voxels)
    else:
        surface_reduced = reduce_surface(n_x, n_y, n_z, limiter_x_start, limiter_y_start, limiter_z_start,
                                         limiter_x_end, limiter_y_end, limiter_z_end+2, surface,
                                         np.zeros((surface_elements, 3), dtype=np.int32), a, a_rad, b, b_rad,
                                         misplaced_voxels)
    return surface, surface_reduced, sample_holder


@jit
def reduce_surface(n_x, n_y, n_z, limiter_x, limiter_y, limiter_z, limiter_x_end, limiter_y_end, limiter_z_end, surface, surface_reduced, a, a_rad, b, b_rad, misplaced_voxels):
    count = 0
    #z only runs up to n_z-2 to ignore the bottom plus the puffer layer
    for i in range(limiter_z, limiter_z_end-2):
        for j in range(limiter_y, limiter_y_end):
            for k in range(limiter_x, limiter_x_end):
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
            if array[1][j][k] == 1 and max(array[1][j-1][k], array[1][j+1][k]) + max(array[1][j][k-1], array[1][j][k+1]) == 2 and array[2][j][k] == 0: #and ((k - a) / a_rad) ** 2 + ((j - b) / b_rad) ** 2 <= 1.015:
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
    nr_of_last_sus_elements = 0
    for count, each in enumerate(array):
        for i in range(0, 6):
            if np.sum(surface[each[2] + n_z_lr[i]][each[1] + n_y_lr[i]][each[0] + n_x_lr[i]]) == 0 and temperature[each[2] + n_z_lr[i]][each[1] + n_y_lr[i]][each[0] + n_x_lr[i]] == 0:
                surrounding_surface[count] = np.array([each[0] + n_x_lr[i], each[1] + n_y_lr[i], each[2] + n_z_lr[i]], dtype=np.float32)
                count += 1
                if count == len(array) - 1:
                    nr_of_last_sus_elements += 1
    for i in range(len(surrounding_surface)):
        for j in range(i+1-len(surrounding_surface)):
            if surrounding_surface[i][0] == surrounding_surface[j][0] and surrounding_surface[i][1] == surrounding_surface[j][1] and surrounding_surface[i][2] == surrounding_surface[j][2]:
                np.delete(surrounding_surface, j)
    return surrounding_surface[0:count]


@njit
def update_surface_arrays(voxels_to_delete, surface, reduced_surface, temperature, n_x, n_y, n_z, a, a_rad, b, b_rad):
    for each in voxels_to_delete:
        temperature[each[2]][each[1]][each[0]] = 0
        #print(reduced_surface)
        surface, new_reduced_surface_elements = find_surface(n_x, n_y, n_z, each[0]-1, each[1]-1, each[2]-1, each[0]+2, each[1]+2, each[2]+2, temperature, surface, a, a_rad, b, b_rad, False)[0:2]
        #new_surrounding_surface = surrounding_checker(np.append(new_reduced_surface_elements, ), surface, n_x_lr, n_y_lr, n_z_lr)
        #doubled_elements = np.empty((0, 0), dtype=np.int32)
        #print(new_reduced_surface_elements)
        #print(reduced_surface)
        non_duplicate_indicies = np.empty(0, dtype=np.int32)
        mask = np.arange(len(reduced_surface))
        empty_voxel_counted = np.nan
        #delete_indicies = np.array([])
        for count, red_elements in enumerate(new_reduced_surface_elements):
            is_in = False
            for count_el, elements in enumerate(reduced_surface):
                if red_elements[0] == elements[0] and red_elements[1] == elements[1] and red_elements[2] == elements[2]:
                    is_in = True
                    break
                if elements[0] == each[0] and elements[1] == each[1] and elements[2] == each[2]:
                    empty_voxel_counted = count_el
            if not is_in:
                non_duplicate_indicies = np.append(non_duplicate_indicies, np.int32(count))
        if empty_voxel_counted != np.nan:
            mask = np.delete(mask, int(empty_voxel_counted))
            reduced_surface  = reduced_surface[mask]
        new_r_temp = np.zeros((len(reduced_surface) + len(non_duplicate_indicies), 3), dtype=np.int32)
        new_r_temp[0:len(reduced_surface)] = reduced_surface
        count_2 = len(reduced_surface)
        for i in non_duplicate_indicies:
            new_r_temp[count_2] = new_reduced_surface_elements[i]
            count_2 += 1
        #print(new_reduced_surface_elements)
        #new_reduced_surface_elements = np.delete(new_reduced_surface_elements, delete_indicies, axis=0)
        #print(new_reduced_surface_elements)
        #reduced_surface = np.append(reduced_surface, new_reduced_surface_elements, axis=0)
        reduced_surface = new_r_temp
    return surface, reduced_surface

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


def get_sample_holder_adjacency(n_x, n_y, n_z, sample_holder, temperature):
    sh_adjacent_voxels = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if sample_holder[i+1][j][k] == 1 and temperature[i][j][k] > 0:
                    sh_adjacent_voxels[i][j][k][0] = 1
                if sample_holder[i-1][j][k] == 1 and temperature[i][j][k] > 0:
                    sh_adjacent_voxels[i][j][k][1] = 1
                if sample_holder[i][j+1][k] == 1 and temperature[i][j][k] > 0:
                    sh_adjacent_voxels[i][j][k][2] = 1
                if sample_holder[i][j-1][k] == 1 and temperature[i][j][k] > 0:
                    sh_adjacent_voxels[i][j][k][3] = 1
                if sample_holder[i][j][k+1] == 1 and temperature[i][j][k] > 0:
                    sh_adjacent_voxels[i][j][k][4] = 1
                if sample_holder[i][j][k-1] == 1 and temperature[i][j][k] > 0:
                    sh_adjacent_voxels[i][j][k][5] = 1
    return sh_adjacent_voxels
