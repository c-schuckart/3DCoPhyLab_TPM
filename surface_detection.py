import numpy as np
from numba import jit
import settings as sett


'''This function creates an equidistant mesh of either cylindrical or rectangular shape. 
The surface is marked as an array of six values, which each correspond to one of the sides, a "1" signaling it being
an exposed tile, while a "0" denotes a non surface element. 
Array entries 0 to 5 correspond to "z positive", "z negative", "y positve", "y negative", "x positive" and "x negative"'''
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
    surface_reduced = reduce_surface(n_x, n_y, n_z, limiter_x, limiter_y, limiter_z, surface, np.zeros((surface_elements, 3), dtype=np.int32), a, a_rad, b, b_rad)
    return surface, surface_reduced


@jit
def reduce_surface(n_x, n_y, n_z, limiter_x, limiter_y, limiter_z, surface, surface_reduced, a, a_rad, b, b_rad):
    count = 0
    #z only runs up to n_z-2 to ignore the bottom plus the puffer layer
    for i in range(limiter_z, n_z-2):
        for j in range(limiter_y, n_y):
            for k in range(limiter_x, n_x):
                if sum(surface[i][j][k]) != 0 and ((k - a)/a_rad)**2 + ((j - b)/b_rad)**2 <= 1:
                    surface_reduced[count] = np.array([k, j, i], dtype=np.int32)
                    count += 1
    return surface_reduced

#@jit
def DEBUG_print_3D_arrays(n_x, n_y, n_z, mesh):
    for i in range(0, n_z):
        print('Layer: ' + str(i + 1))
        for j in range(0, n_y):
            for k in range(0, n_x):
                print(mesh[i][j][k], end=' ')
            print('\n')
