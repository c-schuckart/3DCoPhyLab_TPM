import numpy as np
from numba import jit
import settings as sett


'''This function creates an equidistant mesh of either cylindrical or rectangular shape. 
The surface is marked as an array of six values, which each correspond to one of the sides, a "1" signaling it being
an exposed tile, while a "0" denotes a non surface element. 
Array entries 0 to 5 correspond to "z positve", "z negative", "y positve", "y negative", "x positive" and "x negative"'''
def create_equidistant_mesh(n_x, n_y, n_z, temperature_ini):
    if sett.mesh_form == 1:
        a = n_x//2
        a_rad = (n_x - 2)//2
        b = n_y//2
        b_rad = (n_y - 2) // 2
        x, y = np.ogrid[:n_x, :n_y]
        mesh = np.zeros((n_z, n_y, n_x))
        slice = np.zeros((n_y, n_x))
        mask = ((x-a)/a_rad)**2 + ((y-b)/b_rad)**2 <= 1
        slice[mask] = temperature_ini
        for i in range(0, n_z-1):
            if i !=  0:
                mesh[i] = slice
    elif sett.mesh_form == 0:
        mesh = np.full((n_z, n_y, n_x), temperature_ini)
    else:
        raise NotImplementedError
    return mesh


@jit
def find_surface(n_x, n_y, n_z, limiter_x, limiter_y, limiter_z, mesh, surface):
    for i in range(limiter_z, n_z):
        for j in range(limiter_y, n_y):
            for k in range(limiter_x, n_x):
                if mesh[i][j][k] != 0:
                    #Check if it is a surface in positive z direction
                    if mesh[i-1][j][k] == 0:
                        surface[i][j][k][0] = 1
                    # Check if it is a surface in negative z direction
                    if mesh[i+1][j][k] == 0:
                        surface[i][j][k][1] = 1
                    # Check if it is a surface in positive y direction
                    if mesh[i][j+1][k] == 0:
                        surface[i][j][k][2] = 1
                    # Check if it is a surface in negative y direction
                    if mesh[i][j-1][k] == 0:
                        surface[i][j][k][3] = 1
                    # Check if it is a surface in positive x direction
                    if mesh[i][j][k+1] == 0:
                        surface[i][j][k][1] = 1
                    # Check if it is a surface in negative x direction
                    if mesh[i][j][k-1] == 0:
                        surface[i][j][k][1] = 1
    return surface


#@jit
def DEBUG_print_3D_arrays(n_x, n_y, n_z, mesh):
    for i in range(0, n_z):
        print('Layer: ' + str(i + 1))
        for j in range(0, n_y):
            for k in range(0, n_x):
                print(mesh[i][j][k], end=' ')
            print('\n')
