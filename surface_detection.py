import numpy as np
from numba import jit
import settings as sett

@jit
def create_equidistant_mesh(n_x, n_y, n_z, temperature_ini):
    if sett.mesh_form == 1:
        a = n_x//2
        b = n_y//2
        x, y = np.ogrid[n_x, n_y]
        mesh = np.zeros((n_z, n_y, n_x))
        slice = np.zeros(n_y, n_x)
        mask = ((x-a)/a)**2 + ((y-b)/b)**2 <= 1
        slice[mask] = temperature_ini
        for i in range(0, n_z):
            mesh[i] = slice
    elif sett.mesh_form == 0:
        mesh = np.full((n_z, n_y, n_x), temperature_ini)
    else:
        raise NotImplementedError
    return mesh

