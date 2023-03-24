import numpy as np
from numba import jit, njit, prange

@njit(parallel=True)
def set_inner_matrices(n_x, n_y, n_z, dx, dy, dz, Dr, Lambda, dt, density, heat_capacity, Q_const, Q_lin, temperature, surface_reduced, sample_holder, input_energy, albedo, sigma, epsilon):
    A_bottom = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_top = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_south = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_north = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_west = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_east = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_point = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    q = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature > 0 and sample_holder[i][j][k] != 1:
                    A_bottom[i][j][k] = - Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
                    A_top[i][j][k] = - Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
                    A_south[i][j][k] = - Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
                    A_north[i][j][k] = - Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
                    A_west[i][j][k] = - Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
                    A_east[i][j][k] = - Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
                    A_point[i][j][k] = - A_bottom[i][j][k] - A_top[i][j][k] - A_south[i][j][k] - A_north[i][j][k] - A_west[i][j][k] - A_east[i][j][k] + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - Q_lin * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
                    q[i][j][k] = Q_const * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
    #Right now the volume at the surface is not a half volume. I'm not sure if this is effecting anything and it will have to be tested. It would then always require the adaptive mesh algorithm that slices the z-blocks.
    for each in surface_reduced:
        A_bottom[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0]
        A_top[each[2]][each[1]][each[0]] = 0
        A_south[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2]
        A_north[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3]
        A_west[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4]
        A_east[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[5]
        A_point[each[2]][each[1]][each[0]] = - A_bottom[each[2]][each[1]][each[0]] - A_top[each[2]][each[1]][each[0]] - A_south[each[2]][each[1]][each[0]] - A_north[each[2]][each[1]][each[0]] - A_west[each[2]][each[1]][each[0]] - \
                           A_east[each[2]][each[1]][each[0]] + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * \
                           dz[each[2]][each[1]][each[0]] / dt - Q_lin * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - (-4 * sigma * epsilon * temperature[each[2]][each[1]][each[0]]**3)
        q[each[2]][each[1]][each[0]] = Q_const * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * \
                     dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt * temperature[each[2]][each[1]][each[0]] + input_energy * (1 - albedo) + 3 * sigma * epsilon * temperature[each[2]][each[1]][each[0]]**4