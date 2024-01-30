import numpy as np
from numba import jit, njit, prange
from tri_diag_solve import tridiagonal_matrix_solver, periodic_tridiagonal_matrix_solver


@njit
def set_matrices_lhs_z_sweep_de(n_z, j, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[i] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_t = 1 / 2 * Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = 1 / 2 * Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            sub_alpha[i] = - a_t
            sub_gamma[i] = - a_b
            diag[i] = a_t + a_b + dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif temperature[i][j][k] == 0 and diag[i] == 0:
            diag[i] = dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif np.sum(surface[i][j][k]) != 0 and diag[i] == 0:
            diag[i] = dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_z_sweep_de(n_z, j, k, rhs, gas_mass, x_sweep_gas_mass, y_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[i] == 0:
            a_t = 1 / 2 * Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = 1 / 2 * Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = Diffusion_coefficient[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = Diffusion_coefficient[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            rhs[i] = a_t * gas_mass[i+1][j][k] + a_b * gas_mass[i-1][j][k] + a_n * (y_sweep_gas_mass[i][j+1][k] + gas_mass[i][j+1][k])/2 + a_s * (y_sweep_gas_mass[i][j-1][k] + gas_mass[i][j-1][k])/2 + a_e * (x_sweep_gas_mass[i][j][k+1] + gas_mass[i][j][k+1])/2 + a_w * (x_sweep_gas_mass[i][j][k-1] + gas_mass[i][j][k-1])/2 + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b) * gas_mass[i][j][k] - (a_n + a_s) * (y_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2 - (a_w + a_e) * (x_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2
        elif temperature[i][j][k] == 0 and rhs[i] == 0:
            rhs[i] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[i] == 0:
            rhs[i] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
    return rhs


@njit
def set_matrices_rhs_z_sweep_de_zfirst(n_z, j, k, rhs, gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[i] == 0:
            a_t = 1 / 2 * Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = 1 / 2 * Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = Diffusion_coefficient[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = Diffusion_coefficient[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            rhs[i] = a_t * gas_mass[i+1][j][k] + a_b * gas_mass[i-1][j][k] + a_n * gas_mass[i][j+1][k] + a_s * gas_mass[i][j-1][k] + a_e * gas_mass[i][j][k+1] + a_w * gas_mass[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s - a_w - a_e) * gas_mass[i][j][k]
        elif temperature[i][j][k] == 0 and rhs[i] == 0:
            rhs[i] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[i] == 0:
            #rhs[i] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
            rhs[i] = 0
    return rhs


@njit
def boundary_condition_implicit_z_sweep_de(dt, gas_mass, x_sweep_gas_mass, y_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero):
    for each in surface_reduced:
        if not top_layer_zero:
            a_t = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[2]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_t + a_b + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[2]] = - a_t
            sub_gamma[each[2]] = - a_b
            rhs[each[2]] = a_t * gas_mass[each[2] + 1][each[1]][each[0]] + a_b * gas_mass[each[2] - 1][each[1]][each[0]] + a_n * (y_sweep_gas_mass[each[2]][each[1] + 1][each[0]] + gas_mass[each[2]][each[1] + 1][each[0]])/2 + a_s * (y_sweep_gas_mass[each[2]][each[1] - 1][each[0]] + gas_mass[each[2]][each[1] - 1][each[0]])/2 + a_e * (x_sweep_gas_mass[each[2]][each[1]][each[0] + 1] + gas_mass[each[2]][each[1]][each[0] + 1])/2 + a_w * (x_sweep_gas_mass[each[2]][each[1]][each[0] - 1] + gas_mass[each[2]][each[1]][each[0] - 1])/2 + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b) * gas_mass[each[2]][each[1]][each[0]] - (a_n + a_s) * (y_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2 - (a_w + a_e) * (x_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2
        else:
            a_t = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = 0
            a_n = Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[2]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_t + a_b + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[2]] = - a_t
            sub_gamma[each[2]] = - a_b
            rhs[each[2]] = a_t * gas_mass[each[2] + 1][each[1]][each[0]] + a_b * gas_mass[each[2] - 1][each[1]][each[0]] + a_n * (y_sweep_gas_mass[each[2]][each[1] + 1][each[0]] + gas_mass[each[2]][each[1] + 1][each[0]])/2 + a_s * (y_sweep_gas_mass[each[2]][each[1] - 1][each[0]] + gas_mass[each[2]][each[1] - 1][each[0]])/2 + a_e * (x_sweep_gas_mass[each[2]][each[1]][each[0] + 1] + gas_mass[each[2]][each[1]][each[0] + 1])/2 + a_w * (x_sweep_gas_mass[each[2]][each[1]][each[0] - 1] + gas_mass[each[2]][each[1]][each[0] - 1])/2 + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b) * gas_mass[each[2]][each[1]][each[0]] - (a_n + a_s) * (y_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2 - (a_w + a_e) * (x_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2
    return sub_alpha, diag, sub_gamma, rhs


@njit
def boundary_condition_implicit_z_sweep_de_zfirst(dt, gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero):
    for each in surface_reduced:
        if not top_layer_zero:
            a_t = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[2]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_t + a_b + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[2]] = - a_t
            sub_gamma[each[2]] = - a_b
            rhs[each[2]] = a_t * gas_mass[each[2] + 1][each[1]][each[0]] + a_b * gas_mass[each[2] - 1][each[1]][each[0]] + a_n * gas_mass[each[2]][each[1] + 1][each[0]] + a_s * gas_mass[each[2]][each[1] - 1][each[0]] + a_e * gas_mass[each[2]][each[1]][each[0] + 1] + a_w * gas_mass[each[2]][each[1]][each[0] - 1] + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_w - a_e) * gas_mass[each[2]][each[1]][each[0]]
        else:
            diag[each[2]] = dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            rhs[each[2]] = 0
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_lhs_y_sweep_de(n_y, i, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature, simulate_region):
    for j in range(0, n_y):
        if (temperature[i][j][k] > 0 or simulate_region[i][j][k] == 1) and np.sum(surface[i][j][k]) == 0 and diag[j] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_n = 1 / 2 * Diffusion_coefficient[i][j][k][2] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = 1 / 2 * Diffusion_coefficient[i][j][k][3] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            sub_alpha[j] = - a_n
            sub_gamma[j] = - a_s
            diag[j] = a_n + a_s + dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif temperature[i][j][k] == 0 and diag[j] == 0:
            diag[j] = dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif np.sum(surface[i][j][k]) != 0 and diag[j] == 0:
            diag[j] = dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_y_sweep_de(n_y, i, k, rhs, gas_mass, x_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature):
    for j in range(0, n_y):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[j] == 0:
            a_t = Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = 1 / 2 * Diffusion_coefficient[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = 1 / 2 * Diffusion_coefficient[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            rhs[j] = a_t * gas_mass[i+1][j][k] + a_b * gas_mass[i-1][j][k] + a_n * gas_mass[i][j+1][k] + a_s * gas_mass[i][j-1][k] + a_e * (x_sweep_gas_mass[i][j][k+1] + gas_mass[i][j][k+1])/2 + a_w * (x_sweep_gas_mass[i][j][k-1] + gas_mass[i][j][k-1])/2 + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s) * gas_mass[i][j][k] - (a_w + a_e) * (x_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2
        elif temperature[i][j][k] == 0 and rhs[j] == 0:
            rhs[j] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[j] == 0:
            rhs[j] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
    return rhs


@njit
def set_matrices_rhs_y_sweep_de_zfirst(n_y, i, k, rhs, gas_mass, z_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature):
    for j in range(0, n_y):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[j] == 0:
            a_t = Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = 1 / 2 * Diffusion_coefficient[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = 1 / 2 * Diffusion_coefficient[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            rhs[j] = a_t * (z_sweep_gas_mass[i+1][j][k] + gas_mass[i+1][j][k])/2 + a_b * (z_sweep_gas_mass[i-1][j][k] + gas_mass[i-1][j][k])/2 + a_n * gas_mass[i][j+1][k] + a_s * gas_mass[i][j-1][k] + a_e * gas_mass[i][j][k+1] + a_w * gas_mass[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_n - a_s - a_w - a_e) * gas_mass[i][j][k] - (a_t + a_b) * (z_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2
        elif temperature[i][j][k] == 0 and rhs[j] == 0:
            rhs[j] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[j] == 0:
            #rhs[j] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
            rhs[i] = 0
    return rhs


@njit
def boundary_condition_implicit_y_sweep_de(dt, gas_mass, x_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero):
    for each in surface_reduced:
        if not top_layer_zero:
            a_t = Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[1]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_n + a_s + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[1]] = - a_n
            sub_gamma[each[1]] = - a_s
            rhs[each[1]] = a_t * gas_mass[each[2] + 1][each[1]][each[0]] + a_b * gas_mass[each[2] - 1][each[1]][each[0]] + a_n * gas_mass[each[2]][each[1] + 1][each[0]] + a_s * gas_mass[each[2]][each[1] - 1][each[0]] + a_e * (x_sweep_gas_mass[each[2]][each[1]][each[0] + 1] + gas_mass[each[2]][each[1]][each[0] + 1])/2 + a_w * (x_sweep_gas_mass[each[2]][each[1]][each[0] - 1] + gas_mass[each[2]][each[1]][each[0] - 1])/2 + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s) * gas_mass[each[2]][each[1]][each[0]] - (a_w + a_e) * (x_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2
        else:
            a_t = Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = 0
            a_n = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[1]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] +  a_n + a_s + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[1]] = - a_n
            sub_gamma[each[1]] = - a_s
            rhs[each[1]] = a_t * gas_mass[each[2] + 1][each[1]][each[0]] + a_b * gas_mass[each[2] - 1][each[1]][each[0]] + a_n * gas_mass[each[2]][each[1] + 1][each[0]] + a_s * gas_mass[each[2]][each[1] - 1][each[0]] + a_e * (x_sweep_gas_mass[each[2]][each[1]][each[0] + 1] + gas_mass[each[2]][each[1]][each[0] + 1])/2 + a_w * (x_sweep_gas_mass[each[2]][each[1]][each[0] - 1] + gas_mass[each[2]][each[1]][each[0] - 1])/2 + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s) * gas_mass[each[2]][each[1]][each[0]] - (a_w + a_e) * (x_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2
    return sub_alpha, diag, sub_gamma, rhs


@njit
def boundary_condition_implicit_y_sweep_de_zfirst(dt, gas_mass, z_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero):
    for each in surface_reduced:
        if not top_layer_zero:
            a_t = Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[1]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_n + a_s + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[1]] = - a_n
            sub_gamma[each[1]] = - a_s
            rhs[each[1]] = a_t * (z_sweep_gas_mass[each[2] + 1][each[1]][each[0]] + gas_mass[each[2] + 1][each[1]][each[0]])/2 + a_b * (z_sweep_gas_mass[each[2] - 1][each[1]][each[0]] + gas_mass[each[2] - 1][each[1]][each[0]])/2 + a_n * gas_mass[each[2]][each[1] + 1][each[0]] + a_s * gas_mass[each[2]][each[1] - 1][each[0]] + a_e * gas_mass[each[2]][each[1]][each[0] + 1] + a_w * gas_mass[each[2]][each[1]][each[0] - 1] + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_n - a_s - a_w - a_e) * gas_mass[each[2]][each[1]][each[0]] - (a_t + a_b) * (z_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2
        else:
            diag[each[1]] = dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            rhs[each[1]] = 0
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_lhs_x_sweep_de(n_x, i, j, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature, simulate_region):
    for k in range(0, n_x):
        if (temperature[i][j][k] > 0 or simulate_region[i][j][k] == 1) and np.sum(surface[i][j][k]) == 0 and diag[k] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_e = 1/2 * Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = 1/2 * Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            sub_alpha[k] = - a_e
            sub_gamma[k] = - a_w
            diag[k] = a_e + a_w + dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif temperature[i][j][k] == 0 and diag[k] == 0:
            diag[k] = dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif np.sum(surface[i][j][k]) != 0 and diag[k] == 0:
            diag[k] = dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_x_sweep_de(n_x, i, j, rhs, gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature):
    for k in range(0, n_x):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[k] == 0:
            a_t = Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = Diffusion_coefficient[i][j][k][2] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = Diffusion_coefficient[i][j][k][3] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = 1 / 2 * Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = 1 / 2 * Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            rhs[k] = a_t * gas_mass[i+1][j][k] + a_b * gas_mass[i-1][j][k] + a_n * gas_mass[i][j+1][k] + a_s * gas_mass[i][j-1][k] + a_e * gas_mass[i][j][k+1] + a_w * gas_mass[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s - a_e - a_w) * gas_mass[i][j][k]
        elif temperature[i][j][k] == 0 and rhs[k] == 0:
            rhs[k] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[k] == 0:
            rhs[k] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
    return rhs


@njit
def set_matrices_rhs_x_sweep_de_zfirst(n_x, i, j, rhs, gas_mass, z_sweep_gas_mass, y_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature):
    for k in range(0, n_x):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[k] == 0:
            a_t = Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = Diffusion_coefficient[i][j][k][2] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = Diffusion_coefficient[i][j][k][3] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = 1 / 2 * Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = 1 / 2 * Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            rhs[k] = a_t * (z_sweep_gas_mass[i+1][j][k] + gas_mass[i+1][j][k])/2 + a_b * (z_sweep_gas_mass[i-1][j][k] + gas_mass[i-1][j][k])/2 + a_n * (y_sweep_gas_mass[i][j+1][k] + gas_mass[i][j+1][k])/2 + a_s * (y_sweep_gas_mass[i][j-1][k] + gas_mass[i][j-1][k])/2 + a_e * gas_mass[i][j][k+1] + a_w * gas_mass[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_e - a_w) * gas_mass[i][j][k] - (a_t + a_b) * (z_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2 - (a_n + a_s) * (y_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2
        elif temperature[i][j][k] == 0 and rhs[k] == 0:
            rhs[k] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[k] == 0:
            #rhs[k] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
            rhs[i] = 0
    return rhs


@njit
def boundary_condition_implicit_x_sweep_de(dt, gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero):
    for each in surface_reduced:
        if not top_layer_zero:
            a_t = Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_e + a_w + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[0]] = - a_e
            sub_gamma[each[0]] = - a_w
            rhs[each[0]] = a_t * gas_mass[each[2] + 1][each[1]][each[0]] + a_b * gas_mass[each[2] - 1][each[1]][each[0]] + a_n * gas_mass[each[2]][each[1] + 1][each[0]] + a_s * gas_mass[each[2]][each[1] - 1][each[0]] + a_e * gas_mass[each[2]][each[1]][each[0] + 1] + a_w * gas_mass[each[2]][each[1]][each[0] - 1] + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_e - a_w) * gas_mass[each[2]][each[1]][each[0]]
        else:
            a_t = Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = 0
            a_n = Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_e + a_w + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[0]] = - a_e
            sub_gamma[each[0]] = - a_w
            rhs[each[0]] = a_t * gas_mass[each[2] + 1][each[1]][each[0]] + a_b * gas_mass[each[2] - 1][each[1]][each[0]] + a_n * gas_mass[each[2]][each[1] + 1][each[0]] + a_s * gas_mass[each[2]][each[1] - 1][each[0]] + a_e * gas_mass[each[2]][each[1]][each[0] + 1] + a_w * gas_mass[each[2]][each[1]][each[0] - 1] + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_e - a_w) * gas_mass[each[2]][each[1]][each[0]]
        '''if rhs[each[0]] < 0:
            print(gas_mass[each[2]][each[1]][each[0]], gas_mass[each[2]+1][each[1]][each[0]], gas_mass[each[2]-1][each[1]][each[0]], gas_mass[each[2]][each[1]+1][each[0]], gas_mass[each[2]][each[1]-1][each[0]], gas_mass[each[2]][each[1]][each[0]+1], gas_mass[each[2]][each[1]][each[0]-1])
            print(a_t, a_b, a_n, a_s, a_e, a_w, Diffusion_coefficient[each[2]][each[1]][each[0]])
            print((dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_e - a_w), S_c[each[2]][each[1]][each[0]])
            print((dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_e - a_w) * gas_mass[each[2]][each[1]][each[0]], S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])'''
    return sub_alpha, diag, sub_gamma, rhs


@njit
def boundary_condition_implicit_x_sweep_de_zfirst(dt, gas_mass, z_sweep_gas_mass, y_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero):
    for each in surface_reduced:
        if not top_layer_zero:
            a_t = Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_e + a_w + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[0]] = - a_e
            sub_gamma[each[0]] = - a_w
            rhs[each[0]] = a_t * (z_sweep_gas_mass[each[2] + 1][each[1]][each[0]] + gas_mass[each[2] + 1][each[1]][each[0]])/2 + a_b * (z_sweep_gas_mass[each[2] - 1][each[1]][each[0]] + gas_mass[each[2] - 1][each[1]][each[0]])/2 + a_n * (y_sweep_gas_mass[each[2]][each[1] + 1][each[0]] + gas_mass[each[2]][each[1] + 1][each[0]])/2 + a_s * (y_sweep_gas_mass[each[2]][each[1] - 1][each[0]] + gas_mass[each[2]][each[1] - 1][each[0]])/2 + a_e * gas_mass[each[2]][each[1]][each[0] + 1] + a_w * gas_mass[each[2]][each[1]][each[0] - 1] + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_e - a_w) * gas_mass[each[2]][each[1]][each[0]] - (a_t + a_b) * (z_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2 - (a_n + a_s) * (y_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2
        else:
            diag[each[0]] = dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            rhs[each[0]] = 0
    return sub_alpha, diag, sub_gamma, rhs


@njit
def de_implicit_DGADI(n_x, n_y, n_z, surface_reduced, dt, gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, S_c, S_p, sh_adjacent_voxels, temperature, top_layer_zero, surrounding_layer, t):
    next_step_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    x_sweep_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    y_sweep_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            sub_alpha = np.zeros(n_x, dtype=np.float64)
            diag = np.zeros(n_x, dtype=np.float64)
            sub_gamma = np.zeros(n_x, dtype=np.float64)
            rhs = np.zeros(n_x, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[1] == j:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            for each in surrounding_layer:
                if each[2] == i and each[1] == j:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_x_sweep_de(dt, gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero)
            '''if t == 1 and i == 2 and j == 4 or i == 2 and j == 4:
                print(rhs)'''
            sub_alpha, diag, sub_gamma = set_matrices_lhs_x_sweep_de(n_x, i, j, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature)
            rhs = set_matrices_rhs_x_sweep_de(n_x, i, j, rhs, gas_mass, surface,dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature)
            '''print(sub_gamma)
            print(diag)
            print(sub_alpha)
            print(rhs)
            print(i, j)'''
            x_sweep_gas_mass[i, j, 0:n_x] = tridiagonal_matrix_solver(n_x, diag, sub_gamma, sub_alpha, rhs)
            '''if t == 1 and i == 2 and j == 4 or (np.greater(np.zeros(n_x, dtype=np.float64), x_sweep_gas_mass[i, j, 0:n_x])).any():
                print(sub_gamma)
                print(diag)
                print(sub_alpha)
                print(rhs)
                print(x_sweep_gas_mass[i, j, 0:n_x])
                print(i, j)'''
    #print(np.sum(x_sweep_gas_mass * dx * dy * dz) - np.sum(gas_mass * dx * dy * dz))
    #print((x_sweep_gas_mass * dx * dy * dz)[1:3, 10:20, 10:20], 'x')
    #print(x_sweep_gas_mass[1])
    #print(x_sweep_gas_mass[2])
    for i in range(1, n_z-1):
        for k in range(1, n_x-1):
            sub_alpha = np.zeros(n_y, dtype=np.float64)
            diag = np.zeros(n_y, dtype=np.float64)
            sub_gamma = np.zeros(n_y, dtype=np.float64)
            rhs = np.zeros(n_y, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            for each in surrounding_layer:
                if each[2] == i and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_y_sweep_de(dt, gas_mass, x_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_y_sweep_de(n_y, i, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature)
            rhs = set_matrices_rhs_y_sweep_de(n_y, i, k, rhs, gas_mass, x_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c,sh_adjacent_voxels, temperature)
            y_sweep_gas_mass[i, 0:n_y, k] = tridiagonal_matrix_solver(n_y, diag, sub_gamma, sub_alpha, rhs)
    #print(np.sum(y_sweep_gas_mass * dx * dy * dz) - np.sum(gas_mass * dx * dy * dz))
    #print((y_sweep_gas_mass * dx * dy * dz)[1:3, 10:20, 10:20], 'y')
    for j in range(1, n_y-1):
        for k in range(1, n_x-1):
            sub_alpha = np.zeros(n_z, dtype=np.float64)
            diag = np.zeros(n_z, dtype=np.float64)
            sub_gamma = np.zeros(n_z, dtype=np.float64)
            rhs = np.zeros(n_z, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[1] == j and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            for each in surrounding_layer:
                if each[1] == j and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            #print(surface_elements_in_line[0:counter])
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_z_sweep_de(dt, gas_mass, x_sweep_gas_mass, y_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero)
            '''if j == 16 and k == 16:
                print(sub_gamma)
                print(diag)
                print(sub_alpha)
                print(rhs)'''
            sub_alpha, diag, sub_gamma = set_matrices_lhs_z_sweep_de(n_z, j, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature)
            rhs = set_matrices_rhs_z_sweep_de(n_z, j, k, rhs, gas_mass, x_sweep_gas_mass, y_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature)
            '''if j == 16 and k == 16:
                print(sub_gamma)
                print(diag)
                print(sub_alpha)
                print(rhs)'''
            next_step_gas_mass[0:n_z, j, k] = tridiagonal_matrix_solver(n_z, diag, sub_gamma, sub_alpha, rhs)
    #print(np.sum(next_step_gas_mass * dx * dy * dz) - np.sum(gas_mass * dx * dy * dz))
    #print((next_step_gas_mass * dx * dy * dz)[1:3, 10:20, 10:20], 'z')
    return next_step_gas_mass


@njit
def de_implicit_DGADI_zfirst(n_x, n_y, n_z, surface_reduced, dt, gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, S_c, S_p, sh_adjacent_voxels, top_layer_zero, temperature, simulate_region):

    '''
    Solving of the gas diffusion equation (Fick's second law) via the ADI method proposed by Douglas 1962.
    The function calls within each set parts of the needed matrix to calculate the gas densities.

    n_x : float
			number of numerical layers in x-direction
		n_y : float
			number of numerical layers in y-direction
		n_z : float
			number of numerical layers in x-direction
        surface_reduced : ndarray
            Array containing the coordinates of all surface elements of dimension s * 3, s = #surface voxels
        dt : float
            Time step
        gas_mass : ndarray
            Gas density of the system at the current time step of dimension n_z * n_y * n_x
        Diffusion_coefficient : ndarray
            Diffusion coefficients between the faces of each voxel of dimension n_z * n_y * n_x * 6
        Dr : ndarray
			Array containing the distances between the mid-points of the numerical layers of dimension n_z * n_y * n_x * 6
        dx : ndarray
			Array containing the thickness of the numerical layers in x-direction of dimension n_z * n_y * n_x
		dy : ndarray
			Array containing the thickness of the numerical layers in y-direction of dimension n_z * n_y * n_x
		dz : ndarray
			Array containing the thickness of the numerical layers in z-direction of dimension n_z * n_y * n_x
        surface : ndarray
            Array marking if a face of a voxel is on the surface of the geometry or not of dimension n_z * n_y * n_x
        S_c : ndarray
            0th order terms for the Taylor series expansion of the temperature dependent source and sink terms of dimension n_z * n_y * n_x
        S_p : ndarray
            1st order terms for the Taylor series expansion of the temperature dependent source and sink terms of dimension n_z * n_y * n_x
        sh_adjacent_voxels : ndarray
            To be deleted
        top_layer_zero : bool
            To be deleted
        temperature : ndarray
			Temperature of the system at the current time step of dimension n_z * n_y * n_x
        simulate_region : ndarray
            To be deleted

    Returns:
		next_step_gas_mass : ndarray
			Array containing the gas densities calculated during this time step of dimension n_z * n_y * n_x
    '''

    next_step_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    z_sweep_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    y_sweep_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for j in range(1, n_y-1):
        for k in range(1, n_x-1):
            sub_alpha = np.zeros(n_z, dtype=np.float64)
            diag = np.zeros(n_z, dtype=np.float64)
            sub_gamma = np.zeros(n_z, dtype=np.float64)
            rhs = np.zeros(n_z, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[1] == j and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            '''for each in surrounding_layer:
                if each[1] == j and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1'''
            #print(surface_elements_in_line[0:counter])
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_z_sweep_de_zfirst(dt, gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_z_sweep_de(n_z, j, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature)
            rhs = set_matrices_rhs_z_sweep_de_zfirst(n_z, j, k, rhs, gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature)
            z_sweep_gas_mass[0:n_z, j, k] = tridiagonal_matrix_solver(n_z, diag, sub_gamma, sub_alpha, rhs)
            '''if np.isnan(z_sweep_gas_mass[0:n_z, j, k]).any() and k == 1 and j == 1:
                print('g:', sub_gamma)
                print('d:', diag)
                print('a:', sub_alpha)
                print('rhs 2.:', rhs)
                print('z:', z_sweep_gas_mass[1:n_z - 1, j, k], j, k)'''
    for i in range(1, n_z-1):
        for k in range(1, n_x-1):
            sub_alpha = np.zeros(n_y, dtype=np.float64)
            diag = np.zeros(n_y, dtype=np.float64)
            sub_gamma = np.zeros(n_y, dtype=np.float64)
            rhs = np.zeros(n_y, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            '''for each in surrounding_layer:
                if each[2] == i and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1'''
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_y_sweep_de_zfirst(dt, gas_mass, z_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_y_sweep_de(n_y, i, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature, simulate_region)
            rhs = set_matrices_rhs_y_sweep_de_zfirst(n_y, i, k, rhs, gas_mass, z_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c,sh_adjacent_voxels, temperature)
            y_sweep_gas_mass[i, 0:n_y, k] = tridiagonal_matrix_solver(n_y, diag, sub_gamma, sub_alpha, rhs)
            '''if np.isnan(y_sweep_gas_mass[i, 0:n_y, k]).any():
                print('g:', sub_gamma)
                print('d:', diag)
                print('a:', sub_alpha)
                print('rhs 2.:', rhs)
                print('y:', y_sweep_gas_mass[i, 0:n_y, k], i, k)'''
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            sub_alpha = np.zeros(n_x, dtype=np.float64)
            diag = np.zeros(n_x, dtype=np.float64)
            sub_gamma = np.zeros(n_x, dtype=np.float64)
            rhs = np.zeros(n_x, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[1] == j:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            '''for each in surrounding_layer:
                if each[2] == i and each[1] == j:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1'''
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_x_sweep_de_zfirst(dt, gas_mass, z_sweep_gas_mass, y_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_x_sweep_de(n_x, i, j, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature, simulate_region)
            rhs = set_matrices_rhs_x_sweep_de_zfirst(n_x, i, j, rhs, gas_mass, z_sweep_gas_mass, y_sweep_gas_mass, surface,dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, temperature)
            next_step_gas_mass[i, j, 0:n_x] = tridiagonal_matrix_solver(n_x, diag, sub_gamma, sub_alpha, rhs)
            if np.isnan(next_step_gas_mass[i, j, 0:n_x]).any():
                print('g:', sub_gamma)
                print('d:', diag)
                print('a:', sub_alpha)
                print('rhs 2.:', rhs)
                print('nxt:', next_step_gas_mass[i, j, 0:n_x], i, j)
    return next_step_gas_mass


@njit
def set_matrices_lhs_z_sweep_de_periodic(n_z, j, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature, simulate_region):
    for i in range(0, n_z):
        if (temperature[i][j][k] > 0 or simulate_region[i][j][k] == 1) and np.sum(surface[i][j][k]) == 0 and diag[i] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_t = 1 / 2 * Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = 1 / 2 * Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            sub_alpha[i] = - a_t
            sub_gamma[i] = - a_b
            if i < n_z-1 and temperature[i+1][j][k] == 0:
                a_t = 0
            diag[i] = a_t + a_b + dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif temperature[i][j][k] == 0 and diag[i] == 0:
            diag[i] = dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif np.sum(surface[i][j][k]) != 0 and diag[i] == 0:
            diag[i] = dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_z_sweep_de_periodic(n_z, j, k, rhs, gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, n_x, n_y, temperature, simulate_region):
    x_neg_periodic, x_pos_periodic, y_neg_periodic, y_pos_periodic = 0, 0, 0, 0
    if k == 1:
        x_neg_periodic = 1
    if k == n_x - 2:
        x_pos_periodic = 1
    if j == 1:
        y_neg_periodic = 1
    if j == n_y - 2:
        y_pos_periodic = 1
    for i in range(0, n_z):
        if (temperature[i][j][k] > 0 or simulate_region[i][j][k] == 1) and np.sum(surface[i][j][k]) == 0 and rhs[i] == 0:
            a_t = 1 / 2 * Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = 1 / 2 * Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = Diffusion_coefficient[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = Diffusion_coefficient[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            if i < n_z-1 and gas_mass[i+1][j][k] == 0:
                a_t = 0
            rhs[i] = a_t * gas_mass[i+1][j][k] + a_b * gas_mass[i-1][j][k] + a_n * (gas_mass[i][j+1][k] * (1 - y_pos_periodic) + gas_mass[i][1][k] * y_pos_periodic) + a_s * (gas_mass[i][j-1][k] * (1 - y_neg_periodic) + gas_mass[i][n_y-2][k] * y_neg_periodic) + a_e * (gas_mass[i][j][k+1] * (1 - x_pos_periodic) + gas_mass[i][j][1] * x_pos_periodic) + a_w * (gas_mass[i][j][k-1] * (1 - x_neg_periodic) + gas_mass[i][j][n_x-2] * x_neg_periodic) + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s - a_w - a_e) * gas_mass[i][j][k]
        elif temperature[i][j][k] == 0 and rhs[i] == 0:
            rhs[i] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[i] == 0:
            rhs[i] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
    return rhs


@njit
def boundary_condition_implicit_z_sweep_de_periodic(dt, gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero, n_x, n_y):
    for each in surface_reduced:
        x_neg_periodic, x_pos_periodic, y_neg_periodic, y_pos_periodic = 0, 0, 0, 0
        if each[0] == 1:
            x_neg_periodic = 1
        if each[0] == n_x - 2:
            x_pos_periodic = 1
        if each[1] == 1:
            y_neg_periodic = 1
        if each[1] == n_y - 2:
            y_pos_periodic = 1
        if not top_layer_zero:
            a_t = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[2]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_t + a_b + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[2]] = - a_t
            sub_gamma[each[2]] = - a_b
            rhs[each[2]] = a_t * gas_mass[each[2] + 1][each[1]][each[0]] + a_b * gas_mass[each[2] - 1][each[1]][each[0]] + a_n * (gas_mass[each[2]][each[1] + 1][each[0]] * (1 - y_pos_periodic) + gas_mass[each[2]][1][each[0]] * y_pos_periodic) + a_s * (gas_mass[each[2]][each[1] - 1][each[0]] * (1 - y_neg_periodic) + gas_mass[each[2]][n_y-2][each[0]] * y_neg_periodic) + a_e * (gas_mass[each[2]][each[1]][each[0] + 1] * (1- x_pos_periodic) + gas_mass[each[2]][each[1]][1] * x_pos_periodic) + a_w * (gas_mass[each[2]][each[1]][each[0] - 1] * (1- x_neg_periodic) + gas_mass[each[2]][each[1]][n_x-2] * x_neg_periodic) + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_w - a_e) * gas_mass[each[2]][each[1]][each[0]]
        else:
            diag[each[2]] = dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            rhs[each[2]] = 0
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_rhs_y_sweep_de_periodic(n_y, i, k, rhs, gas_mass, z_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, n_x, temperature, simulate_region):
    x_neg_periodic, x_pos_periodic = 0, 0
    if k == 1:
        x_neg_periodic = 1
    if k == n_x - 2:
        x_pos_periodic = 1
    for j in range(0, n_y):
        y_neg_periodic, y_pos_periodic = 0, 0
        if j == 1:
            y_neg_periodic = 1
        if j == n_y - 2:
            y_pos_periodic = 1
        if (temperature[i][j][k] > 0 or simulate_region[i][j][k] == 1) and np.sum(surface[i][j][k]) == 0 and rhs[j] == 0:
            a_t = Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = 1 / 2 * Diffusion_coefficient[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = 1 / 2 * Diffusion_coefficient[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            rhs[j] = a_t * (z_sweep_gas_mass[i+1][j][k] + gas_mass[i+1][j][k])/2 + a_b * (z_sweep_gas_mass[i-1][j][k] + gas_mass[i-1][j][k])/2 + a_n * (gas_mass[i][j+1][k] * (1 - y_pos_periodic) + gas_mass[i][1][k] * y_pos_periodic) + a_s * (gas_mass[i][j-1][k] * (1 - y_neg_periodic) + gas_mass[i][n_y-2][k] * y_neg_periodic) + a_e * (gas_mass[i][j][k+1] * (1 - x_pos_periodic) + gas_mass[i][j][1] * x_pos_periodic) + a_w * (gas_mass[i][j][k-1] * (1 - x_neg_periodic) + gas_mass[i][j][n_x-2] * x_neg_periodic) + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_n - a_s - a_w - a_e) * gas_mass[i][j][k] - (a_t + a_b) * (z_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2
        elif temperature[i][j][k] == 0 and rhs[j] == 0:
            rhs[j] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[j] == 0:
            rhs[j] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
    return rhs


@njit
def boundary_condition_implicit_y_sweep_de_periodic(dt, gas_mass, z_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero, n_x, n_y):
    for each in surface_reduced:
        x_neg_periodic, x_pos_periodic, y_neg_periodic, y_pos_periodic = 0, 0, 0, 0
        if each[0] == 1:
           x_neg_periodic = 1
        if each[0] == n_x-2:
            x_pos_periodic = 1
        if each[1] == 1:
            y_neg_periodic = 1
        if each[1] == n_y-2:
            y_pos_periodic = 1
        if not top_layer_zero:
            a_t = Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[1]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] +  a_n + a_s + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[1]] = - a_n
            sub_gamma[each[1]] = - a_s
            rhs[each[1]] = a_t * (z_sweep_gas_mass[each[2] + 1][each[1]][each[0]] + gas_mass[each[2] + 1][each[1]][each[0]])/2 + a_b * (z_sweep_gas_mass[each[2] - 1][each[1]][each[0]] + gas_mass[each[2] - 1][each[1]][each[0]])/2 + a_n * (gas_mass[each[2]][each[1] + 1][each[0]] * (1 - y_pos_periodic) + gas_mass[each[2]][1][each[0]] * y_pos_periodic) + a_s * (gas_mass[each[2]][each[1] - 1][each[0]] * (1 - y_neg_periodic) + gas_mass[each[2]][n_y-2][each[0]] * y_neg_periodic) + a_e * (gas_mass[each[2]][each[1]][each[0] + 1] * (1 - x_pos_periodic) + gas_mass[each[2]][each[1]][1] * x_pos_periodic) + a_w * (gas_mass[each[2]][each[1]][each[0] - 1] * (1 - x_neg_periodic) + gas_mass[each[2]][each[1]][n_x-2] * x_neg_periodic) + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_n - a_s - a_w - a_e) * gas_mass[each[2]][each[1]][each[0]] - (a_t + a_b) * (z_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2
        else:
            diag[each[1]] = dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            rhs[each[1]] = 0
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_rhs_x_sweep_de_periodic(n_x, i, j, rhs, gas_mass, z_sweep_gas_mass, y_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, n_y, temperature, simulate_region):
    y_neg_periodic, y_pos_periodic = 0, 0
    if j == 1:
        y_neg_periodic = 1
    if j == n_y - 2:
        y_pos_periodic = 1
    for k in range(0, n_x):
        x_neg_periodic, x_pos_periodic = 0, 0
        if k == 1:
            x_neg_periodic = 1
        if k == n_x - 2:
            x_pos_periodic = 1
        if (temperature[i][j][k] > 0 or simulate_region[i][j][k] == 1) and np.sum(surface[i][j][k]) == 0 and rhs[k] == 0:
            a_t = Diffusion_coefficient[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0])
            a_b = Diffusion_coefficient[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1])
            a_n = Diffusion_coefficient[i][j][k][2] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2])
            a_s = Diffusion_coefficient[i][j][k][3] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3])
            a_e = 1 / 2 * Diffusion_coefficient[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4])
            a_w = 1 / 2 * Diffusion_coefficient[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5])
            rhs[k] = a_t * (z_sweep_gas_mass[i+1][j][k] + gas_mass[i+1][j][k])/2 + a_b * (z_sweep_gas_mass[i-1][j][k] + gas_mass[i-1][j][k])/2 + a_n * ((y_sweep_gas_mass[i][j+1][k] + gas_mass[i][j+1][k])/2 * (1 - y_pos_periodic) + (y_sweep_gas_mass[i][1][k] + gas_mass[i][1][k])/2 * y_pos_periodic) + a_s * ((y_sweep_gas_mass[i][j-1][k] + gas_mass[i][j-1][k])/2 * (1 - y_neg_periodic) + (y_sweep_gas_mass[i][n_y-2][k] + gas_mass[i][n_y-2][k])/2 * y_neg_periodic) + a_e * (gas_mass[i][j][k+1] * (1 - x_pos_periodic) + gas_mass[i][j][1] * x_pos_periodic) + a_w * (gas_mass[i][j][k-1] * (1 - x_neg_periodic) + gas_mass[i][j][n_x-2] * x_neg_periodic) + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_e - a_w) * gas_mass[i][j][k] - (a_t + a_b) * (z_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2 - (a_n + a_s) * (y_sweep_gas_mass[i][j][k] + gas_mass[i][j][k])/2
        elif temperature[i][j][k] == 0 and rhs[k] == 0:
            rhs[k] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
        elif np.sum(surface[i][j][k]) != 0 and rhs[k] == 0:
            rhs[k] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt) * gas_mass[i][j][k]
    return rhs


@njit
def boundary_condition_implicit_x_sweep_de_periodic(dt, gas_mass, z_sweep_gas_mass, y_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero, n_x, n_y):
    for each in surface_reduced:
        x_neg_periodic, x_pos_periodic, y_neg_periodic, y_pos_periodic = 0, 0, 0, 0
        if each[0] == 1:
            x_neg_periodic = 1
        if each[0] == n_x - 2:
            x_pos_periodic = 1
        if each[1] == 1:
            y_neg_periodic = 1
        if each[1] == n_y - 2:
            y_pos_periodic = 1
        if not top_layer_zero:
            a_t = Diffusion_coefficient[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][0])
            a_b = Diffusion_coefficient[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][1])
            a_n = Diffusion_coefficient[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][2])
            a_s = Diffusion_coefficient[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][3])
            a_e = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][4])
            a_w = 1 / 2 * Diffusion_coefficient[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5]) * (1 - sh_adjacent_voxels[each[2]][each[1]][each[0]][5])
            diag[each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + a_e + a_w + dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            sub_alpha[each[0]] = - a_e
            sub_gamma[each[0]] = - a_w
            rhs[each[0]] = a_t * (z_sweep_gas_mass[each[2] + 1][each[1]][each[0]] + gas_mass[each[2] + 1][each[1]][each[0]])/2 + a_b * (z_sweep_gas_mass[each[2] - 1][each[1]][each[0]] + gas_mass[each[2] - 1][each[1]][each[0]])/2 + a_n * ((y_sweep_gas_mass[each[2]][each[1] + 1][each[0]] + gas_mass[each[2]][each[1] + 1][each[0]])/2 * (1 - y_pos_periodic) + (y_sweep_gas_mass[each[2]][1][each[0]] + gas_mass[each[2]][1][each[0]])/2 * y_pos_periodic) + a_s * ((y_sweep_gas_mass[each[2]][each[1] - 1][each[0]] + gas_mass[each[2]][each[1] - 1][each[0]])/2 * (1 - y_neg_periodic) + (y_sweep_gas_mass[each[2]][n_y-2][each[0]] + gas_mass[each[2]][n_y-2][each[0]])/2 * y_neg_periodic) + a_e * (gas_mass[each[2]][each[1]][each[0] + 1] * (1 - x_pos_periodic) + gas_mass[each[2]][each[1]][1] * x_pos_periodic) + a_w * (gas_mass[each[2]][each[1]][each[0] - 1] * (1 - x_neg_periodic) + gas_mass[each[2]][each[1]][n_x-2] * x_neg_periodic) + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_e - a_w) * gas_mass[each[2]][each[1]][each[0]] - (a_t + a_b) * (z_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2 - (a_n + a_s) * (y_sweep_gas_mass[each[2]][each[1]][each[0]] + gas_mass[each[2]][each[1]][each[0]])/2
        else:
            diag[each[0]] = dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
            rhs[each[0]] = 0
    return sub_alpha, diag, sub_gamma, rhs


@njit
def de_implicit_DGADI_periodic(n_x, n_y, n_z, surface_reduced, dt, gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, S_c, S_p, sh_adjacent_voxels, top_layer_zero, temperature, simulate_region):
    next_step_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    z_sweep_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    y_sweep_gas_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for j in range(1, n_y-1):
        for k in range(1, n_x-1):
            sub_alpha = np.zeros(n_z, dtype=np.float64)
            diag = np.zeros(n_z, dtype=np.float64)
            sub_gamma = np.zeros(n_z, dtype=np.float64)
            rhs = np.zeros(n_z, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[1] == j and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_z_sweep_de_periodic(dt, gas_mass,  Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero, n_x, n_y)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_z_sweep_de_periodic(n_z, j, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature, simulate_region)
            rhs = set_matrices_rhs_z_sweep_de_periodic(n_z, j, k, rhs, gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, n_x, n_y, temperature, simulate_region)
            z_sweep_gas_mass[1:n_z-1, j, k] = tridiagonal_matrix_solver(n_z-2, diag[1:n_z-1], sub_gamma[1:n_z-1], sub_alpha[1:n_z-1], rhs[1:n_z-1])
            '''if j == 12 and k == 12:
                print(sub_gamma)
                print(diag)
                print(sub_alpha)
                print(rhs)
                print('z:', z_sweep_gas_mass[1:n_z-1, j, k])'''
    #print('z:', np.sum(z_sweep_gas_mass * dx * dy * dz))
    #print(z_sweep_gas_mass[1:24, 12, 12])
    for i in range(1, n_z-1):
        for k in range(1, n_x-1):
            sub_alpha = np.zeros(n_y, dtype=np.float64)
            diag = np.zeros(n_y, dtype=np.float64)
            sub_gamma = np.zeros(n_y, dtype=np.float64)
            rhs = np.zeros(n_y, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_y_sweep_de_periodic(dt, gas_mass, z_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero, n_x, n_y)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_y_sweep_de(n_y, i, k, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature, simulate_region)
            rhs = set_matrices_rhs_y_sweep_de_periodic(n_y, i, k, rhs, gas_mass, z_sweep_gas_mass, surface, dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, n_x, temperature, simulate_region)
            y_sweep_gas_mass[i, 1:n_y-1, k] = periodic_tridiagonal_matrix_solver(n_y-2, diag[1:n_y-1], sub_gamma[1:n_y-1], sub_alpha[1:n_y-1], rhs[1:n_y-1])
    #print('y:', y_sweep_gas_mass[1:n_z-1, 25, 25])
    #print('y:', np.sum(y_sweep_gas_mass * dx * dy * dz))
    #print(y_sweep_gas_mass[1:24, 12, 12])
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            sub_alpha = np.zeros(n_x, dtype=np.float64)
            diag = np.zeros(n_x, dtype=np.float64)
            sub_gamma = np.zeros(n_x, dtype=np.float64)
            rhs = np.zeros(n_x, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[1] == j:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_x_sweep_de_periodic(dt, gas_mass, z_sweep_gas_mass, y_sweep_gas_mass, Diffusion_coefficient, Dr, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, sh_adjacent_voxels, top_layer_zero, n_x, n_y)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_x_sweep_de(n_x, i, j, sub_alpha, diag, sub_gamma, Diffusion_coefficient, dx, dy, dz, Dr, dt, S_p, gas_mass, surface, sh_adjacent_voxels, temperature, simulate_region)
            rhs = set_matrices_rhs_x_sweep_de_periodic(n_x, i, j, rhs, gas_mass, z_sweep_gas_mass, y_sweep_gas_mass, surface,dx, dy, dz, Dr, Diffusion_coefficient, dt, S_c, sh_adjacent_voxels, n_y, temperature, simulate_region)
            next_step_gas_mass[i, j, 1:n_x-1] = periodic_tridiagonal_matrix_solver(n_x-2, diag[1:n_x-1], sub_gamma[1:n_x-1], sub_alpha[1:n_x-1], rhs[1:n_x-1])
    #print('nxt:', np.sum(next_step_gas_mass * dx * dy * dz))
    #print('nxt:', next_step_gas_mass[1:n_z - 1, 25, 25])
    #print(next_step_gas_mass[1:24, 12, 12])
    return next_step_gas_mass