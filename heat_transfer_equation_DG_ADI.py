import numpy as np
from numba import jit, njit, prange
import scipy
from tri_diag_solve import tridiagonal_matrix_solver, periodic_tridiagonal_matrix_solver


@njit
def set_matrices_lhs_z_sweep(n_z, j, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[i] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_t = 1 / 2 * Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][0]
            a_b = 1 / 2 * Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][1]
            sub_alpha[i] = - a_t
            sub_gamma[i] = - a_b
            diag[i] = a_t + a_b + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[i] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[i] = 1
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_z_sweep(n_z, j, k, rhs, temperature, x_sweep_temperature, y_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[i] == 0:
            a_t = 1 / 2 * Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
            a_b = 1 / 2 * Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
            a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
            rhs[i] = a_t * temperature[i+1][j][k] + a_b * temperature[i-1][j][k] + a_n * (y_sweep_temperature[i][j+1][k] + temperature[i][j+1][k])/2 + a_s * (y_sweep_temperature[i][j-1][k] + temperature[i][j-1][k])/2 + a_e * (x_sweep_temperature[i][j][k+1] + temperature[i][j][k+1])/2 + a_w * (x_sweep_temperature[i][j][k-1] + temperature[i][j][k-1])/2 + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b) * temperature[i][j][k] - (a_n + a_s) * (y_sweep_temperature[i][j][k] + temperature[i][j][k])/2 - (a_w + a_e) * (x_sweep_temperature[i][j][k] + temperature[i][j][k])/2
        elif sample_holder[i][j][k] != 0:
            rhs[i] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[i] = 0
    return rhs


@njit
def set_matrices_rhs_z_sweep_zfirst(n_z, j, k, rhs, temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[i] == 0:
            a_t = 1 / 2 * Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
            a_b = 1 / 2 * Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
            a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
            rhs[i] = a_t * temperature[i+1][j][k] + a_b * temperature[i-1][j][k] + a_n * temperature[i][j+1][k] + a_s * temperature[i][j-1][k] + a_e * temperature[i][j][k+1] + a_w * temperature[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s - a_w - a_e) * temperature[i][j][k]
        elif sample_holder[i][j][k] != 0:
            rhs[i] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[i] = 0
    return rhs


@njit
def boundary_condition_implicit_z_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, x_sweep_temperature, y_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature):
    for each in surface_reduced:
        S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        if S_p_rad > 0:
            S_p_rad = 0
            S_c_rad = - epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = 1 / 2 * Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = 1 / 2 * Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[2]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_t + a_b + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[2]] = - a_t
        sub_gamma[each[2]] = - a_b
        rhs[each[2]] = a_t * temperature[each[2] + 1][each[1]][each[0]] + a_b * temperature[each[2] - 1][each[1]][each[0]] + a_n * (y_sweep_temperature[each[2]][each[1] + 1][each[0]] + temperature[each[2]][each[1] + 1][each[0]])/2 + a_s * (y_sweep_temperature[each[2]][each[1] - 1][each[0]] + temperature[each[2]][each[1] - 1][each[0]])/2 + a_e * (x_sweep_temperature[each[2]][each[1]][each[0] + 1] + temperature[each[2]][each[1]][each[0] + 1])/2 + a_w * (x_sweep_temperature[each[2]][each[1]][each[0] - 1] + temperature[each[2]][each[1]][each[0] - 1])/2 + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b) * temperature[each[2]][each[1]][each[0]] - (a_n + a_s) * (y_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2 - (a_w + a_e) * (x_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2
    return sub_alpha, diag, sub_gamma, rhs


@njit
def boundary_condition_implicit_z_sweep_zfirst(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature):
    for each in surface_reduced:
        #S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        #S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_c_rad = 3 * epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[0]**4) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[2]**4) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[4]**4))
        S_p_rad = -4 * epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[0]**3) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[2]**3) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[4]**3))
        if S_p_rad > 0:
            S_p_rad = 0
            #S_c_rad = - epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - 300**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
            S_c_rad = - epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[0]**4) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[2]**4) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[4]**4))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = 1 / 2 * Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = 1 / 2 * Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[2]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_t + a_b + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[2]] = - a_t
        sub_gamma[each[2]] = - a_b
        rhs[each[2]] = a_t * temperature[each[2] + 1][each[1]][each[0]] + a_b * temperature[each[2] - 1][each[1]][each[0]] + a_n * temperature[each[2]][each[1] + 1][each[0]] + a_s * temperature[each[2]][each[1] - 1][each[0]] + a_e * temperature[each[2]][each[1]][each[0] + 1] + a_w * temperature[each[2]][each[1]][each[0] - 1] + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_w - a_e) * temperature[each[2]][each[1]][each[0]]
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_lhs_y_sweep(n_y, i, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder):
    for j in range(0, n_y):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[j] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_n = 1 / 2 * Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][2]
            a_s = 1 / 2 * Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][3]
            sub_alpha[j] = - a_n
            sub_gamma[j] = - a_s
            diag[j] = a_n + a_s + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[j] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[j] = 1
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_y_sweep(n_y, i, k, rhs, temperature, x_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for j in range(0, n_y):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[j] == 0:
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
            a_n = 1 / 2 * Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
            a_s = 1 / 2 * Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
            rhs[j] = a_t * temperature[i+1][j][k] + a_b * temperature[i-1][j][k] + a_n * temperature[i][j+1][k] + a_s * temperature[i][j-1][k] + a_e * (x_sweep_temperature[i][j][k+1] + temperature[i][j][k+1])/2 + a_w * (x_sweep_temperature[i][j][k-1] + temperature[i][j][k-1])/2 + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s) * temperature[i][j][k] - (a_w + a_e) * (x_sweep_temperature[i][j][k] + temperature[i][j][k])/2
        elif sample_holder[i][j][k] != 0:
            rhs[j] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[j] = 0
    return rhs


@njit
def set_matrices_rhs_y_sweep_zfirst(n_y, i, k, rhs, temperature, z_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for j in range(0, n_y):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[j] == 0:
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
            a_n = 1 / 2 * Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
            a_s = 1 / 2 * Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
            rhs[j] = a_t * (z_sweep_temperature[i+1][j][k] + temperature[i+1][j][k])/2 + a_b * (z_sweep_temperature[i-1][j][k] + temperature[i-1][j][k])/2 + a_n * temperature[i][j+1][k] + a_s * temperature[i][j-1][k] + a_e * temperature[i][j][k+1] + a_w * temperature[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_n - a_s - a_w - a_e) * temperature[i][j][k] - (a_t + a_b) * (z_sweep_temperature[i][j][k] + temperature[i][j][k])/2
        elif sample_holder[i][j][k] != 0:
            rhs[j] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[j] = 0
    return rhs


@njit
def boundary_condition_implicit_y_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, x_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature):
    for each in surface_reduced:
        S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        #S_c_rad = 3 * epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[0]**4) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[2]**4) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[4]**4))
        #S_p_rad = -4 * epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[0]**3) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[0]**3) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[0]**3))
        if S_p_rad > 0:
            S_p_rad = 0
            S_c_rad = - epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = 1 / 2 * Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = 1 / 2 * Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[1]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad +  a_n + a_s + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[1]] = - a_n
        sub_gamma[each[1]] = - a_s
        rhs[each[1]] = a_t * temperature[each[2] + 1][each[1]][each[0]] + a_b * temperature[each[2] - 1][each[1]][each[0]] + a_n * temperature[each[2]][each[1] + 1][each[0]] + a_s * temperature[each[2]][each[1] - 1][each[0]] + a_e * (x_sweep_temperature[each[2]][each[1]][each[0] + 1] + temperature[each[2]][each[1]][each[0] + 1])/2 + a_w * (x_sweep_temperature[each[2]][each[1]][each[0] - 1] + temperature[each[2]][each[1]][each[0] - 1])/2 + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s) * temperature[each[2]][each[1]][each[0]] - (a_w + a_e) * (x_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2
    return sub_alpha, diag, sub_gamma, rhs


@njit
def boundary_condition_implicit_y_sweep_zfirst(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, z_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature):
    for each in surface_reduced:
        #S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        #S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_c_rad = 3 * epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[0]**4) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[2]**4) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[4]**4))
        S_p_rad = -4 * epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[0]**3) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[2]**3) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[4]**3))
        if S_p_rad > 0:
            S_p_rad = 0
            #S_c_rad = epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - 300**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
            S_c_rad = - epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[0]**4) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[2]**4) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[4]**4))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = 1 / 2 * Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = 1 / 2 * Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[1]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad +  a_n + a_s + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[1]] = - a_n
        sub_gamma[each[1]] = - a_s
        rhs[each[1]] = a_t * (z_sweep_temperature[each[2] + 1][each[1]][each[0]] + temperature[each[2] + 1][each[1]][each[0]])/2 + a_b * (z_sweep_temperature[each[2] - 1][each[1]][each[0]] + temperature[each[2] - 1][each[1]][each[0]])/2 + a_n * temperature[each[2]][each[1] + 1][each[0]] + a_s * temperature[each[2]][each[1] - 1][each[0]] + a_e * temperature[each[2]][each[1]][each[0] + 1] + a_w * temperature[each[2]][each[1]][each[0] - 1] + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_n - a_s - a_w - a_e) * temperature[each[2]][each[1]][each[0]] - (a_t + a_b) * (z_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_lhs_x_sweep(n_x, i, j, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder):
    for k in range(0, n_x):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[k] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_e = 1/2 * Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][4]
            a_w = 1/2 * Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][5]
            sub_alpha[k] = - a_e
            sub_gamma[k] = - a_w
            diag[k] = a_e + a_w + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[k] = 1
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_x_sweep(n_x, i, j, rhs, temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for k in range(0, n_x):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[k] == 0:
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
            a_n = Lambda[i][j][k][2] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
            a_e = 1 / 2 * Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
            a_w = 1 / 2 * Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
            rhs[k] = a_t * temperature[i+1][j][k] + a_b * temperature[i-1][j][k] + a_n * temperature[i][j+1][k] + a_s * temperature[i][j-1][k] + a_e * temperature[i][j][k+1] + a_w * temperature[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s - a_e - a_w) * temperature[i][j][k]
        elif sample_holder[i][j][k] != 0:
            rhs[k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[k] = 0
    return rhs


@njit
def set_matrices_rhs_x_sweep_zfirst(n_x, i, j, rhs, temperature, z_sweep_temperature, y_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for k in range(0, n_x):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[k] == 0:
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
            a_n = Lambda[i][j][k][2] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
            a_e = 1 / 2 * Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
            a_w = 1 / 2 * Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
            rhs[k] = a_t * (z_sweep_temperature[i+1][j][k] + temperature[i+1][j][k])/2 + a_b * (z_sweep_temperature[i-1][j][k] + temperature[i-1][j][k])/2 + a_n * (y_sweep_temperature[i][j+1][k] + temperature[i][j+1][k])/2 + a_s * (y_sweep_temperature[i][j-1][k] + temperature[i][j-1][k])/2 + a_e * temperature[i][j][k+1] + a_w * temperature[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_e - a_w) * temperature[i][j][k] - (a_t + a_b) * (z_sweep_temperature[i][j][k] + temperature[i][j][k])/2 - (a_n + a_s) * (y_sweep_temperature[i][j][k] + temperature[i][j][k])/2
        elif sample_holder[i][j][k] != 0:
            rhs[k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[k] = 0
    return rhs


@njit
def boundary_condition_implicit_x_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature):
    for each in surface_reduced:
        S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        if S_p_rad > 0:
            S_p_rad = 0
            S_c_rad = - epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = 1 / 2 * Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = 1 / 2 * Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_e + a_w + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[0]] = - a_e
        sub_gamma[each[0]] = - a_w
        rhs[each[0]] = a_t * temperature[each[2] + 1][each[1]][each[0]] + a_b * temperature[each[2] - 1][each[1]][each[0]] + a_n * temperature[each[2]][each[1] + 1][each[0]] + a_s * temperature[each[2]][each[1] - 1][each[0]]  + a_e * temperature[each[2]][each[1]][each[0] + 1] + a_w * temperature[each[2]][each[1]][each[0] - 1] + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_e - a_w) * temperature[each[2]][each[1]][each[0]]
    return sub_alpha, diag, sub_gamma, rhs


@njit
def boundary_condition_implicit_x_sweep_zfirst(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, z_sweep_temperature, y_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature):
    for each in surface_reduced:
        #S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        #S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_c_rad = 3 * epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[0]**4) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[2]**4) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[4]**4))
        S_p_rad = -4 * epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[0]**3) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[2]**3) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature[4]**3))
        if S_p_rad > 0:
            S_p_rad = 0
            #S_c_rad = epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - 300**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
            S_c_rad = - epsilon * sigma * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[0]**4) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[2]**4) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]) * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature[4]**4))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = 1 / 2 * Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = 1 / 2 * Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_e + a_w + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[0]] = - a_e
        sub_gamma[each[0]] = - a_w
        rhs[each[0]] = a_t * (z_sweep_temperature[each[2] + 1][each[1]][each[0]] + temperature[each[2] + 1][each[1]][each[0]])/2 + a_b * (z_sweep_temperature[each[2] - 1][each[1]][each[0]] + temperature[each[2] - 1][each[1]][each[0]])/2 + a_n * (y_sweep_temperature[each[2]][each[1] + 1][each[0]] + temperature[each[2]][each[1] + 1][each[0]])/2 + a_s * (y_sweep_temperature[each[2]][each[1] - 1][each[0]] + temperature[each[2]][each[1] - 1][each[0]])/2  + a_e * temperature[each[2]][each[1]][each[0] + 1] + a_w * temperature[each[2]][each[1]][each[0] - 1] + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_e - a_w) * temperature[each[2]][each[1]][each[0]] - (a_t + a_b) * (z_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2 - (a_n + a_s) * (y_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2
    return sub_alpha, diag, sub_gamma, rhs


@njit
def hte_implicit_DGADI(n_x, n_y, n_z, surface_reduced, r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder, ambient_temperature):
    next_step_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    x_sweep_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    y_sweep_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_x_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_x_sweep(n_x, i, j, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_x_sweep(n_x, i, j, rhs, temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            x_sweep_temperature[i, j, 0:n_x] = tridiagonal_matrix_solver(n_x, diag, sub_gamma, sub_alpha, rhs)
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_y_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, x_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_y_sweep(n_y, i, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_y_sweep(n_y, i, k, rhs, temperature, x_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            y_sweep_temperature[i, 0:n_y, k] = tridiagonal_matrix_solver(n_y, diag, sub_gamma, sub_alpha, rhs)
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_z_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, x_sweep_temperature, y_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_z_sweep(n_z, j, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_z_sweep(n_z, j, k, rhs, temperature, x_sweep_temperature, y_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            next_step_temperature[0:n_z, j, k] = tridiagonal_matrix_solver(n_z, diag, sub_gamma, sub_alpha, rhs)
    return next_step_temperature


@njit
def hte_implicit_DGADI_zfirst(n_x, n_y, n_z, surface_reduced, r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder, ambient_temperature):
    next_step_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    z_sweep_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    y_sweep_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_z_sweep_zfirst(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_z_sweep(n_z, j, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_z_sweep_zfirst(n_z, j, k, rhs, temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            z_sweep_temperature[0:n_z, j, k] = tridiagonal_matrix_solver(n_z, diag, sub_gamma, sub_alpha, rhs)
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_y_sweep_zfirst(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, z_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_y_sweep(n_y, i, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_y_sweep_zfirst(n_y, i, k, rhs, temperature, z_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            y_sweep_temperature[i, 0:n_y, k] = tridiagonal_matrix_solver(n_y, diag, sub_gamma, sub_alpha, rhs)
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_x_sweep_zfirst(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, z_sweep_temperature, y_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_x_sweep(n_x, i, j, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_x_sweep_zfirst(n_x, i, j, rhs, temperature, z_sweep_temperature, y_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            next_step_temperature[i, j, 0:n_x] = tridiagonal_matrix_solver(n_x, diag, sub_gamma, sub_alpha, rhs)
    return next_step_temperature


@njit
def set_matrices_lhs_z_sweep_periodic(n_z, j, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder, inner_structure):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[i] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_t = 1 / 2 * Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][0] * (1 - inner_structure[i][j][k][0])
            a_b = 1 / 2 * Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][1] * (1 - inner_structure[i][j][k][1])
            sub_alpha[i] = - a_t
            sub_gamma[i] = - a_b
            if i < n_z - 1 and temperature[i + 1][j][k] == 0:
                a_t = 0
            diag[i] = a_t + a_b + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[i] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[i] = 1
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_z_sweep_periodic(n_z, j, k, rhs, temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c, n_x, n_y, inner_structure):
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
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[i] == 0:
            a_t = 1 / 2 * Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - inner_structure[i][j][k][0])
            a_b = 1 / 2 * Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - inner_structure[i][j][k][1])
            a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - inner_structure[i][j][k][2])
            a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - inner_structure[i][j][k][3])
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - inner_structure[i][j][k][4])
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - inner_structure[i][j][k][5])
            if i < n_z - 1 and temperature[i + 1][j][k] == 0:
                a_t = 0
            rhs[i] = a_t * temperature[i+1][j][k] + a_b * temperature[i-1][j][k] + a_n * (temperature[i][j+1][k] * (1 - y_pos_periodic) + temperature[i][1][k] * y_pos_periodic) + a_s * (temperature[i][j-1][k] * (1 - y_neg_periodic) + temperature[i][n_y-2][k] * y_neg_periodic) + a_e * (temperature[i][j][k+1] * (1 - x_pos_periodic) + temperature[i][j][1] * x_pos_periodic) + a_w * (temperature[i][j][k-1] * (1 - x_neg_periodic) + temperature[i][j][n_x-2] * x_neg_periodic) + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s - a_w - a_e) * temperature[i][j][k]
        elif sample_holder[i][j][k] != 0:
            rhs[i] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[i] = 0
    return rhs


#@njit
def boundary_condition_implicit_z_sweep_periodic(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, n_x, n_y, inner_structure, ambient_temperature):
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
        S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        if S_p_rad > 0:
            S_p_rad = 0
            S_c_rad = - epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = 1 / 2 * Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0]) * (1 - inner_structure[each[2]][each[1]][each[0]][0])
        a_b = 1 / 2 * Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1]) * (1 - inner_structure[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2]) * (1 - inner_structure[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3]) * (1 - inner_structure[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4]) * (1 - inner_structure[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5]) * (1 - inner_structure[each[2]][each[1]][each[0]][5])
        diag[each[2]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_t + a_b + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[2]] = - a_t
        sub_gamma[each[2]] = - a_b
        rhs[each[2]] = a_t * temperature[each[2] + 1][each[1]][each[0]] + a_b * temperature[each[2] - 1][each[1]][each[0]] + a_n * (temperature[each[2]][each[1] + 1][each[0]] * (1 - y_pos_periodic) + temperature[each[2]][1][each[0]] * y_pos_periodic)+ a_s * (temperature[each[2]][each[1] - 1][each[0]] * (1 - y_neg_periodic) + temperature[each[2]][n_y-2][each[0]] * y_neg_periodic) + a_e * (temperature[each[2]][each[1]][each[0] + 1] * (1 - x_pos_periodic) + temperature[each[2]][each[1]][1] * x_pos_periodic) + a_w * (temperature[each[2]][each[1]][each[0] - 1] * (1 - x_neg_periodic) + temperature[each[2]][each[1]][n_x-2] * x_neg_periodic) + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s - a_w - a_e) * temperature[each[2]][each[1]][each[0]]
        '''if each[2] == 1:
            print(each[0], each[1], each[2])
            print(x_pos_periodic, x_neg_periodic, y_pos_periodic, y_neg_periodic)'''
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_lhs_y_sweep_periodic(n_y, i, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder, inner_structure):
    for j in range(0, n_y):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[j] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_n = 1 / 2 * Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][2] * (1 - inner_structure[i][j][k][2])
            a_s = 1 / 2 * Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][3] * (1 - inner_structure[i][j][k][3])
            sub_alpha[j] = - a_n
            sub_gamma[j] = - a_s
            diag[j] = a_n + a_s + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[j] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[j] = 1
    return sub_alpha, diag, sub_gamma

@njit
def set_matrices_rhs_y_sweep_periodic(n_y, i, k, rhs, temperature, z_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c, n_x, inner_structure):
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
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[j] == 0:
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - inner_structure[i][j][k][0])
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - inner_structure[i][j][k][1])
            a_n = 1 / 2 * Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - inner_structure[i][j][k][2])
            a_s = 1 / 2 * Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - inner_structure[i][j][k][3])
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - inner_structure[i][j][k][4])
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - inner_structure[i][j][k][5])
            rhs[j] = a_t * (z_sweep_temperature[i+1][j][k] + temperature[i+1][j][k])/2 + a_b * (z_sweep_temperature[i-1][j][k] + temperature[i-1][j][k])/2 + a_n * (temperature[i][j+1][k] * (1 - y_pos_periodic) + temperature[i][1][k] * y_pos_periodic) + a_s * (temperature[i][j-1][k] * (1 - y_neg_periodic) + temperature[i][n_y-2][k] * y_neg_periodic) + a_e * (temperature[i][j][k+1] * (1 - x_pos_periodic) + temperature[i][j][1] * x_pos_periodic) + a_w * (temperature[i][j][k-1] * (1 - x_neg_periodic) + temperature[i][j][n_x-2] * x_neg_periodic) + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_n - a_s - a_w - a_e) * temperature[i][j][k] - (a_t + a_b) * (z_sweep_temperature[i][j][k] + temperature[i][j][k])/2
        elif sample_holder[i][j][k] != 0:
            rhs[j] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[j] = 0
    return rhs


@njit
def boundary_condition_implicit_y_sweep_periodic(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, z_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, n_x, n_y, inner_structure, ambient_temperature):
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
        S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        if S_p_rad > 0:
            S_p_rad = 0
            S_c_rad = - epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0]) * (1 - inner_structure[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1]) * (1 - inner_structure[each[2]][each[1]][each[0]][1])
        a_n = 1 / 2 * Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2]) * (1 - inner_structure[each[2]][each[1]][each[0]][2])
        a_s = 1 / 2 * Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3]) * (1 - inner_structure[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4]) * (1 - inner_structure[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5]) * (1 - inner_structure[each[2]][each[1]][each[0]][5])
        diag[each[1]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad +  a_n + a_s + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[1]] = - a_n
        sub_gamma[each[1]] = - a_s
        rhs[each[1]] = a_t * (z_sweep_temperature[each[2] + 1][each[1]][each[0]] + temperature[each[2] + 1][each[1]][each[0]])/2 + a_b * (z_sweep_temperature[each[2] - 1][each[1]][each[0]] + temperature[each[2] - 1][each[1]][each[0]])/2 + a_n * (temperature[each[2]][each[1] + 1][each[0]] * (1 - y_pos_periodic) + temperature[each[2]][1][each[0]] * y_pos_periodic) + a_s * (temperature[each[2]][each[1] - 1][each[0]] * (1 - y_neg_periodic) + temperature[each[2]][n_y-2][each[0]] * y_neg_periodic) + a_e * (temperature[each[2]][each[1]][each[0] + 1] * (1 - x_pos_periodic) + temperature[each[2]][each[1]][1] * x_pos_periodic) + a_w * (temperature[each[2]][each[1]][each[0] - 1] * (1 - x_neg_periodic) + temperature[each[2]][each[1]][n_x-2] * x_neg_periodic) + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_n - a_s - a_w - a_e) * temperature[each[2]][each[1]][each[0]] - (a_t + a_b) * (z_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2
        '''if each[2] == 1:
            print(each[0], each[1], each[2])
            print(x_pos_periodic, x_neg_periodic, y_pos_periodic, y_neg_periodic)'''
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_lhs_x_sweep_periodic(n_x, i, j, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder, inner_structure):
    for k in range(0, n_x):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[k] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_e = 1/2 * Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][4] * (1 - inner_structure[i][j][k][4])
            a_w = 1/2 * Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][5] * (1 - inner_structure[i][j][k][5])
            sub_alpha[k] = - a_e
            sub_gamma[k] = - a_w
            diag[k] = a_e + a_w + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[k] = 1
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_x_sweep_periodic(n_x, i, j, rhs, temperature, z_sweep_temperature, y_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c, n_y, inner_structure):
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
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[k] == 0:
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0] * (1 - inner_structure[i][j][k][0])
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1] * (1 - inner_structure[i][j][k][1])
            a_n = Lambda[i][j][k][2] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][2] * (1 - inner_structure[i][j][k][2])
            a_s = Lambda[i][j][k][3] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][3] * (1 - inner_structure[i][j][k][3])
            a_e = 1 / 2 * Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4] * (1 - inner_structure[i][j][k][4])
            a_w = 1 / 2 * Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5] * (1 - inner_structure[i][j][k][5])
            rhs[k] = a_t * (z_sweep_temperature[i+1][j][k] + temperature[i+1][j][k])/2 + a_b * (z_sweep_temperature[i-1][j][k] + temperature[i-1][j][k])/2 + a_n * ((y_sweep_temperature[i][j+1][k] + temperature[i][j+1][k])/2 * (1 - y_pos_periodic) + (y_sweep_temperature[i][1][k] + temperature[i][1][k])/2 * y_pos_periodic) + a_s * ((y_sweep_temperature[i][j-1][k] + temperature[i][j-1][k])/2 * (1 - y_neg_periodic) + (y_sweep_temperature[i][n_y-2][k] + temperature[i][n_y-2][k])/2 * y_neg_periodic) + a_e * (temperature[i][j][k+1] * (1 - x_pos_periodic) + temperature[i][j][1] * x_pos_periodic) + a_w * (temperature[i][j][k-1] * (1 - x_neg_periodic) + temperature[i][j][n_x-2] * x_neg_periodic) + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_e - a_w) * temperature[i][j][k] - (a_t + a_b) * (z_sweep_temperature[i][j][k] + temperature[i][j][k])/2 - (a_n + a_s) * (y_sweep_temperature[i][j][k] + temperature[i][j][k])/2
        elif sample_holder[i][j][k] != 0:
            rhs[k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[k] = 0
    return rhs


@njit
def boundary_condition_implicit_x_sweep_periodic(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, z_sweep_temperature, y_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p, n_x, n_y, inner_structure, ambient_temperature):
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
        S_c_rad = 3 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**3 - ambient_temperature**3) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        if S_p_rad > 0:
            S_p_rad = 0
            S_c_rad = - epsilon * sigma * (temperature[each[2]][each[1]][each[0]]**4 - ambient_temperature**4) * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0]) * (1 - inner_structure[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1]) * (1 - inner_structure[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2]) * (1 - inner_structure[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3]) * (1 - inner_structure[each[2]][each[1]][each[0]][3])
        a_e = 1 / 2 * Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4]) * (1 - inner_structure[each[2]][each[1]][each[0]][4])
        a_w = 1 / 2 * Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5]) * (1 - inner_structure[each[2]][each[1]][each[0]][5])
        diag[each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_e + a_w + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[0]] = - a_e
        sub_gamma[each[0]] = - a_w
        rhs[each[0]] = a_t * (z_sweep_temperature[each[2] + 1][each[1]][each[0]] + temperature[each[2] + 1][each[1]][each[0]])/2 + a_b * (z_sweep_temperature[each[2] - 1][each[1]][each[0]] + temperature[each[2] - 1][each[1]][each[0]])/2 + a_n * ((y_sweep_temperature[each[2]][each[1] + 1][each[0]] + temperature[each[2]][each[1] + 1][each[0]])/2 * (1 - y_pos_periodic) + (y_sweep_temperature[each[2]][1][each[0]] + temperature[each[2]][1][each[0]])/2 * y_pos_periodic) + a_s * ((y_sweep_temperature[each[2]][each[1] - 1][each[0]] + temperature[each[2]][each[1] - 1][each[0]])/2 * (1 - y_neg_periodic) + (y_sweep_temperature[each[2]][n_y-2][each[0]] + temperature[each[2]][n_y-2][each[0]])/2 * y_neg_periodic) + a_e * (temperature[each[2]][each[1]][each[0] + 1] * (1 - x_pos_periodic) + temperature[each[2]][each[1]][1] * x_pos_periodic) + a_w * (temperature[each[2]][each[1]][each[0] - 1] * (1 - x_neg_periodic) + temperature[each[2]][each[1]][n_x-2] * x_neg_periodic) + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_e - a_w) * temperature[each[2]][each[1]][each[0]] - (a_t + a_b) * (z_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2 - (a_n + a_s) * (y_sweep_temperature[each[2]][each[1]][each[0]] + temperature[each[2]][each[1]][each[0]])/2
        if np.isnan(rhs[each[0]]):
            print(a_t, a_b, a_n, a_s, a_e, a_w, Q, S_c_rad, temperature[each[2] + 1][each[1]][each[0]], temperature[each[2] - 1][each[1]][each[0]], temperature[each[2]][each[1]+1][each[0]], temperature[each[2]][each[1]-1][each[0]], temperature[each[2]][each[1]][each[0]+1], temperature[each[2]][each[1]][each[0]-1], temperature[each[2]][each[1]][each[0]], Lambda[each[2]][each[1]][each[0]], each[0], each[1], each[2])
            #print(x_pos_periodic, x_neg_periodic, y_pos_periodic, y_neg_periodic)
    return sub_alpha, diag, sub_gamma, rhs


#@njit
def hte_implicit_DGADI_periodic(n_x, n_y, n_z, surface_reduced, r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder, inner_structure, ambient_temperature):
    next_step_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    z_sweep_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    y_sweep_temperature = np.zeros((n_z, n_y, n_x), dtype=np.float64)
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_z_sweep_periodic(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, n_x, n_y, inner_structure, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_z_sweep_periodic(n_z, j, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder, inner_structure)
            rhs = set_matrices_rhs_z_sweep_periodic(n_z, j, k, rhs, temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c, n_x, n_y, inner_structure)
            z_sweep_temperature[1:n_z-1, j, k] = tridiagonal_matrix_solver(n_z-2, diag[1:n_z-1], sub_gamma[1:n_z-1], sub_alpha[1:n_z-1], rhs[1:n_z-1])
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_y_sweep_periodic(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, z_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, n_x, n_y, inner_structure, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_y_sweep_periodic(n_y, i, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder, inner_structure)
            rhs = set_matrices_rhs_y_sweep_periodic(n_y, i, k, rhs, temperature, z_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c, n_x, inner_structure)
            y_sweep_temperature[i, 1:n_y-1, k] = periodic_tridiagonal_matrix_solver(n_y-2, diag[1:n_y-1], sub_gamma[1:n_y-1], sub_alpha[1:n_y-1], rhs[1:n_y-1])
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
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_x_sweep_periodic(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, z_sweep_temperature, y_sweep_temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p, n_x, n_y, inner_structure, ambient_temperature)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_x_sweep_periodic(n_x, i, j, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder, inner_structure)
            rhs = set_matrices_rhs_x_sweep_periodic(n_x, i, j, rhs, temperature, z_sweep_temperature, y_sweep_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c, n_y, inner_structure)
            next_step_temperature[i, j, 1:n_x-1] = periodic_tridiagonal_matrix_solver(n_x-2, diag[1:n_x-1], sub_gamma[1:n_x-1], sub_alpha[1:n_x-1], rhs[1:n_x-1])
    return next_step_temperature