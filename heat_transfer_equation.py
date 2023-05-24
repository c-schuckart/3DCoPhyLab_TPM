import numpy as np
from numba import jit, njit, prange
import variables_and_arrays as var
import constants as const
from scipy.optimize import brentq
from molecule_transfer import gas_mass_function

'''
Numerical implementation of the heat transfer equation in a finite differences, forward time centered space, explicit scheme.

Input parameters:
    j : float
	    Number of the current timestep
	n : float
		number of numerical layers
	delta_T_0 : float
	    Difference of the temperature of the previous to the current timestep in the top layer
	temperature : ndarray
		Temperature of the system at the current time step of dimension n+1
	Lambda : ndarray
		Array containing the total heat conductivity of each layer of dimension n
	DX : ndarray
		Array containing the distances between the mid-points of the numerical layers of dimension n
	dx : ndarray
		Array containing the thickness of the numerical layers of dimension n+1
	dt : float
	    Length of a timestep
	density : ndarray
	    Array containing the density of each numerical layer of dimension n+1
	heat_capacity : ndarray
	    Array containing the heat capacity of each numerical layer of dimension n+1
	j_leave : ndarray
	    Array containing the sublimating water molecules for each layer of dimension n+1
	j_inward : ndarray
	    Array containing the water molecules that resublimate again within the system for each layer of dimension n+1
	latent_heat_water : float
	    Latent heat for the sublimation of water ice
	j_leave_co2 : ndarray
	    Array containing the sublimating CO2 molecules for each layer of dimension n+1
	j_inward_co2 : ndarray
	    Array containing the CO2 molecules that resublimate again within the system for each layer of dimension n+1
	latent_heat_co2 : float
	    Latent heat for the sublimation of CO2 ice
	Fourier_number : ndarray
	    Array storing the Fourier numbers of each layer of dimension n+1
	Energy_Increase_per_Layer : ndarray
	    Array storing the increse of energy per layer of dimension n+1
	surface_area : float
	    Simulated surface area, standard: 1 square meter
	surface_temperature : float
	    Surface temperature of the previous timestep

Returns:
    delta_T : ndarray
        Difference of the temperature of the previous to the current timestep for each numerical layer of dimension n
    Fourier_number : ndarray
	    Array storing the Fourier numbers of each layer of dimension n+1
	Energy_Increase_per_Layer : ndarray
	    Array storing the increse of energy per layer of dimension n+1
	surface_temperature : float
	    Surface temperature of the previous timestep   
'''


@njit(parallel=True)
def hte_calculate(n_x, n_y, n_z, surface, delta_T_0, temperature, Lambda, Dr, dx, dy, dz, dt, density, heat_capacity, sublimated_mass, resublimated_mass, latent_heat_water, sample_holder):
    delta_T = np.zeros((n_z, n_y, n_x), dtype=np.float64) + delta_T_0
    Energy_Increase_per_Layer = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    Latent_Heat_per_Layer = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    Fourier_number = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    EcoPL = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
    E_sample_holder = 0
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if np.sum(surface[i][j][k]) == 0 and temperature[i][j][k] > 0:
                    # Standard Thermal Diffusivity Equation 3D explicit
                    delta_T[i][j][k] = ((((temperature[i][j][k + 1] - temperature[i][j][k]) * Lambda[i][j][k][4] / (Dr[i][j][k][4])) - ((temperature[i][j][k] - temperature[i][j][k - 1]) * Lambda[i][j][k][5] / (Dr[i][j][k][5]))) / dx[i][j][k]) * dt / (
                                             density[i][j][k] * heat_capacity[i][j][k]) + \
                                       ((((temperature[i][j + 1][k] - temperature[i][j][k]) * Lambda[i][j][k][2] / (Dr[i][j][k][2]))
                                         - ((temperature[i][j][k] - temperature[i][j - 1][k]) * Lambda[i][j][k][3] / (
                                               Dr[i][j][k][3]))) / dy[i][j][k]) * dt / (
                                               density[i][j][k] * heat_capacity[i][j][k]) + \
                                       ((((temperature[i + 1][j][k] - temperature[i][j][k]) * Lambda[i][j][k][0] / (Dr[i][j][k][0]))
                                         - ((temperature[i][j][k] - temperature[i - 1][j][k]) * Lambda[i][j][k][1] / (
                                               Dr[i][j][k][1]))) / dz[i][j][k]) * dt / (
                                               density[i][j][k] * heat_capacity[i][j][k]) - (sublimated_mass[i][j][k] - resublimated_mass[i][j][k]) * latent_heat_water[i][j][k] * 1 / (density[i][j][k] * heat_capacity[i][j][k])
                    Fourier_number[i][j][k] = np.max(Lambda[i][j][k]) / (density[i][j][k] * heat_capacity[i][j][k]) * dt * (1 / dx[i][j][k] ** 2 + 1 / dy[i][j][k] ** 2 + 1 / dz[i][j][k] ** 2)# [-]
                    Latent_Heat_per_Layer[i][j][k] = - (sublimated_mass[i][j][k] - resublimated_mass[i][j][k]) * latent_heat_water[i][j][k]
                    Energy_Increase_per_Layer[i][j][k] = heat_capacity[i][j][k] * density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] * delta_T[i][j][k]  # [J]
                    #if i == 2 and j == 2 and k == 50:
                        #print('E_INpL: ', Energy_Increase_per_Layer[i][j][k])
                elif sample_holder[i][j][k] == 1:
                    pos = np.zeros(6, dtype=np.float64)
                    if temperature[i+1][j][k] != 0 and sample_holder[i+1][j][k] == 0:
                        pos[0] = 1
                        #Lambda[i][j][k][0] = Lambda[i+1][j][k][1]
                    if temperature[i-1][j][k] != 0 and sample_holder[i-1][j][k] == 0:
                        pos[1] = 1
                        #Lambda[i][j][k][1] = Lambda[i-1][j][k][0]
                    if temperature[i][j+1][k] != 0 and sample_holder[i][j+1][k] == 0:
                        pos[2] = 1
                        #Lambda[i][j][k][2] = Lambda[i][j+1][k][3]
                    if temperature[i][j-1][k] != 0 and sample_holder[i][j-1][k] == 0:
                        pos[3] = 1
                        #Lambda[i][j][k][3] = Lambda[i][j-1][k][2]
                    if temperature[i][j][k+1] != 0 and sample_holder[i][j][k+1] == 0:
                        pos[4] = 1
                        #Lambda[i][j][k][4] = Lambda[i][j][k+1][5]
                    if temperature[i][j][k-1] != 0 and sample_holder[i][j][k-1] == 0:
                        pos[5] = 1
                        #Lambda[i][j][k][5] = Lambda[i][j][k-1][4]
                    '''if Lambda[i][j][k][4] != Lambda[i][j][k+1][5] and pos[4] == 1:
                                            print(i, j, k)'''
                    '''dT_energy_cons = ((((temperature[i][j][k + 1] - temperature[i][j][k]) * Lambda[i][j][k][4] / (Dr[i][j][k][4])) * pos[4] - ((temperature[i][j][k] - temperature[i][j][k - 1]) * Lambda[i][j][k][5] / (Dr[i][j][k][5])) * pos[5]) / dx[i][j][k]) * dt / (
                                               density[i][j][k] * heat_capacity[i][j][k]) + \
                                       ((((temperature[i][j + 1][k] - temperature[i][j][k]) * Lambda[i][j][k][2] / (
                                       Dr[i][j][k][2])) * pos[2]
                                         - ((temperature[i][j][k] - temperature[i][j - 1][k]) * Lambda[i][j][k][3] / (
                                                   Dr[i][j][k][3])) * pos[3]) / dy[i][j][k]) * dt / (
                                               density[i][j][k] * heat_capacity[i][j][k]) + \
                                       ((((temperature[i + 1][j][k] - temperature[i][j][k]) * Lambda[i][j][k][0] / (
                                       Dr[i][j][k][0])) * pos[0]
                                         - ((temperature[i][j][k] - temperature[i - 1][j][k]) * Lambda[i][j][k][1] / (
                                                   Dr[i][j][k][1])) * pos[1]) / dz[i][j][k]) * dt / (
                                               density[i][j][k] * heat_capacity[i][j][k])
                    #Energy_Increase_per_Layer[i][j][k] = heat_capacity[i][j][k] * density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] * dT_energy_cons
                    E_sample_holder += heat_capacity[i][j][k] * density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] * dT_energy_cons'''
                    E_Cond_z_pos = Lambda[i][j][k][0] * (
                                temperature[i + 1][j][k] - temperature[i][j][k]) / \
                                   Dr[i][j][k][0] * dt * dx[i][j][k] * \
                                   dy[i][j][k] * (pos[0])
                    E_Cond_z_neg = Lambda[i][j][k][1] * (
                                temperature[i - 1][j][k] - temperature[i][j][k]) / \
                                   Dr[i][j][k][1] * dt * dx[i][j][k] * \
                                   dy[i][j][k] * (pos[1])
                    E_Cond_y_pos = Lambda[i][j][k][2] * (
                                temperature[i][j + 1][k] - temperature[i][j][k]) / \
                                   Dr[i][j][k][2] * dt * dx[i][j][k] * \
                                   dz[i][j][k] * (pos[2])
                    E_Cond_y_neg = Lambda[i][j][k][3] * (
                                temperature[i][j - 1][k] - temperature[i][j][k]) / \
                                   Dr[i][j][k][3] * dt * dx[i][j][k] * \
                                   dz[i][j][k] * (pos[3])
                    E_Cond_x_pos = Lambda[i][j][k][4] * (
                                temperature[i][j][k + 1] - temperature[i][j][k]) / \
                                   Dr[i][j][k][4] * dt * dy[i][j][k] * \
                                   dz[i][j][k] * (pos[4])
                    E_Cond_x_neg = Lambda[i][j][k][5] * (
                                temperature[i][j][k - 1] - temperature[i][j][k]) / \
                                   Dr[i][j][k][5] * dt * dy[i][j][k] * \
                                   dz[i][j][k] * (pos[5])
                    E_sample_holder += (E_Cond_x_pos + E_Cond_x_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_z_pos + E_Cond_z_neg)
                    EcoPL[i][j][k][0], EcoPL[i][j][k][1], EcoPL[i][j][k][2], EcoPL[i][j][k][3], EcoPL[i][j][k][4], EcoPL[i][j][k][5] = E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg
                '''if (E_Cond_x_pos + E_Cond_x_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_z_pos + E_Cond_z_neg) != 0 and i == 1:
                        print(temperature[i-1:i+2, j-1:j+2, k-1:k+2])'''
                '''if i == 2 and j == 1 and k == 50:
                        print('SAMPLE HOLDER: ', E_Cond_z_pos,E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg)
                        print(temperature[i][j][k], temperature[i][j+1][k])'''
    return delta_T, Energy_Increase_per_Layer, Latent_Heat_per_Layer, np.max(Fourier_number), E_sample_holder, EcoPL


@njit
def test_E_cond(n_x, n_y, n_z, surface, delta_T_0, temperature, Lambda, Dr, dx, dy, dz, dt, density, heat_capacity, sublimated_mass, resublimated_mass, latent_heat_water, sample_holder, Energy_conduction_per_Layer):
    #Energy_conduction_per_Layer = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
    Delta_cond = 0
    DCmax = 0
    DCmax_pos = np.zeros(3, dtype=np.int32)
    E_sample_holder = 0
    Delta_cond_fl = 0
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if np.sum(surface[i][j][k]) == 0 and temperature[i][j][k] > 0:
                    Energy_conduction_per_Layer[i][j][k][0] = ((((temperature[i + 1][j][k] - temperature[i][j][k]) *
                                                                 Lambda[i][j][k][0] / (Dr[i][j][k][0]))
                                                                - ((0) * Lambda[i][j][k][1] / (
                                Dr[i][j][k][1]))) / dz[i][j][k]) * dt / (
                                                                      density[i][j][k] * heat_capacity[i][j][k]) * \
                                                              density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][
                                                                  k] * heat_capacity[i][j][k]
                    Energy_conduction_per_Layer[i][j][k][1] = ((((0) * Lambda[i][j][k][0] / (Dr[i][j][k][0]))
                                                                - ((temperature[i][j][k] - temperature[i - 1][j][k]) *
                                                                   Lambda[i][j][k][1] / (
                                                                       Dr[i][j][k][1]))) / dz[i][j][k]) * dt / (
                                                                      density[i][j][k] * heat_capacity[i][j][k]) * \
                                                              density[i][j][k] * dx[i][j][k] * dy[i][j][k] * \
                                                              dz[i][j][k] * heat_capacity[i][j][k]
                    Energy_conduction_per_Layer[i][j][k][2] = ((((temperature[i][j + 1][k] - temperature[i][j][k]) *
                                                                 Lambda[i][j][k][2] / (Dr[i][j][k][2]))
                                                                - ((0) * Lambda[i][j][k][3] / (
                                Dr[i][j][k][3]))) / dy[i][j][k]) * dt / (
                                                                      density[i][j][k] * heat_capacity[i][j][k]) * \
                                                              density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][
                                                                  k] * heat_capacity[i][j][k]
                    Energy_conduction_per_Layer[i][j][k][3] = ((((0) * Lambda[i][j][k][2] / (Dr[i][j][k][2]))
                                                                - ((temperature[i][j][k] - temperature[i][j - 1][k]) *
                                                                   Lambda[i][j][k][3] / (
                                                                       Dr[i][j][k][3]))) / dy[i][j][k]) * dt / (
                                                                      density[i][j][k] * heat_capacity[i][j][k]) * \
                                                              density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][
                                                                  k] * heat_capacity[i][j][k]
                    Energy_conduction_per_Layer[i][j][k][4] = ((((temperature[i][j][k + 1] - temperature[i][j][k]) *
                                                                 Lambda[i][j][k][4] / (Dr[i][j][k][4])) - (
                                                                            (0) * Lambda[i][j][k][5] / (
                                                                    Dr[i][j][k][5]))) / dx[i][j][k]) * dt / (
                                                                      density[i][j][k] * heat_capacity[i][j][k]) * \
                                                              density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][
                                                                  k] * heat_capacity[i][j][k]
                    Energy_conduction_per_Layer[i][j][k][5] = ((((0) *
                                                                 Lambda[i][j][k][4] / (Dr[i][j][k][4])) - ((temperature[
                                                                                                                i][j][
                                                                                                                k] -
                                                                                                            temperature[
                                                                                                                i][j][
                                                                                                                k - 1]) *
                                                                                                           Lambda[i][j][
                                                                                                               k][5] / (
                                                                                                           Dr[i][j][k][
                                                                                                               5]))) /
                                                               dx[i][j][k]) * dt / (
                                                                      density[i][j][k] * heat_capacity[i][j][k]) * \
                                                              density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][
                                                                  k] * heat_capacity[i][j][k]
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if sample_holder[i][j][k] == 1:
                    pos = np.zeros(6, dtype=np.float64)
                    if temperature[i + 1][j][k] != 0 and sample_holder[i + 1][j][k] == 0:
                        pos[0] = 1
                        # Lambda[i][j][k][0] = Lambda[i+1][j][k][1]
                    if temperature[i - 1][j][k] != 0 and sample_holder[i - 1][j][k] == 0:
                        pos[1] = 1
                        # Lambda[i][j][k][1] = Lambda[i-1][j][k][0]
                    if temperature[i][j + 1][k] != 0 and sample_holder[i][j + 1][k] == 0:
                        pos[2] = 1
                        # Lambda[i][j][k][2] = Lambda[i][j+1][k][3]
                    if temperature[i][j - 1][k] != 0 and sample_holder[i][j - 1][k] == 0:
                        pos[3] = 1
                        # Lambda[i][j][k][3] = Lambda[i][j-1][k][2]
                    if temperature[i][j][k + 1] != 0 and sample_holder[i][j][k + 1] == 0:
                        pos[4] = 1
                        # Lambda[i][j][k][4] = Lambda[i][j][k+1][5]
                    if temperature[i][j][k - 1] != 0 and sample_holder[i][j][k - 1] == 0:
                        pos[5] = 1
                        # Lambda[i][j][k][5] = Lambda[i][j][k-1][4]
                    E_Cond_z_pos = Lambda[i][j][k][0] * (
                            temperature[i + 1][j][k] - temperature[i][j][k]) / \
                                   Dr[i][j][k][0] * dt * dx[i][j][k] * \
                                   dy[i][j][k] * (pos[0])
                    E_Cond_z_neg = Lambda[i][j][k][1] * (
                            temperature[i - 1][j][k] - temperature[i][j][k]) / \
                                   Dr[i][j][k][1] * dt * dx[i][j][k] * \
                                   dy[i][j][k] * (pos[1])
                    E_Cond_y_pos = Lambda[i][j][k][2] * (
                            temperature[i][j + 1][k] - temperature[i][j][k]) / \
                                   Dr[i][j][k][2] * dt * dx[i][j][k] * \
                                   dz[i][j][k] * (pos[2])
                    E_Cond_y_neg = Lambda[i][j][k][3] * (
                            temperature[i][j - 1][k] - temperature[i][j][k]) / \
                                   Dr[i][j][k][3] * dt * dx[i][j][k] * \
                                   dz[i][j][k] * (pos[3])
                    E_Cond_x_pos = Lambda[i][j][k][4] * (
                            temperature[i][j][k + 1] - temperature[i][j][k]) / \
                                   Dr[i][j][k][4] * dt * dy[i][j][k] * \
                                   dz[i][j][k] * (pos[4])
                    E_Cond_x_neg = Lambda[i][j][k][5] * (
                            temperature[i][j][k - 1] - temperature[i][j][k]) / \
                                   Dr[i][j][k][5] * dt * dy[i][j][k] * \
                                   dz[i][j][k] * (pos[5])
                    Delta_cond_z_pos = E_Cond_z_pos + Energy_conduction_per_Layer[i+1][j][k][1]
                    Delta_cond_z_neg = E_Cond_z_neg + Energy_conduction_per_Layer[i-1][j][k][0]
                    Delta_cond_y_pos = E_Cond_y_pos + Energy_conduction_per_Layer[i][j+1][k][3]
                    Delta_cond_y_neg = E_Cond_y_neg + Energy_conduction_per_Layer[i][j-1][k][2]
                    Delta_cond_x_pos = E_Cond_x_pos + Energy_conduction_per_Layer[i][j][k+1][5]
                    Delta_cond_x_neg = E_Cond_x_neg + Energy_conduction_per_Layer[i][j][k-1][4]
                    E_sample_holder += (E_Cond_x_pos + E_Cond_x_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_z_pos + E_Cond_z_neg)
                    Delta_cond_step = (Delta_cond_z_pos + Delta_cond_z_neg + Delta_cond_y_pos + Delta_cond_y_neg + Delta_cond_x_pos + Delta_cond_x_neg)
                    Delta_cond += (Delta_cond_z_pos + Delta_cond_z_neg + Delta_cond_y_pos + Delta_cond_y_neg + Delta_cond_x_pos + Delta_cond_x_neg)
                    if i == 98:
                        Delta_cond_fl += Delta_cond_step
                    if np.abs(Delta_cond_step) > DCmax:
                        DCmax = np.abs(Delta_cond_step)
                        DCmax_pos[0], DCmax_pos[1], DCmax_pos[2] = k, j, i
                    if (i == 50 and j == 1 and k == 50) or (i == 50 and j == 2 and k == 49) or (i == 50 and j == 2 and k == 51):
                        #print(temperature[i][j][k], temperature[i-1][j][k], Lambda[i][j][k], Lambda[i-1][j][k])
                        print(i, j, k)
                        print(E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg)
                        print(Energy_conduction_per_Layer[i + 1][j][k][1], Energy_conduction_per_Layer[i - 1][j][k][0],
                              Energy_conduction_per_Layer[i][j + 1][k][3], Energy_conduction_per_Layer[i][j - 1][k][2],
                              Energy_conduction_per_Layer[i][j][k + 1][5], Energy_conduction_per_Layer[i][j][k - 1][4])
                    '''if (Delta_cond_z_pos + Delta_cond_z_neg + Delta_cond_y_pos + Delta_cond_y_neg + Delta_cond_x_pos + Delta_cond_x_neg) != 0:
                        print(i,j,k)
                        print(Delta_cond)
                        print(E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg)
                        print(Energy_conduction_per_Layer[i+1][j][k][1], Energy_conduction_per_Layer[i-1][j][k][0], Energy_conduction_per_Layer[i][j+1][k][3], Energy_conduction_per_Layer[i][j-1][k][2], Energy_conduction_per_Layer[i][j][k+1][5], Energy_conduction_per_Layer[i][j][k-1][4])
                        break'''
    #print('Delta Cond: ', Delta_cond, DCmax_pos, Delta_cond_fl, DCmax)
    #print('E_sh: ', E_sample_holder)
    return Delta_cond


@njit
def test_e_cond_2(n_x, n_y, n_z, Energy_Increase_per_layer, EcoPL, target_height, sample_holder):
    EcoPl_ges = np.zeros((n_z, n_y, n_y), dtype=np.float64)
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if sample_holder[i+1][j][k] == 1:
                    EcoPl_ges[i][j][k] += EcoPL[i+1][j][k][1]
                if sample_holder[i-1][j][k] == 1:
                    EcoPl_ges[i][j][k] += EcoPL[i-1][j][k][0]
                if sample_holder[i][j+1][k] == 1:
                    EcoPl_ges[i][j][k] += EcoPL[i][j+1][k][3]
                if sample_holder[i][j-1][k] == 1:
                    EcoPl_ges[i][j][k] += EcoPL[i][j-1][k][2]
                if sample_holder[i][j][k+1] == 1:
                    EcoPl_ges[i][j][k] += EcoPL[i][j][k+1][5]
                if sample_holder[i][j][k-1] == 1:
                    EcoPl_ges[i][j][k] += EcoPL[i][j][k-1][4]
    Delta_ges = 0
    for i in range(target_height, target_height+1):
        for j in range(1, n_y-1):
            for k in range(1,n_x-1):
                if np.abs(Energy_Increase_per_layer[i][j][k] + EcoPl_ges[i][j][k]) > 1E-17:
                    print(i, j, k)
                    print(EcoPl_ges[i][j][k], Energy_Increase_per_layer[i][j][k])
                    Delta_ges += np.abs(Energy_Increase_per_layer[i][j][k] - EcoPl_ges[i][j][k])
    print('DELTA: ', Delta_ges)
'''
Numerical implementation of the heat transfer equation in a finite differences, forward time centered space, explicit scheme.

Input parameters:
	n : float
		number of numerical layers
	temperature : ndarray
		Temperature of the system at the previous time step of dimension n+1
    Energy_Increase_Total_per_time_Step : float
        Total energy increase in the system summed over all the previous time steps
    water_content_per_layer : ndarray
		Array containing the number of water ice molecules within each layer of dimension n+1
	co2_content_per_layer : ndarray
		Array containing the number of CO2 ice molecules within each layer of dimension n+1
	outgassed_molecule_per_time_step : float
	    Total outgassed water molecules of the system summed over all the previous time steps
	outgassed_molecule_per_time_step_co2 : float
	    Total outgassed CO2 molecules of the system summed over all the previous time steps
	delta_T : ndarray
        Difference of the temperature of the previous to the current timestep for each numerical layer of dimension n
    Energy_Increase_per_Layer : ndarray
	    Array storing the increse of energy per layer of dimension n+1
	surface_area : float
	    Simulated surface area, standard: 1 square meter
	j_leave : ndarray
	    Array containing the sublimating water molecules for each layer of dimension n+1
	j_inward : ndarray
	    Array containing the water molecules that resublimate again within the system for each layer of dimension n+1
	j_leave_co2 : ndarray
	    Array containing the sublimating CO2 molecules for each layer of dimension n+1
	j_inward_co2 : ndarray
	    Array containing the CO2 molecules that resublimate again within the system for each layer of dimension n+1
	dt : float
	    Length of a timestep
	avogadro_constant : float
	    Avogadro constant
	molar_mass_water : float
	    Molar mass of water
	molar_mass_co2 : float
	    Molar mass of CO2
	heat_capacity : ndarray
	    Array containing the heat capacity of each numerical layer of dimension n+1
	heat_capacity_dust : float
		Heat capacity of the dust component of cometary material
	heat_capacity_water_ice : float
		Heat conductivity of the water ice at low temperatures
	heat_capacity_co2_ice : float
		Heat conductivity of the CO2 ice at low temperatures
	dust_mass_in_dust_ice_layer : float
	    Mass of the dust component in a layer with the global dust ice ratio
	dust_ice_ratio_per_layer : ndarray
		Array containing the ratio of dust to ices for each layer of dimension n+1
	co2_h2o_ratio_per_layer : ndarray
		Array containing the ratio of CO2 ice to water ice for each layer of dimension n+1
	
	

Returns:
    temperature : ndarray
		Temperature of the system at the current time step of dimension n+1
	Energy_Increase_Total_per_time_Step : float
        Total energy increase in the system summed over all the previous time steps, including the current one
    ater_content_per_layer : ndarray
		Array containing the number of water ice molecules within each layer of dimension n+1
	co2_content_per_layer : ndarray
		Array containing the number of CO2 ice molecules within each layer of dimension n+1
	outgassed_molecule_per_time_step : float
	    Total outgassed water molecules of the system summed over all the previous time steps, including the current one
	outgassed_molecule_per_time_step_co2 : float
	    Total outgassed CO2 molecules of the system summed over all the previous time steps, including the current one
	heat_capacity : ndarray
	    Array containing the heat capacity of each numerical layer of dimension n+1
	dust_ice_ratio_per_layer : ndarray
		Array containing the ratio of dust to ices for each layer of dimension n+1
	co2_h2o_ratio_per_layer : ndarray
		Array containing the ratio of CO2 ice to water ice for each layer of dimension n+1	    
'''
@njit(parallel=False)
def update_thermal_arrays(n_x, n_y, n_z, temperature, uniform_water_mass,  delta_T, Energy_Increase_per_Layer, sublimated_mass, resublimated_mass, dt, avogadro_constant, molar_mass_water, molar_mass_co2, heat_capacity, heat_capacity_water_ice, heat_capacity_co2_ice, EIpL_0, Latent_Heat_per_Layer, E_Lat_0, E_Rad, E_In, E_sh):
    temperature_o = temperature + delta_T
    Energy_Increase_Total_per_time_Step = 0
    Latent_Heat_per_time_step = 0
    dust_ice_ratio_per_layer = 0
    co2_h2o_ratio_per_layer = 0
    uniform_water_mass = uniform_water_mass - sublimated_mass
    '''for i in range(0, n + 1):
        #temperature_o[i] = temperature[i] + delta_T[i]  # [K]
        #print(temperature[i], delta_T[i], temperature_o[i])
        Energy_Increase_Total_per_time_Step += Energy_Increase_per_Layer[i] * surface_area  # [J/m^2 * m^2]
        Latent_Heat_per_time_step += Latent_Heat_per_Layer[i]
        water_content_per_layer[i] += (j_inward[i] - j_leave[i]) * surface_area * dt / molar_mass_water * avogadro_constant
        co2_content_per_layer[i] += (j_inward_co2[i] - j_leave_co2[i]) * surface_area * dt / molar_mass_co2 * avogadro_constant
        outgassed_molecules_per_time_step += j_leave[i] / 2 * surface_area * dt / molar_mass_water * avogadro_constant
        outgassed_molecules_per_time_step_co2 += j_leave_co2[i] / 2 * surface_area * dt / molar_mass_co2 * avogadro_constant
        #Heat capacity update part
        if water_content_per_layer[i] < 0:
            water_content_per_layer[i+1] += water_content_per_layer[i]
            water_content_per_layer[i] = 0
        if co2_content_per_layer[i] < 0:
            co2_content_per_layer[i+1] += co2_content_per_layer[i]
            co2_content_per_layer[i] = 0
        mass_ice = water_content_per_layer[i] / avogadro_constant * molar_mass_water + co2_content_per_layer[i] / avogadro_constant * molar_mass_co2
        #dust_ice_ratio_per_layer[i] = mass_ice / (mass_ice + dust_mass_in_dust_ice_layers)
        mass_co2 = co2_content_per_layer[i] / avogadro_constant * molar_mass_co2
        if mass_ice > 0:
            co2_h2o_ratio_per_layer[i] = mass_co2 / mass_ice
        else:
            co2_h2o_ratio_per_layer[i] = 0
        heat_capacity[i] = heat_capacity_dust * (1 - dust_ice_ratio_per_layer[i]) + heat_capacity_water_ice * (
                    dust_ice_ratio_per_layer[i] * (1 - co2_h2o_ratio_per_layer[i])) + heat_capacity_co2_ice * (
                                       dust_ice_ratio_per_layer[i] * co2_h2o_ratio_per_layer[i])'''
    Energy_Increase_Total_per_time_Step = np.sum(Energy_Increase_per_Layer) + EIpL_0
    Latent_Heat_per_time_step = np.sum(Latent_Heat_per_Layer) + E_Lat_0
    E_conservation = Energy_Increase_Total_per_time_Step - E_Rad - Latent_Heat_per_time_step - E_In + E_sh
    # Set Energy Loss per Timestep = 0 -> Differential Counting of Energy Loss
    return temperature_o, uniform_water_mass, heat_capacity, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, E_conservation, Energy_Increase_Total_per_time_Step, E_Rad, Latent_Heat_per_time_step, E_In


#@njit(parallel=False)
def update_thermal_arrays_diffusion(n_x, n_y, n_z, temperature, uniform_water_mass,  delta_T, Energy_Increase_per_Layer, sublimated_mass, resublimated_mass, dt, avogadro_constant, molar_mass_water, molar_mass_co2, heat_capacity, heat_capacity_water_ice, heat_capacity_co2_ice, EIpL_0, Latent_Heat_per_Layer, E_Lat_0, E_Rad, E_In, gas_mass, delta_gm, pressure, VFF, dx, dy, dz, a_1, b_1, c_1, d_1, m_H2O, k_Boltzmann, r_mono, sample_holder, temperature_ini):
    temperature_o = temperature + delta_T
    gas_mass = gas_mass + delta_gm
    temperature_outgassing = np.zeros((n_z, n_y, n_x), dtype=np.float64) + delta_T + sample_holder * temperature_ini
    pressure_outgassing = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    sublimated_mass_outgassing = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    pressure = gas_mass * np.sqrt(2 * np.pi * k_Boltzmann * temperature_o / m_H2O)
    Energy_Increase_Total_per_time_Step = 0
    Latent_Heat_per_time_step = np.sum(Latent_Heat_per_Layer) + E_Lat_0
    dust_ice_ratio_per_layer = 0
    co2_h2o_ratio_per_layer = 0
    uniform_water_mass = uniform_water_mass - sublimated_mass
    '''for i in range(0, n + 1):
        #temperature_o[i] = temperature[i] + delta_T[i]  # [K]
        #print(temperature[i], delta_T[i], temperature_o[i])
        Energy_Increase_Total_per_time_Step += Energy_Increase_per_Layer[i] * surface_area  # [J/m^2 * m^2]
        Latent_Heat_per_time_step += Latent_Heat_per_Layer[i]
        water_content_per_layer[i] += (j_inward[i] - j_leave[i]) * surface_area * dt / molar_mass_water * avogadro_constant
        co2_content_per_layer[i] += (j_inward_co2[i] - j_leave_co2[i]) * surface_area * dt / molar_mass_co2 * avogadro_constant
        outgassed_molecules_per_time_step += j_leave[i] / 2 * surface_area * dt / molar_mass_water * avogadro_constant
        outgassed_molecules_per_time_step_co2 += j_leave_co2[i] / 2 * surface_area * dt / molar_mass_co2 * avogadro_constant
        #Heat capacity update part
        if water_content_per_layer[i] < 0:
            water_content_per_layer[i+1] += water_content_per_layer[i]
            water_content_per_layer[i] = 0
        if co2_content_per_layer[i] < 0:
            co2_content_per_layer[i+1] += co2_content_per_layer[i]
            co2_content_per_layer[i] = 0
        mass_ice = water_content_per_layer[i] / avogadro_constant * molar_mass_water + co2_content_per_layer[i] / avogadro_constant * molar_mass_co2
        #dust_ice_ratio_per_layer[i] = mass_ice / (mass_ice + dust_mass_in_dust_ice_layers)
        mass_co2 = co2_content_per_layer[i] / avogadro_constant * molar_mass_co2
        if mass_ice > 0:
            co2_h2o_ratio_per_layer[i] = mass_co2 / mass_ice
        else:
            co2_h2o_ratio_per_layer[i] = 0
        heat_capacity[i] = heat_capacity_dust * (1 - dust_ice_ratio_per_layer[i]) + heat_capacity_water_ice * (
                    dust_ice_ratio_per_layer[i] * (1 - co2_h2o_ratio_per_layer[i])) + heat_capacity_co2_ice * (
                                       dust_ice_ratio_per_layer[i] * co2_h2o_ratio_per_layer[i])'''
    Energy_Increase_Total_per_time_Step = np.sum(Energy_Increase_per_Layer) + EIpL_0
    E_conservation = Energy_Increase_Total_per_time_Step - E_Rad - Latent_Heat_per_time_step - E_In
    # Set Energy Loss per Timestep = 0 -> Differential Counting of Energy Loss
    return temperature_o, uniform_water_mass, heat_capacity, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, E_conservation, Energy_Increase_Total_per_time_Step, E_Rad, Latent_Heat_per_time_step, E_In, gas_mass, pressure