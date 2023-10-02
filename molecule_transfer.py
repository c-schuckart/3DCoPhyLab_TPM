import numpy as np
from numba import jit, njit, prange
#import main
import constants as const
import variables_and_arrays as var
from scipy.linalg import solve


'''
Calculation of sublimating and resublimating molecules of the system volatiles.

Input parameters:
    temperature : ndarray
		Temperature of the system at the current time step of dimension n+1
	j_leave : ndarray
	    Array containing the sublimating water molecules for each layer of dimension n+1
	j_leave_co2 : ndarray
	    Array containing the sublimating CO2 molecules for each layer of dimension n+1
	a_H2O : float
		Empirical coefficient for the sublimation pressure calculation of water ice
	b_H2O : float
		Empirical coefficient for the sublimation pressure calculation of water ice
	m_H2O : float
	    Particle mass of H2O
	k_boltzmann : float
	    Boltzmann constant
	b : float
	    Coefficient used in calculating the efficiency function in Gundlach et al. (2020) equating to 4 times the diffusion scale length
	water_content_per_layer : ndarray
		Array containing the number of water ice molecules within each layer of dimension n+1
	avogadro_constant : float
	    Avogadro constant
	molar_mass_water : float
	    Molar mass of water
	dt : float
	    Length of a timestep
	dx : float
	    Thickness of the numerical layer
	n : float
		number of numerical layers
	co2_content_per_layer : ndarray
		Array containing the number of CO2 ice molecules within each layer of dimension n+1
	a_CO2 : float
		Empirical coefficient for the sublimation pressure calculation of CO2 ice
	b_CO2 : float
		Empirical coefficient for the sublimation pressure calculation of CO2 ice
	m_CO2 : float
	    Particle mass of CO2
	molar_mass_co2 : float
	    Molar mass of CO2
	diffusion_factors : ndarray
	    Array containing the distribution factors for the inward diffusion/resublimation of volatiles of dimension 3
	deeper_diffusion : float
	    Catches water molecules that would diffuse into layers deeper than the modelled scope 
	deeper_diffusion_co2 : float
	    Catches CO2 molecules that would diffuse into layers deeper than the modelled scope 

Returns:
    j_leave : ndarray
	    Array containing the sublimating water molecules for each layer of dimension n+1
	j_inward : ndarray
	    Array containing the water molecules that resublimate again within the system for each layer of dimension n+1
	j_leave_co2 : ndarray
	    Array containing the sublimating CO2 molecules for each layer of dimension n+1
	j_inward_co2 : ndarray
	    Array containing the CO2 molecules that resublimate again within the system for each layer of dimension n+1
	deeper_diffusion : float
	    Catches water molecules that would diffuse into layers deeper than the modelled scope 
	deeper_diffusion_co2 : float
	    Catches CO2 molecules that would diffuse into layers deeper than the modelled scope 
'''
#@njit(parallel=True)
def calculate_molecule_flux(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, mol_mass, R_gas, VFF, r_grain, Phi, tortuosity, dx, dy, dz, dt, surface_reduced, avogadro_constant, k_B, sample_holder, water_mass_per_layer, n_x_lr, n_y_lr, n_z_lr, Dr):
    p_sub = 10 ** (a_1[0] + b_1[0] / temperature + c_1[0] * np.log10(temperature) + d_1[0] * temperature)
    sublimated_mass = (p_sub - pressure) * np.sqrt(mol_mass[0]/(2 * np.pi * R_gas * temperature)) * (3 * VFF / r_grain * dx * dy * dz)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Placeholder
    outgassed_mass = 0
    empty_voxels = np.empty((0, 0), dtype=np.int32)
    for each in surface_reduced:
        #Setting p_surface to zero since outgassing can be assumed to always happen towards the vacuum
        outgassed_mass += sublimated_mass[each[2]][each[1]][each[0]]
        p_sub[each[2]][each[1]][each[0]] = 0
    mass_flux = np.zeros(np.shape(sublimated_mass), dtype=np.float64)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] != 1:
                    empty = False
                    if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                        sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                        empty_voxels = np.append(empty_voxels, np.array([k, j, i], dtype=np.int32))
                        empty = True
                    diff_z = (1 - VFF[i][j][k]) * np.sqrt(1/(2 * np.pi * mol_mass[0] * R_gas * (temperature[i-1][j][k] + temperature[i][j][k])/2)) * (p_sub[i-1][j][k] - p_sub[i][j][k])/(1 + 3 * (1 - (1 - VFF[i][j][k]))/(2 * (1 - VFF[i][j][k]) * r_grain) * Phi * tortuosity * Dr[i][j][k][0] / 4)
                    diff_y = (1 - VFF[i][j][k]) * np.sqrt(1/(2 * np.pi * mol_mass[0] * R_gas * (temperature[i][j-1][k] + temperature[i][j][k])/2)) * (p_sub[i][j-1][k] - p_sub[i][j][k])/(1 + 3 * (1 - (1 - VFF[i][j][k]))/(2 * (1 - VFF[i][j][k]) * r_grain) * Phi * tortuosity * Dr[i][j][k][2] / 4)
                    diff_x = (1 - VFF[i][j][k]) * np.sqrt(1/(2 * np.pi * mol_mass[0] * R_gas * (temperature[i][j][k-1] + temperature[i][j][k])/2)) * (p_sub[i][j][k-1] - p_sub[i][j][k])/(1 + 3 * (1 - (1 - VFF[i][j][k]))/(2 * (1 - VFF[i][j][k]) * r_grain) * Phi * tortuosity * Dr[i][j][k][4] / 4)
                    #print(i, j, k)
                    #print(sample_holder[i-1:i+2][j][k])
                    #print(sample_holder[i][j][k], sample_holder[i][j-1][k], sample_holder[i][j+1][k])
                    #print(sample_holder[1][j-1:j+2][:])
                    if np.sum(np.array([sample_holder[i-1][j][k], sample_holder[i][j][k], sample_holder[i+1][j][k]])) != 0:
                        diff_z = 0
                    if np.sum(np.array([sample_holder[i][j-1][k], sample_holder[i][j][k], sample_holder[i][j+1][k]])) != 0:
                        diff_y = 0
                    if np.sum(np.array([sample_holder[i][j][k-1], sample_holder[i][j][k], sample_holder[i][j][k+1]])) != 0:
                        diff_x = 0
                    mass_flux[i-1][j][k] -= diff_z
                    mass_flux[i+1][j][k] += diff_z
                    mass_flux[i][j-1][k] -= diff_y
                    mass_flux[i][j+1][k] += diff_y
                    mass_flux[i][j][k-1] -= diff_x
                    mass_flux[i][j][k+1] += diff_x
                    if temperature[i-1][j][k] > temperature[i][j][k] and diff_z > 0 and not empty:
                        resublimated_mass[i][j][k] += diff_z
                        mass_flux[i-1][j][k] = 0
                    elif temperature[i+1][j][k] > temperature[i][j][k] and diff_z < 0 and not empty:
                        resublimated_mass[i][j][k] -= diff_z
                        mass_flux[i+1][j][k] = 0
                    if temperature[i][j-1][k] > temperature[i][j][k] and diff_y > 0 and not empty:
                        resublimated_mass[i][j][k] += diff_y
                        mass_flux[i][j-1][k] = 0
                    elif temperature[i][j+1][k] > temperature[i][j][k] and diff_y < 0 and not empty:
                        resublimated_mass[i][j][k] -= diff_y
                        mass_flux[i][j+1][k] = 0
                    if temperature[i][j][k-1] > temperature[i][j][k] and diff_x > 0 and not empty:
                        resublimated_mass[i][j][k] += diff_x
                        mass_flux[i][j][k-1] = 0
                    elif temperature[i][j][k+1] > temperature[i][j][k] and diff_x < 0 and not empty:
                        resublimated_mass[i][j][k] -= diff_x
                        mass_flux[i][j][k+1] = 0
                    for a in range(0, 6):
                        if temperature[i + n_z_lr[a]][j + n_y_lr[a]][k + n_x_lr[a]] == 0:
                            outgassed_mass += mass_flux[i + n_z_lr[a]][j + n_y_lr[a]][k + n_x_lr[a]]
                        else:
                            p_sub[i + n_z_lr[a]][j + n_y_lr[a]][k + n_x_lr[a]] += mass_flux[i + n_z_lr[a]][j + n_y_lr[a]][k + n_x_lr[a]] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i + n_z_lr[a]][j + n_y_lr[a]][k + n_x_lr[a]] / Dr[a]
                    '''p_sub[i+1][j][k] += mass_flux[i+1][j][k] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i+1][j][k] / dz[i+1][j][k]
                    p_sub[i][j-1][k] += mass_flux[i][j-1][k] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i][j-1][k] / dy[i][j-1][k]
                    p_sub[i][j+1][k] += mass_flux[i][j+1][k] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i][j+1][k] / dy[i][j+1][k]
                    p_sub[i][j][k-1] += mass_flux[i][j][k-1] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i][j][k-1] / dx[i][j][k-1]
                    p_sub[i][j][k+1] += mass_flux[i][j][k+1] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i][j][k+1] / dx[i][j][k+1]'''
    '''for each in surrounding_surface:
        #This should always be >= 1 since p_surface is set to 0
        outgassed_mass += max(mass_flux[each[2]][each[1]][each[0]], 0)
        p_sub[each[2]][each[1]][each[0]] = 0'''

    pressure = p_sub
    #Non 100% resublimation missing

    return sublimated_mass, resublimated_mass, pressure, outgassed_mass/dt, empty_voxels, mass_flux


#@njit(parallel=True)
def calculate_molecule_flux_diag(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, mol_mass, R_gas, VFF, r_grain, Phi, tortuosity, dx, dy, dz, dt, surface_reduced, avogadro_constant, k_B, sample_holder, water_mass_per_layer, n_x_lr, n_y_lr, n_z_lr, Dr):
    p_sub = 10 ** (a_1[0] + b_1[0] / temperature + c_1[0] * np.log10(temperature) + d_1[0] * temperature)
    sublimated_mass = (p_sub - pressure) * np.sqrt(mol_mass[0]/(2 * np.pi * R_gas * temperature)) * (3 * VFF / r_grain * dx * dy * dz)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Placeholder
    outgassed_mass = 0
    empty_voxels = np.empty((0, 0), dtype=np.int32)
    for each in surface_reduced:
        #Setting p_surface to zero since outgassing can be assumed to always happen towards the vacuum
        outgassed_mass += sublimated_mass[each[2]][each[1]][each[0]]
        p_sub[each[2]][each[1]][each[0]] = 0
    mass_flux = np.zeros(np.shape(sublimated_mass), dtype=np.float64)
    j = n_y//2
    k = n_x//2
    matrix = np.zeros((n_z, n_z), dtype=np.float64)
    pressures_z = np.zeros(n_z, dtype=np.float64)
    for i in prange(1, n_z-1):
        if temperature[i][j][k] > 0 and sample_holder[i][j][k] != 1:
            empty = False
            if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                empty_voxels = np.append(empty_voxels, np.array([k, j, i], dtype=np.int32))
                empty = True
            alpha_top = (1 - VFF[i][j][k]) * np.sqrt(1/(2 * np.pi * mol_mass[0] * R_gas * (temperature[i-1][j][k] + temperature[i][j][k])/2)) / (1 + 3 * (1 - (1 - VFF[i][j][k]))/(2 * (1 - VFF[i][j][k]) * r_grain) * Phi * tortuosity * Dr[i][j][k][0] / 4)
            alpha_bottom = (1 - VFF[i][j][k]) * np.sqrt(1/(2 * np.pi * mol_mass[0] * R_gas * (temperature[i+1][j][k] + temperature[i][j][k])/2)) / (1 + 3 * (1 - (1 - VFF[i][j][k]))/(2 * (1 - VFF[i][j][k]) * r_grain) * Phi * tortuosity * Dr[i][j][k][1] / 4)
            matrix[i][i] = (alpha_bottom - alpha_top)
            matrix[i][i+1] = alpha_top
            matrix[i][i-1] = alpha_bottom
            pressures_z[i] = p_sub[i]
    pressures_result = solve(matrix, pressures_z)


    #pressure = p_sub
    #Non 100% resublimation missing

    return sublimated_mass, resublimated_mass, pressure, outgassed_mass/dt, empty_voxels, mass_flux

@njit
def calculate_molecule_surface(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, m_H2O, R_gas, VFF, r_grain, Phi, tortuosity, dx, dy, dz, dt, surface_reduced, avogadro_constant, k_B, sample_holder, water_mass_per_layer, n_x_lr, n_y_lr, n_z_lr, Dr, surface_reduction_factor):
    p_sub = np.zeros(np.shape(temperature), dtype=np.float64)
    sublimated_mass = np.zeros(np.shape(temperature), dtype=np.float64)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Placeholder
    outgassed_mass = 0
    mass_flux = np.zeros(np.shape(sublimated_mass), dtype=np.float64)
    #Replace surface_reduced with len(temperature.flatten() because it could technically be that deeper voxels are drained at the same time step
    empty_voxels = np.zeros((len(surface_reduced), 3), dtype=np.int32)
    empty_voxel_count = 0
    for each in surface_reduced:
        p_sub[each[2]][each[1]][each[0]] = 10 ** (a_1[0] + b_1[0] / temperature[each[2]][each[1]][each[0]] + c_1[0] * np.log10(temperature[each[2]][each[1]][each[0]]) + d_1[0] * temperature[each[2]][each[1]][each[0]])
        sublimated_mass[each[2]][each[1]][each[0]] = (p_sub[each[2]][each[1]][each[0]] - pressure[each[2]][each[1]][each[0]]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[each[2]][each[1]][each[0]])) * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dt#* (
                    #3 * VFF[each[2]][each[1]][each[0]] / r_grain * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]]) * surface_reduction_factor
        if sublimated_mass[each[2]][each[1]][each[0]] > water_mass_per_layer[each[2]][each[1]][each[0]]:
            sublimated_mass[each[2]][each[1]][each[0]] = water_mass_per_layer[each[2]][each[1]][each[0]]
            empty_voxels[empty_voxel_count] = np.array([each[0], each[1], each[2]], dtype=np.int32)
            empty_voxel_count += 1
        outgassed_mass += sublimated_mass[each[2]][each[1]][each[0]]
        mass_flux[each[2]][each[1]][each[0]] = sublimated_mass[each[2]][each[1]][each[0]]
        #p_sub[each[2]][each[1]][each[0]] = 0
    #pressure = p_sub
    #Non 100% resublimation missing

    return sublimated_mass, resublimated_mass, pressure, outgassed_mass/dt, empty_voxels[0:empty_voxel_count]


@njit
def calculate_molecule_surface_Q(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, m_H2O, dx, dy, dz, dt, surface_reduced, k_B, water_mass_per_layer, latent_heat_water):
    p_sub = np.zeros(np.shape(temperature), dtype=np.float64)
    sublimated_mass = np.zeros(np.shape(temperature), dtype=np.float64)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    S_c = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    # Placeholder
    outgassed_mass = 0
    mass_flux = np.zeros(np.shape(sublimated_mass), dtype=np.float64)
    # Replace surface_reduced with len(temperature.flatten() because it could technically be that deeper voxels are drained at the same time step
    empty_voxels = np.zeros((len(surface_reduced), 3), dtype=np.int32)
    empty_voxel_count = 0
    for each in surface_reduced:
        p_sub[each[2]][each[1]][each[0]] = 10 ** (
                    a_1[0] + b_1[0] / temperature[each[2]][each[1]][each[0]] + c_1[0] * np.log10(
                temperature[each[2]][each[1]][each[0]]) + d_1[0] * temperature[each[2]][each[1]][each[0]])
        sublimated_mass[each[2]][each[1]][each[0]] = (p_sub[each[2]][each[1]][each[0]] - pressure[each[2]][each[1]][
            each[0]]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[each[2]][each[1]][each[0]])) * \
                                                     dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][
                                                         each[0]] * dt  # * (
        # 3 * VFF[each[2]][each[1]][each[0]] / r_grain * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]]) * surface_reduction_factor
        if sublimated_mass[each[2]][each[1]][each[0]] > water_mass_per_layer[each[2]][each[1]][each[0]]:
            sublimated_mass[each[2]][each[1]][each[0]] = water_mass_per_layer[each[2]][each[1]][each[0]]
            water_mass_per_layer[each[2]][each[1]][each[0]] = 0
            empty_voxels[empty_voxel_count] = np.array([each[0], each[1], each[2]], dtype=np.int32)
            empty_voxel_count += 1
        outgassed_mass += sublimated_mass[each[2]][each[1]][each[0]]
        mass_flux[each[2]][each[1]][each[0]] = sublimated_mass[each[2]][each[1]][each[0]]
        # p_sub[each[2]][each[1]][each[0]] = 0
        S_c[each[2]][each[1]][each[0]] = - sublimated_mass[each[2]][each[1]][each[0]] * latent_heat_water[each[2]][each[1]][each[0]] / (dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])
    # pressure = p_sub
    # Non 100% resublimation missing
    return S_c, sublimated_mass, empty_voxels[0:empty_voxel_count], water_mass_per_layer

@njit
def calculate_molecule_flux_test(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, m_H2O, R_gas, VFF, r_grain, Phi, tortuosity, dx, dy, dz, dt, surface_reduced, avogadro_constant, k_B, sample_holder, water_mass_per_layer, n_x_lr, n_y_lr, n_z_lr, Dr, surface_reduction_factor, surface):
    p_sub = np.zeros(np.shape(temperature), dtype=np.float64)
    sublimated_mass = np.zeros(np.shape(temperature), dtype=np.float64)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Placeholder
    outgassed_mass = 0
    mass_flux = np.zeros(np.shape(sublimated_mass), dtype=np.float64)
    #Replace surface_reduced with len(temperature.flatten() because it could technically be that deeper voxels are drained at the same time step
    empty_voxels = np.zeros((len(surface_reduced), 3), dtype=np.int32)
    empty_voxel_count = 0
    for each in surface_reduced:
        p_sub[each[2]][each[1]][each[0]] = 10 ** (a_1[0] + b_1[0] / temperature[each[2]][each[1]][each[0]] + c_1[0] * np.log10(temperature[each[2]][each[1]][each[0]]) + d_1[0] * temperature[each[2]][each[1]][each[0]])
        sublimated_mass[each[2]][each[1]][each[0]] = (p_sub[each[2]][each[1]][each[0]] - pressure[each[2]][each[1]][each[0]]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[each[2]][each[1]][each[0]])) * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dt#* (
                    #3 * VFF[each[2]][each[1]][each[0]] / r_grain * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]]) * surface_reduction_factor
        if sublimated_mass[each[2]][each[1]][each[0]] > water_mass_per_layer[each[2]][each[1]][each[0]]:
            sublimated_mass[each[2]][each[1]][each[0]] = water_mass_per_layer[each[2]][each[1]][each[0]]
            empty_voxels[empty_voxel_count] = np.array([each[0], each[1], each[2]], dtype=np.int32)
            empty_voxel_count += 1
        outgassed_mass += sublimated_mass[each[2]][each[1]][each[0]]
        mass_flux[each[2]][each[1]][each[0]] = sublimated_mass[each[2]][each[1]][each[0]]
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if np.sum(surface[i][j][k]) == 0 and temperature[i][j][k] > 0:
                    p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    sublimated_mass[i][j][k] = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * dx[i][j][k] * dy[i][j][k] * dt
                '''if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                    sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                    empty_voxels[empty_voxel_count] = np.array([[i][j][k]], dtype=np.int32)
                    empty_voxel_count += 1
                outgassed_mass += sublimated_mass[i][j][k]
                mass_flux[i][j][k] = sublimated_mass[i][j][k]'''
        #p_sub[each[2]][each[1]][each[0]] = 0
    #pressure = p_sub
    #Non 100% resublimation missing

    return sublimated_mass, resublimated_mass, pressure, outgassed_mass/dt, empty_voxels[0:empty_voxel_count]


@njit
def calculate_molecule_flux_test_Q(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, m_H2O, R_gas, VFF, r_grain, Phi, tortuosity, dx, dy, dz, dt, surface_reduced, avogadro_constant, k_B, sample_holder, water_mass_per_layer, n_x_lr, n_y_lr, n_z_lr, Dr, surface_reduction_factor, latent_heat_water, surface):
    p_sub = np.zeros(np.shape(temperature), dtype=np.float64)
    sublimated_mass = np.zeros(np.shape(temperature), dtype=np.float64)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    S_c = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    # Placeholder
    outgassed_mass = 0
    mass_flux = np.zeros(np.shape(sublimated_mass), dtype=np.float64)
    # Replace surface_reduced with len(temperature.flatten() because it could technically be that deeper voxels are drained at the same time step
    empty_voxels = np.zeros((len(surface_reduced), 3), dtype=np.int32)
    empty_voxel_count = 0
    for each in surface_reduced:
        p_sub[each[2]][each[1]][each[0]] = 10 ** (
                    a_1[0] + b_1[0] / temperature[each[2]][each[1]][each[0]] + c_1[0] * np.log10(
                temperature[each[2]][each[1]][each[0]]) + d_1[0] * temperature[each[2]][each[1]][each[0]])
        sublimated_mass[each[2]][each[1]][each[0]] = (p_sub[each[2]][each[1]][each[0]] - pressure[each[2]][each[1]][
            each[0]]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[each[2]][each[1]][each[0]])) * \
                                                     dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][
                                                         each[0]] * dt  # * (
        # 3 * VFF[each[2]][each[1]][each[0]] / r_grain * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]]) * surface_reduction_factor
        if sublimated_mass[each[2]][each[1]][each[0]] > water_mass_per_layer[each[2]][each[1]][each[0]]:
            sublimated_mass[each[2]][each[1]][each[0]] = water_mass_per_layer[each[2]][each[1]][each[0]]
            empty_voxels[empty_voxel_count] = np.array([each[0], each[1], each[2]], dtype=np.int32)
            empty_voxel_count += 1
        outgassed_mass += sublimated_mass[each[2]][each[1]][each[0]]
        mass_flux[each[2]][each[1]][each[0]] = sublimated_mass[each[2]][each[1]][each[0]]
        # p_sub[each[2]][each[1]][each[0]] = 0
        S_c[each[2]][each[1]][each[0]] = - sublimated_mass[each[2]][each[1]][each[0]] * latent_heat_water[each[2]][each[1]][each[0]] / (dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if np.sum(surface[i][j][k]) == 0 and temperature[i][j][k] > 0:
                    p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    sublimated_mass[i][j][k] = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * dx[i][j][k] * dy[i][j][k] * dt
                    S_c[i][j][k] = - sublimated_mass[i][j][k] * latent_heat_water[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
    # pressure = p_sub
    # Non 100% resublimation missing
    return S_c, sublimated_mass

@njit
def calculate_molecule_flux_moon(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, m_H2O, dx, dy, dz, dt, k_B, sample_holder, water_mass_per_layer, latent_heat_water, water_particle_number, r_mono_water):
    p_sub = np.zeros(np.shape(temperature), dtype=np.float64)
    sublimated_mass = np.zeros(np.shape(temperature), dtype=np.float64)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    S_c = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    # Placeholder
    outgassed_mass = 0
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if sample_holder[i][j][k] == 0 and temperature[i][j][k] > 0:
                    p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    sublimated_mass[i][j][k] = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) * dt
                    if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                        sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                    S_c[i][j][k] = - sublimated_mass[i][j][k] * latent_heat_water[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    outgassed_mass += sublimated_mass[i][j][k]
    # pressure = p_sub
    return S_c, sublimated_mass, outgassed_mass


@njit
def calculate_molecule_flux_moon_test(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, m_H2O, dx, dy, dz, dt, k_B, sample_holder, water_mass_per_layer, latent_heat_water, water_particle_number, r_mono_water):
    p_sub = np.zeros(np.shape(temperature), dtype=np.float64)
    sublimated_mass = np.zeros(np.shape(temperature), dtype=np.float64)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    S_c = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_p = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    # Placeholder
    outgassed_mass = 0
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if sample_holder[i][j][k] == 0 and temperature[i][j][k] > 0:
                    p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    p_first_deriv = temperature[i][j][k]**(c_1[0]-2) * 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + d_1[0] * temperature[i][j][k]) * (np.log(10) * (d_1[0] * temperature[i][j][k] ** 2 - b_1[0]) + c_1[0] * temperature[i][j][k])
                    sublimated_mass[i][j][k] = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) * dt
                    if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                        sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                    if water_mass_per_layer[i][j][k] > 0:
                        S_c[i][j][k] = - np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * (3/2 * (p_sub[i][j][k] - pressure[i][j][k]) - p_first_deriv * temperature[i][j][k]) * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) * latent_heat_water[i][j][k] / (dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                        S_p[i][j][k] = - np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * (p_first_deriv - (p_sub[i][j][k] - pressure[i][j][k]) * 1/(2 * temperature[i][j][k])) * latent_heat_water[i][j][k] * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) / (dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    outgassed_mass += sublimated_mass[i][j][k]
    # pressure = p_sub
    return S_c, S_p, sublimated_mass, outgassed_mass


@njit
def calculate_molecule_flux_sintering(n_x, n_y, n_z, temperature, dx, dy, dz, dt, sample_holder, water_mass_per_layer, latent_heat_water, sublimated_mass):
    S_c = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_p = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    outgassed_mass = 0
    empty_voxels = np.zeros((n_x*n_y*n_z, 3), dtype=np.int32)
    empty_voxel_count = 0
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if sample_holder[i][j][k] == 0 and temperature[i][j][k] > 0:
                    if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                        sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                        empty_voxels[empty_voxel_count] = np.array([k, j, i], dtype=np.int32)
                        empty_voxel_count += 1
                    S_c[i][j][k] = - sublimated_mass[i][j][k] * latent_heat_water[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    if S_c[i][j][k] < 0:
                        S_p[i][j][k] = 3 * S_c[i][j][k] / temperature[i][j][k]
                        S_c[i][j][k] = - 2 * S_c[i][j][k]
                    outgassed_mass += sublimated_mass[i][j][k]
    # pressure = p_sub
    return S_c, S_p, empty_voxels[0:empty_voxel_count]


@njit
def calculate_source_terms_sintering_diffusion(n_x, n_y, n_z, temperature, dx, dy, dz, dt, sample_holder, water_mass_per_layer, latent_heat_water, sublimated_mass, gas_density):
    S_c_hte = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_p_hte = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_c_de = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_p_de = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    outgassed_mass = 0
    empty_voxels = np.zeros((n_x*n_y*n_z, 3), dtype=np.int32)
    empty_voxel_count = 0
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if sample_holder[i][j][k] == 0 and temperature[i][j][k] > 0:
                    if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                        sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                        empty_voxels[empty_voxel_count] = np.array([k, j, i], dtype=np.int32)
                        empty_voxel_count += 1
                    S_c_hte[i][j][k] = - sublimated_mass[i][j][k] * latent_heat_water[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    if S_c_hte[i][j][k] < 0:
                        S_p_hte[i][j][k] = 3 * S_c_hte[i][j][k] / temperature[i][j][k]
                        S_c_hte[i][j][k] = - 2 * S_c_hte[i][j][k]
                    S_c_de[i][j][k] = sublimated_mass[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    if S_c_de[i][j][k] < 0:
                        S_p_de[i][j][k] = 3 * S_c_de[i][j][k] / gas_density[i][j][k]
                        S_c_de[i][j][k] = - 2 * S_c_de[i][j][k]
                    outgassed_mass += sublimated_mass[i][j][k]
    # pressure = p_sub
    return S_c_hte, S_p_hte, S_c_de, S_p_de, empty_voxels[0:empty_voxel_count]


@njit
def sintered_surface_checker(n_x, n_y, n_z, r_n, r_p):
    blocked_lanes = np.full((n_z, n_y, n_x), 1, dtype=np.int32)
    for j in range(0, n_y):
        for k in range(0, n_x):
            for i in range(0, n_z):
                if r_n[i][j][k] > r_p[i][j][k]:
                    blocked_lanes[i][j][k] = -1
                    blocked_lanes[i+1:n_z, j, k] = 0
                    break
    return blocked_lanes


@njit(parallel=True)
def diffusion_parameters(n_x, n_y, n_z, a_1, b_1, c_1, d_1, temperature, temps, m_mol, R_gas, VFF, r_mono, Phi, q, pressure, m_H2O, k_B, dx, dy, dz, Dr, dt, sample_holder):
    diffusion_coefficient = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
    p_sub = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    sublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Using Güttler et al. 2023 calculation for q together with Phi = 13/6
    q = 1.60 - 0.73 * (1 - VFF)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] == 0 and (temperature[i + 1][j][k] + temperature[i - 1][j][k] + temperature[i][j + 1][k] + temperature[i][j - 1][k] + temperature[i][j][k + 1] + temperature[i][j][k - 1]) != 0:
                    temps[i][j][k][4] = temperature[i][j][k + 1] + (temperature[i][j][k] - temperature[i][j][k + 1]) / Dr[i][j][k][4] * 1/2 * dx[i][j][k + 1] * (1 - sample_holder[i][j][k+1])
                    temps[i][j][k][5] = temperature[i][j][k] + (temperature[i][j][k - 1] - temperature[i][j][k]) / Dr[i][j][k][5] * 1 / 2 * dx[i][j][k] * (1 - sample_holder[i][j][k-1])
                    temps[i][j][k][2] = temperature[i][j + 1][k] + (temperature[i][j][k] - temperature[i][j + 1][k]) / Dr[i][j][k][2] * 1 / 2 * dy[i][j + 1][k] * (1 - sample_holder[i][j+1][k])
                    temps[i][j][k][3] = temperature[i][j][k] + (temperature[i][j - 1][k] - temperature[i][j][k]) / Dr[i][j][k][3] * 1 / 2 * dy[i][j][k] * (1 - sample_holder[i][j-1][k])
                    temps[i][j][k][0] = temperature[i + 1][j][k] + (temperature[i][j][k] - temperature[i + 1][j][k]) / Dr[i][j][k][0] * 1 / 2 * dz[i + 1][j][k] * (1 - sample_holder[i+1][j][k])
                    temps[i][j][k][1] = temperature[i][j][k] + (temperature[i - 1][j][k] - temperature[i][j][k]) / Dr[i][j][k][1] * 1 / 2 * dz[i][j][k] * (1 - sample_holder[i-1][j][k])
                    for a in range(len(temps[i][j][k])):
                        #diff_coeff = permeability * (porosity/(R*T))**-1
                        if temps[i][j][k][a] == 0:
                            diffusion_coefficient[i][j][k][a] = 0
                        else:
                            diffusion_coefficient[i][j][k][a] = (1/(R_gas * temps[i][j][k][a]))**(-1) * 1/np.sqrt(2 * np.pi * m_mol * R_gas * temps[i][j][k][a]) * (1 - VFF[i][j][k])**2 * 2 * r_mono/(3 * (1 - (1 - VFF[i][j][k]))) * 4 / (Phi * q[i][j][k])
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] != 1:
                    p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    sublimated_mass[i][j][k] = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * (3 * VFF[i][j][k] / r_mono * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]) * dt
                    '''Permeability needs to be an interface parameter like Lambda, so VFF needs also be calculated on the interface. r_mono should be r_p from sintering. And look up calculation of D from k_m0'''
                    for a in range(len(temps[i][j][k])):
                        #diff_coeff = permeability * (porosity/(R*T))**-1
                        diffusion_coefficient[i][j][k][a] = (1/(R_gas * temps[i][j][k][a]))**(-1) * 1/np.sqrt(2 * np.pi * m_mol * R_gas * temps[i][j][k][a]) * (1 - VFF[i][j][k])**2 * 2 * r_mono/(3 * (1 - (1 - VFF[i][j][k]))) * 4 / (Phi * q[i][j][k])
    return diffusion_coefficient, p_sub, sublimated_mass


@njit(parallel=True)
def diffusion_parameters_sintering(n_x, n_y, n_z, a_1, b_1, c_1, d_1, temperature, temps, m_mol, R_gas, VFF, r_mono, Phi, pressure, m_H2O, k_B, dx, dy, dz, Dr, dt, sample_holder, blocked_voxels, n_x_arr, n_y_arr, n_z_arr):
    diffusion_coefficient = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
    p_sub = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    sublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Using Güttler et al. 2023 calculation for q together with Phi = 13/6
    q = 1.60 - 0.73 * (1 - VFF)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] == 0 and (temperature[i + 1][j][k] + temperature[i - 1][j][k] + temperature[i][j + 1][k] + temperature[i][j - 1][k] + temperature[i][j][k + 1] + temperature[i][j][k - 1]) != 0:
                    temps[i][j][k][4] = temperature[i][j][k + 1] + (temperature[i][j][k] - temperature[i][j][k + 1]) / Dr[i][j][k][4] * 1/2 * dx[i][j][k + 1] * (1 - sample_holder[i][j][k+1])
                    temps[i][j][k][5] = temperature[i][j][k] + (temperature[i][j][k - 1] - temperature[i][j][k]) / Dr[i][j][k][5] * 1 / 2 * dx[i][j][k] * (1 - sample_holder[i][j][k-1])
                    temps[i][j][k][2] = temperature[i][j + 1][k] + (temperature[i][j][k] - temperature[i][j + 1][k]) / Dr[i][j][k][2] * 1 / 2 * dy[i][j + 1][k] * (1 - sample_holder[i][j+1][k])
                    temps[i][j][k][3] = temperature[i][j][k] + (temperature[i][j - 1][k] - temperature[i][j][k]) / Dr[i][j][k][3] * 1 / 2 * dy[i][j][k] * (1 - sample_holder[i][j-1][k])
                    temps[i][j][k][0] = temperature[i + 1][j][k] + (temperature[i][j][k] - temperature[i + 1][j][k]) / Dr[i][j][k][0] * 1 / 2 * dz[i + 1][j][k] * (1 - sample_holder[i+1][j][k])
                    temps[i][j][k][1] = temperature[i][j][k] + (temperature[i - 1][j][k] - temperature[i][j][k]) / Dr[i][j][k][1] * 1 / 2 * dz[i][j][k] * (1 - sample_holder[i-1][j][k])
                    for a in range(len(temps[i][j][k])):
                        #diff_coeff = permeability * (porosity/(R*T))**-1
                        if temps[i][j][k][a] == 0:
                            diffusion_coefficient[i][j][k][a] = 0
                        else:
                            diffusion_coefficient[i][j][k][a] = (R_gas * temps[i][j][k][a]) * 1/np.sqrt(2 * np.pi * m_mol * R_gas * temps[i][j][k][a]) * (1 - VFF[i][j][k])**2 * 2 * r_mono[i][j][k]/(3 * (1 - (1 - VFF[i][j][k]))) * 4 / (Phi * q[i][j][k])
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] != 1:
                    temps[i][j][k][4] = temperature[i][j][k + 1] + (temperature[i][j][k] - temperature[i][j][k + 1]) / Dr[i][j][k][4] * 1/2 * dx[i][j][k + 1] * (1 - sample_holder[i][j][k+1])
                    temps[i][j][k][5] = temperature[i][j][k] + (temperature[i][j][k - 1] - temperature[i][j][k]) / Dr[i][j][k][5] * 1 / 2 * dx[i][j][k] * (1 - sample_holder[i][j][k-1])
                    temps[i][j][k][2] = temperature[i][j + 1][k] + (temperature[i][j][k] - temperature[i][j + 1][k]) / Dr[i][j][k][2] * 1 / 2 * dy[i][j + 1][k] * (1 - sample_holder[i][j+1][k])
                    temps[i][j][k][3] = temperature[i][j][k] + (temperature[i][j - 1][k] - temperature[i][j][k]) / Dr[i][j][k][3] * 1 / 2 * dy[i][j][k] * (1 - sample_holder[i][j-1][k])
                    temps[i][j][k][0] = temperature[i + 1][j][k] + (temperature[i][j][k] - temperature[i + 1][j][k]) / Dr[i][j][k][0] * 1 / 2 * dz[i + 1][j][k] * (1 - sample_holder[i+1][j][k])
                    temps[i][j][k][1] = temperature[i][j][k] + (temperature[i - 1][j][k] - temperature[i][j][k]) / Dr[i][j][k][1] * 1 / 2 * dz[i][j][k] * (1 - sample_holder[i-1][j][k])
                    p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    #sublimated_mass[i][j][k] = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * (3 * VFF[i][j][k] / r_mono * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]) * dt
                    '''Permeability needs to be an interface parameter like Lambda, so VFF needs also be calculated on the interface. r_mono should be r_p from sintering. And look up calculation of D from k_m0'''
                    for a in range(len(temps[i][j][k])):
                        #diff_coeff = permeability * (porosity/(R*T))**-1
                        diffusion_coefficient[i][j][k][a] = (R_gas * temps[i][j][k][a]) * 1/np.sqrt(2 * np.pi * m_mol * R_gas * temps[i][j][k][a]) * (1 - VFF[i][j][k])**2 * 2 * r_mono[i][j][k]/(3 * (1 - (1 - VFF[i][j][k]))) * 4 / (Phi * q[i][j][k])
                        if blocked_voxels[i + n_z_arr[a]][j + n_y_arr[a]][k + n_x_arr[a]] == -1:
                            diffusion_coefficient[i][j][k][a] = 0
                if blocked_voxels[i][j][k] == -1 and temperature[i-1][j][k] == 0:
                    diffusion_coefficient[i][j][k][1:5] = np.zeros(5, dtype=np.float64)
                elif blocked_voxels[i][j][k] == -1:
                    diffusion_coefficient[i][j][k] = np.zeros(6, dtype=np.float64)

    return diffusion_coefficient, p_sub


@njit(parallel=True)
def diffusion_parameters_sintering_periodic(n_x, n_y, n_z, a_1, b_1, c_1, d_1, temperature, m_mol, R_gas, VFF, r_mono, Phi, pressure, m_H2O, k_B, dx, dy, dz, Dr, dt, sample_holder, blocked_voxels, n_x_arr, n_y_arr, n_z_arr):
    diffusion_coefficient = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
    diff_coeff_center = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    p_sub = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    sublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Using Güttler et al. 2023 calculation for q together with Phi = 13/6
    q = 1.60 - 0.73 * (1 - VFF)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] > 0:
                    diff_coeff_center[i][j][k] = (R_gas * temperature[i][j][k]) * 1/np.sqrt(2 * np.pi * m_mol * R_gas * temperature[i][j][k]) * (1 - VFF[i][j][k])**2 * 2 * r_mono[i][j][k]/(3 * (1 - (1 - VFF[i][j][k]))) * 4 / (Phi * q[i][j][k])
    diff_coeff_center[1, 1:const.n_y-1, 1:const.n_x-1] = np.full((const.n_y-2, const.n_x-2), 1, dtype=np.float64)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if k == n_x-2:
                    diffusion_coefficient[i][j][k][4] = 2 * (diff_coeff_center[i][j][1] * diff_coeff_center[i][j][k]) / (diff_coeff_center[i][j][1] + diff_coeff_center[i][j][k])
                else:
                    diffusion_coefficient[i][j][k][4] = 2 * (diff_coeff_center[i][j][k + 1] * diff_coeff_center[i][j][k]) / (diff_coeff_center[i][j][k+1] + diff_coeff_center[i][j][k])
                if k == 1:
                    diffusion_coefficient[i][j][k][5] = 2 * (diff_coeff_center[i][j][k] * diff_coeff_center[i][j][n_x - 2]) / (diff_coeff_center[i][j][k] + diff_coeff_center[i][j][n_x-2])
                else:
                    diffusion_coefficient[i][j][k][5] = 2 * (diff_coeff_center[i][j][k] * diff_coeff_center[i][j][k - 1]) / (diff_coeff_center[i][j][k] + diff_coeff_center[i][j][k-1])
                if j == n_y-2:
                    diffusion_coefficient[i][j][k][2] = 2 * (diff_coeff_center[i][1][k] * diff_coeff_center[i][j][k]) / (diff_coeff_center[i][1][k] + diff_coeff_center[i][j][k])
                else:
                    diffusion_coefficient[i][j][k][2] = 2 * (diff_coeff_center[i][j + 1][k] * diff_coeff_center[i][j][k]) / (diff_coeff_center[i][j+1][k] + diff_coeff_center[i][j][k])
                if j == 1:
                    diffusion_coefficient[i][j][k][3] = 2 * (diff_coeff_center[i][j][k] * diff_coeff_center[i][n_y - 2][k]) / (diff_coeff_center[i][j][k] + diff_coeff_center[i][n_y-2][k])
                else:
                    diffusion_coefficient[i][j][k][3] = 2 * (diff_coeff_center[i][j][k] * diff_coeff_center[i][j - 1][k]) / (diff_coeff_center[i][j][k] + diff_coeff_center[i][j-1][k])
                diffusion_coefficient[i][j][k][0] = 2 * (diff_coeff_center[i + 1][j][k] * diff_coeff_center[i][j][k]) / (diff_coeff_center[i+1][j][k] + diff_coeff_center[i][j][k])
                diffusion_coefficient[i][j][k][1] = 2 * (diff_coeff_center[i][j][k] * diff_coeff_center[i - 1][j][k]) / (diff_coeff_center[i][j][k] + diff_coeff_center[i-1][j][k])
                if sample_holder[i][j][k] == 1:
                    diffusion_coefficient[i][j][k] = np.zeros(6, dtype=np.float64)
                for a in range(0, 6):
                    if blocked_voxels[i + n_z_arr[a]][j + n_y_arr[a]][k + n_x_arr[a]] == -1:
                        diffusion_coefficient[i][j][k][a] = 0
                    if sample_holder[i + n_z_arr[a]][j + n_y_arr[a]][k + n_x_arr[a]] == 1:
                        diffusion_coefficient[i][j][k][a] = 0

    return diffusion_coefficient, p_sub


@njit(parallel=True)
def diffusion_parameters_moon(n_x, n_y, n_z, a_1, b_1, c_1, d_1, temperature, temps, m_mol, R_gas, VFF, r_mono, Phi, q, pressure, m_H2O, k_B, dx, dy, dz, Dr, dt, sample_holder, sample_holder_diffusion, water_particle_number, r_mono_water):
    diffusion_coefficient = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
    p_sub = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    sublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Using Güttler et al. 2023 calculation for q together with Phi = 13/6
    q = 1.60 - 0.73 * (1 - VFF)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if sample_holder[i][j][k] == 1 and sample_holder_diffusion[i][j][k] != 1:
                    temps[i][j][k][4] = 0
                    temps[i][j][k][5] = 0
                    temps[i][j][k][2] = 0
                    temps[i][j][k][3] = 0
                    temps[i][j][k][0] = temperature[i + 1][j][k] + (temperature[i][j][k] - temperature[i + 1][j][k]) / Dr[i][j][k][0] * 1 / 2 * dz[i + 1][j][k] * (1 - sample_holder[i+1][j][k])
                    temps[i][j][k][1] = 0
                    for a in range(len(temps[i][j][k])):
                        #diff_coeff = permeability * (porosity/(R*T))**-1
                        if temps[i][j][k][a] == 0:
                            diffusion_coefficient[i][j][k][a] = 0
                        else:
                            diffusion_coefficient[i][j][k][a] = (1/(R_gas * temps[i][j][k][a]))**(-1) * 1/np.sqrt(2 * np.pi * m_mol * R_gas * temps[i][j][k][a]) * (1 - VFF[i][j][k])**2 * 2 * r_mono/(3 * (1 - (1 - VFF[i][j][k]))) * 4 / (Phi * q[i][j][k])
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] != 1:
                    p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    sublimated_mass[i][j][k] = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) * dt
                    '''Permeability needs to be an interface parameter like Lambda, so VFF needs also be calculated on the interface. r_mono should be r_p from sintering. And look up calculation of D from k_m0'''
                    for a in range(len(temps[i][j][k])):
                        #diff_coeff = permeability * (porosity/(R*T))**-1
                        diffusion_coefficient[i][j][k][a] = (1/(R_gas * temps[i][j][k][a]))**(-1) * 1/np.sqrt(2 * np.pi * m_mol * R_gas * temps[i][j][k][a]) * (1 - VFF[i][j][k])**2 * 2 * r_mono/(3 * (1 - (1 - VFF[i][j][k]))) * 4 / (Phi * q[i][j][k])
    return diffusion_coefficient, p_sub, sublimated_mass


@njit
def calculate_source_terms(n_x, n_y, n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, dt, surface_reduced, water_mass_per_layer, latent_heat_water, surface):
    S_c_hte = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_p_hte = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_c_de = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_p_de = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    outgassed_mass = 0
    mass_flux = np.zeros(np.shape(sublimated_mass), dtype=np.float64)
    # Replace surface_reduced with len(temperature.flatten() because it could technically be that deeper voxels are drained at the same time step
    empty_voxels = np.zeros((len(surface_reduced), 3), dtype=np.int32)
    empty_voxel_count = 0
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                '''if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                    sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                    empty_voxels[empty_voxel_count] = np.array([k, j, i], dtype=np.int32)
                    empty_voxel_count += 1'''
                outgassed_mass += sublimated_mass[i][j][k]
                # p_sub[each[2]][each[1]][each[0]] = 0
                S_c_hte[i][j][k] = - sublimated_mass[i][j][k] * latent_heat_water[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                if S_c_hte[i][j][k] < 0:
                    S_p_hte[i][j][k] = 3 * S_c_hte[i][j][k]/temperature[i][j][k]
                    S_c_hte[i][j][k] = - 2 * S_c_hte[i][j][k]
                S_c_de[i][j][k] = sublimated_mass[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                if S_c_de[i][j][k] < 0:
                    S_p_de[i][j][k] = 3 * S_c_de[i][j][k]/gas_density[i][j][k]
                    S_c_de[i][j][k] = - 2 * S_c_de[i][j][k]
    return S_c_hte, S_p_hte, S_c_de, S_p_de


@njit
def calculate_source_terms_linearised(n_x, n_y, n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, Dr, dt, surface_reduced, water_mass_per_layer, latent_heat_water, surface, m_H2O, k_b, a_1, b_1, c_1, d_1, sample_holder, water_particle_number, r_mono_water, diffusion_coefficients, Fluxkompensator):
    S_c_hte = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_p_hte = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_c_de = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    S_p_de = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    flux_corrector = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
    outgassed_mass = 0
    mass_flux = np.zeros(np.shape(sublimated_mass), dtype=np.float64)
    # Replace surface_reduced with len(temperature.flatten() because it could technically be that deeper voxels are drained at the same time step
    empty_voxels = np.zeros((len(surface_reduced), 3), dtype=np.int32)
    empty_voxel_count = 0
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                '''if sublimated_mass[i][j][k] > water_mass_per_layer[i][j][k]:
                    sublimated_mass[i][j][k] = water_mass_per_layer[i][j][k]
                    empty_voxels[empty_voxel_count] = np.array([k, j, i], dtype=np.int32)
                    empty_voxel_count += 1'''
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] == 0:
                    p = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    p_first_deriv = temperature[i][j][k]**(c_1[0]-2) * 10 ** (a_1[0] + b_1[0]/temperature[i][j][k] + d_1[0] * temperature[i][j][k]) * (np.log(10) * (d_1[0] * temperature[i][j][k]**2 - b_1[0]) + c_1[0] * temperature[i][j][k])
                    # p_sub[each[2]][each[1]][each[0]] = 0
                    S_c_hte[i][j][k] = - np.sqrt(m_H2O/(2 * np.pi * k_b * temperature[i][j][k])) * (3/2 * (p - pressure[i][j][k]) - p_first_deriv * temperature[i][j][k]) * latent_heat_water[i][j][k] * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) / (dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    S_p_hte[i][j][k] = - np.sqrt(m_H2O / (2 * np.pi * k_b * temperature[i][j][k])) * (p_first_deriv - (p - pressure[i][j][k]) * 1/(2 * temperature[i][j][k])) * latent_heat_water[i][j][k] * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) / (dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    #if i == 2 and j == 4 and k == 4:
                        #print(S_c_hte[i][j][k], S_p_hte[i][j][k])
                    '''if S_c_hte[i][j][k] < 0:
                        S_p_hte[i][j][k] = - (np.sqrt(m_H2O/(2 * np.pi * k_b * temperature[i][j][k])) * (3/2 * p + pressure[i][j][k] * temperature[i][j][k]) * latent_heat_water[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])) / temperature[i][j][k]
                        S_c_hte[i][j][k] = np.sqrt(m_H2O/(2 * np.pi * k_b * temperature[i][j][k])) * (p_first_deriv + 3/2 * pressure[i][j][k]) * latent_heat_water[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])'''
                    '''if i == 2 and j == 4 and k == 4:
                        print(S_c_hte[i][j][k], S_p_hte[i][j][k])'''
                    #S_c_de[i][j][k] = np.sqrt(m_H2O/(2 * np.pi * k_b * temperature[i][j][k])) * (3/2 * (p - pressure[i][j][k]) - p_first_deriv * temperature[i][j][k]) * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    #S_p_de[i][j][k] = np.sqrt(m_H2O / (2 * np.pi * k_b * temperature[i][j][k])) * (p_first_deriv - (p - pressure[i][j][k]) * 1/(2 * temperature[i][j][k])) * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    if Fluxkompensator:# and sublimated_mass[i][j][k] < (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - 6 * np.average(diffusion_coefficients[i][j][k]) * np.min(Dr[i][j][k])) * gas_density[i][j][k] * dt:
                        #print(sublimated_mass[i][j][k], (gas_density[i][j][k] / dt) * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                        flux_corrector[i][j][k] = (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - 6 * np.max(diffusion_coefficients[i][j][k]) * np.max(Dr[i][j][k])) * gas_density[i][j][k] * dt / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                        #if i == 2 and j == 4 and k == 4:
                            #print(flux_corrector[i][j][k], diffusion_coefficients[i][j][k])
                        #sublimated_mass[i][j][k] = sublimated_mass[i][j][k] + (dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - 6 * np.max(diffusion_coefficients[i][j][k]) * np.max(Dr[i][j][k])) * gas_density[i][j][k] * dt
                        #print('1.21 GigaWatts Marty')
                    #S_c_de[i][j][k] = sublimated_mass[i][j][k] / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    S_c_de[i][j][k] = (10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k]) - pressure[i][j][k]) * np.sqrt(m_H2O/(2 * np.pi * k_b * temperature[i][j][k])) * (water_particle_number[i][j][k] * 4 * np.pi * r_mono_water[i][j][k]**2) / (dx[i][j][k] * dy[i][j][k] * dz[i][j][k])
                    '''if i == 2 and j == 4 and k == 4:
                        print(S_c_de[i][j][k], flux_corrector[i][j][k], 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k]) * np.sqrt(m_H2O/(2 * np.pi * k_b * temperature[i][j][k])))'''
                    if S_c_de[i][j][k] < 0:
                        S_p_de[i][j][k] = 3 * S_c_de[i][j][k] / gas_density[i][j][k] + 2 * flux_corrector[i][j][k] / gas_density[i][j][k]
                        S_c_de[i][j][k] = - 2 * S_c_de[i][j][k] - 2 * flux_corrector[i][j][k]
                    #if i == 2 and j == 4 and k == 4:
                        #print(S_c_de[i][j][k], S_p_de[i][j][k])
                    outgassed_mass += sublimated_mass[i][j][k]
                    '''if S_c_de[i][j][k] < 0:
                        S_p_de[i][j][k] = - (np.sqrt(m_H2O/(2 * np.pi * k_b * temperature[i][j][k])) * (p_first_deriv + 3/2 * pressure[i][j][k]) * 1 / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]))/gas_density[i][j][k]
                        S_c_de[i][j][k] = np.sqrt(m_H2O/(2 * np.pi * k_b * temperature[i][j][k])) * (3/2 * p + pressure[i][j][k] * temperature[i][j][k]) * 1 / (dt * dx[i][j][k] * dy[i][j][k] * dz[i][j][k])'''
    return S_c_hte, S_p_hte, S_c_de, S_p_de


@njit(parallel=True)
def de_calculate(n_x, n_y, n_z, sh_adjacent_voxels, sample_holder, delta_gm_0, gas_mass, temperature, p_sub, Diffusion_coefficient, Dr, dx, dy, dz, dt, pressure, m_H2O, k_Boltzmann, VFF, r_mono):
    delta_gm = np.zeros((n_z, n_y, n_x), dtype=np.float64) + delta_gm_0
    Fourier_number = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    outgassed_mass = 0
    #Reflexive Randbedingung muss eingebaut werden!!!
    for i in prange(0, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] == 0 and (sample_holder[i+1][j][k] + sample_holder[i][j+1][k] + sample_holder[i][j-1][k] + sample_holder[i][j][k+1] + sample_holder[i][j][k-1]) == 0 and (temperature[i+1][j][k] + temperature[i][j+1][k] + temperature[i][j-1][k] + temperature[i][j][k+1] + temperature[i][j][k-1]) != 0:
                    outgassed_mass += ((((gas_mass[i][j][k + 1] - gas_mass[i][j][k]) * Diffusion_coefficient[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4]) / (Dr[i][j][k][4])) - ((gas_mass[i][j][k] - gas_mass[i][j][k - 1]) * Diffusion_coefficient[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5]) / (Dr[i][j][k][5]))) / dx[i][j][k]) * dt + \
                                        ((((gas_mass[i][j + 1][k] - gas_mass[i][j][k]) * Diffusion_coefficient[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2]) / (Dr[i][j][k][2]))
                                          - ((gas_mass[i][j][k] - gas_mass[i][j - 1][k]) * Diffusion_coefficient[i][j][k][3] * ( 1 - sh_adjacent_voxels[i][j][k][3]) / (
                                                 Dr[i][j][k][3]))) / dy[i][j][k]) * dt + \
                                        ((((gas_mass[i + 1][j][k] - gas_mass[i][j][k]) * Diffusion_coefficient[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0]) / (Dr[i][j][k][0]))
                                          - ((0) * Diffusion_coefficient[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1]) / (
                                                 Dr[i][j][k][1]))) / dz[i][j][k]) * dt
                    #outgassed_mass += delta_gm[i][j][k]
                    #delta_gm[i][j][k] = 0
                elif temperature[i][j][k] > 0 and sample_holder[i][j][k] != 1: #and gas_mass[i][j][k] > 0:
                    # Standard Thermal Diffusivity Equation 3D explicit
                    delta_gm[i][j][k] = ((((gas_mass[i][j][k + 1] - gas_mass[i][j][k]) * Diffusion_coefficient[i][j][k][4] * (1 - sh_adjacent_voxels[i][j][k][4]) / (Dr[i][j][k][4])) - ((gas_mass[i][j][k] - gas_mass[i][j][k - 1]) * Diffusion_coefficient[i][j][k][5] * (1 - sh_adjacent_voxels[i][j][k][5]) / (Dr[i][j][k][5]))) / dx[i][j][k]) * dt + \
                                       ((((gas_mass[i][j + 1][k] - gas_mass[i][j][k]) * Diffusion_coefficient[i][j][k][2] * (1 - sh_adjacent_voxels[i][j][k][2]) / (Dr[i][j][k][2]))
                                         - ((gas_mass[i][j][k] - gas_mass[i][j - 1][k]) * Diffusion_coefficient[i][j][k][3] * (1 - sh_adjacent_voxels[i][j][k][3]) / (
                                               Dr[i][j][k][3]))) / dy[i][j][k]) * dt + \
                                       ((((gas_mass[i + 1][j][k] - gas_mass[i][j][k]) * Diffusion_coefficient[i][j][k][0] * (1 - sh_adjacent_voxels[i][j][k][0]) / (Dr[i][j][k][0]))
                                         - ((gas_mass[i][j][k] - gas_mass[i - 1][j][k]) * Diffusion_coefficient[i][j][k][1] * (1 - sh_adjacent_voxels[i][j][k][1]) / (
                                               Dr[i][j][k][1]))) / dz[i][j][k]) * dt + (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O/(2 * np.pi * k_Boltzmann * temperature[i][j][k])) * 3 * VFF[i][j][k]/r_mono * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] * dt
                    #Fourier_number[i][j][k] = np.max(Lambda[i][j][k]) / (density[i][j][k] * heat_capacity[i][j][k]) * dt * (1 / dx[i][j][k] ** 2 + 1 / dy[i][j][k] ** 2 + 1 / dz[i][j][k] ** 2)# [-]
                    #Latent_Heat_per_Layer[i] = - (j_leave[i] - j_inward[i]) * latent_heat_water * dt - (j_leave_co2[i] - j_inward_co2[i]) * latent_heat_co2 * dt
                    #Energy_Increase_per_Layer[i][j][k] = heat_capacity[i][j][k] * density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] * delta_T[i][j][k]  # [J]
    return delta_gm, outgassed_mass


@njit
def sinter_neck_calculation(r_n, dt, temperature, a_1, b_1, c_1, d_1, omega, surface_energy, R_gas, r_grain, alpha, m_mol, density, total_passed_time, pressure, m_H2O, k_B, k_factor):
    #p_sub = 10 ** (a_1[0] + b_1[0] / temperature + c_1[0] * np.log10(temperature) + d_1[0] * temperature)
    #print(omega, surface_energy, r_grain, alpha, total_passed_time)
    p_sub = 3.23E12 * np.exp(-6134.6/temperature)
    #m_H2O, k_B Hertz-Knudsen_eq ausprobieren
    #Z = (p_sub - pressure) * (1/np.sqrt(2 * np.pi * m_mol * R_gas * temperature))
    Z = (p_sub - pressure) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature))
    r_c = 2 * m_mol * surface_energy / (density * R_gas * temperature)
    r_p = r_grain - (r_grain/(r_grain - r_c)) * Z * total_passed_time / density
    delta = r_n**2 / (2 * (r_p - r_n)) * k_factor
    d_s = r_p * (alpha/2 + np.arctan(r_p/(r_n + delta)) - np.pi/2)
    rate = ((omega**2 * surface_energy * p_sub)/(R_gas * temperature) * 1/np.sqrt(2*np.pi * m_mol * R_gas * temperature) * d_s / (d_s + delta * np.arctan(r_p/(r_n + delta))) * (2/r_p + 1/delta - 1/r_n) - Z/density * np.exp(- r_c/(r_n * k_factor)))
    #rate = ((omega ** 2 * surface_energy * p_sub) / (R_gas * temperature) * 1 / np.sqrt(2 * np.pi * m_mol * R_gas * temperature) * d_s / (d_s + delta * np.arctan(r_p / (r_n + delta))) * (2 / r_p + 1 / delta - 1 / r_n) - Z / density * np.exp(- r_c / (delta)))
    r_n = r_n + dt * rate
    return r_n, rate, r_p


@njit
def sinter_neck_calculation_time_dependent(r_n, r_p, dt, temperature, a_1, b_1, c_1, d_1, omega, surface_energy, R_gas, r_grain, alpha, m_mol, density, pressure, m_H2O, k_B, k_factor, water_particle_number, blocked_voxels, n_x, n_y, n_z, sample_holder, dx, dy):
    p_sub = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    sublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] == 0:
                    if blocked_voxels[i][j][k] == -1:
                        sublimated_mass[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k]) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature[i][j][k])) * dx[i][j][k] * dy[i][j][k] * dt
                        p_sub[i][j][k] = 0
                        pressure[i][j][k] = 0
                    elif blocked_voxels[i][j][k] == 0:
                        p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                        #p_sub[i][j][k] = 3.23E12 * np.exp(-6134.6 / temperature[i][j][k]) * blocked_voxels[i][j][k]
                        pressure[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    else:
                        p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                        pressure[i][j][k] = 0
                    Z = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature[i][j][k]))
                    r_c = 2 * m_mol * surface_energy / (density[i][j][k] * R_gas * temperature[i][j][k])
                    r_p[i][j][k] = r_p[i][j][k] - Z * np.exp(r_c/r_p[i][j][k]) * dt / density[i][j][k]
                    #r_p = r_grain - (r_grain / (r_grain - r_c)) * Z * time*dt / density
                    delta = r_n[i][j][k]**2 / (2 * (r_p[i][j][k] - r_n[i][j][k])) * k_factor
                    d_s = r_p[i][j][k] * (alpha/2 + np.arctan(r_p[i][j][k]/(r_n[i][j][k] + delta)) - np.pi/2)
                    rate = ((omega**2 * surface_energy * p_sub[i][j][k])/(R_gas * temperature[i][j][k]) * 1/np.sqrt(2*np.pi * m_mol * R_gas * temperature[i][j][k]) * d_s / (d_s + delta * np.arctan(r_p[i][j][k]/(r_n[i][j][k] + delta))) * (2/r_p[i][j][k] + 1/delta - 1/r_n[i][j][k]) - Z/density[i][j][k] * np.exp(- r_c/(r_n[i][j][k] * k_factor)))
                    neck_area = 4 * np.pi * delta * ((r_n[i][j][k] + delta) * np.arcsin((r_p[i][j][k]*delta/(r_p[i][j][k] + delta) * 1/delta)) - r_p[i][j][k]*delta/(r_p[i][j][k] + delta))
                    cond_rate = (omega**2 * surface_energy * p_sub[i][j][k])/(R_gas * temperature[i][j][k]) * 1/np.sqrt(2*np.pi * m_mol * R_gas * temperature[i][j][k]) * d_s / (d_s + delta * np.arctan(r_p[i][j][k]/(r_n[i][j][k] + delta))) * (2/r_p[i][j][k] + 1/delta - 1/r_n[i][j][k]) * neck_area
                    #sublimated_mass = sublimated_mass + (Z * (water_particle_number * np.exp(r_c/r_p) * 4 * np.pi * r_p**2 + 3 * np.exp(-r_c/(r_n * k_factor)) * neck_area) - 3 * cond_rate) * dt
                    if blocked_voxels[i][j][k] == 1 and delta > 0:
                        sublimated_mass[i][j][k] = Z * (water_particle_number[i][j][k] * np.exp(r_c / r_p[i][j][k]) * 4 * np.pi * r_p[i][j][k] ** 2 + 3 * np.exp(-r_c / (r_n[i][j][k] * k_factor)) * neck_area - 3 * cond_rate) * dt
                        #sublimated_mass[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * dx[i][j][k] * dy[i][j][k]

                    #if np.isnan(sublimated_mass[i][j][k]):
                        #print(Z[i][j][k], water_particle_number[i][j][k], r_p[i][j][k], neck_area[i][j][k], cond_rate[i][j][k], delta[i][j][k], r_n[i][j][k])
                    r_n[i][j][k] = r_n[i][j][k] + dt * rate
    return r_n, r_p, sublimated_mass, pressure


@njit
def sinter_neck_calculation_time_dependent_diffusion(r_n, r_p, dt, temperature, a_1, b_1, c_1, d_1, omega, surface_energy, R_gas, r_grain, alpha, m_mol, density, pressure, m_H2O, k_B, k_factor, water_particle_number, blocked_voxels, n_x, n_y, n_z, sample_holder, dx, dy):
    p_sub = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    sublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    areas = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] == 0:
                    if blocked_voxels[i][j][k] == -1:
                        sublimated_mass[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k]) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature[i][j][k])) * dx[i][j][k] * dy[i][j][k]
                        areas[i][j][k] = dx[i][j][k] * dy[i][j][k]
                        p_sub[i][j][k] = 0
                    elif blocked_voxels[i][j][k] == 0:
                        p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                        #p_sub[i][j][k] = 3.23E12 * np.exp(-6134.6 / temperature[i][j][k]) * blocked_voxels[i][j][k]
                    else:
                        p_sub[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k])
                    Z = (p_sub[i][j][k] - pressure[i][j][k]) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature[i][j][k]))
                    r_c = 2 * m_mol * surface_energy / (density[i][j][k] * R_gas * temperature[i][j][k])
                    r_p[i][j][k] = r_p[i][j][k] - Z * np.exp(r_c/r_p[i][j][k]) * dt / density[i][j][k]
                    #r_p = r_grain - (r_grain / (r_grain - r_c)) * Z * time*dt / density
                    delta = r_n[i][j][k]**2 / (2 * (r_p[i][j][k] - r_n[i][j][k])) * k_factor
                    d_s = r_p[i][j][k] * (alpha/2 + np.arctan(r_p[i][j][k]/(r_n[i][j][k] + delta)) - np.pi/2)
                    rate = ((omega**2 * surface_energy * p_sub[i][j][k])/(R_gas * temperature[i][j][k]) * 1/np.sqrt(2*np.pi * m_mol * R_gas * temperature[i][j][k]) * d_s / (d_s + delta * np.arctan(r_p[i][j][k]/(r_n[i][j][k] + delta))) * (2/r_p[i][j][k] + 1/delta - 1/r_n[i][j][k]) - Z/density[i][j][k] * np.exp(- r_c/(r_n[i][j][k] * k_factor)))
                    neck_area = 4 * np.pi * delta * ((r_n[i][j][k] + delta) * np.arcsin((r_p[i][j][k]*delta/(r_p[i][j][k] + delta) * 1/delta)) - r_p[i][j][k]*delta/(r_p[i][j][k] + delta))
                    cond_rate = (omega**2 * surface_energy * p_sub[i][j][k])/(R_gas * temperature[i][j][k]) * 1/np.sqrt(2*np.pi * m_mol * R_gas * temperature[i][j][k]) * d_s / (d_s + delta * np.arctan(r_p[i][j][k]/(r_n[i][j][k] + delta))) * (2/r_p[i][j][k] + 1/delta - 1/r_n[i][j][k]) * neck_area
                    #sublimated_mass = sublimated_mass + (Z * (water_particle_number * np.exp(r_c/r_p) * 4 * np.pi * r_p**2 + 3 * np.exp(-r_c/(r_n * k_factor)) * neck_area) - 3 * cond_rate) * dt
                    if blocked_voxels[i][j][k] == 1 and delta > 0:
                        areas[i][j][k] = (water_particle_number[i][j][k] * np.exp(r_c / r_p[i][j][k]) * 4 * np.pi * r_p[i][j][k] ** 2 + 3 * np.exp(-r_c / (r_n[i][j][k] * k_factor)) * neck_area - 3 * cond_rate)
                        sublimated_mass[i][j][k] = Z * (water_particle_number[i][j][k] * np.exp(r_c / r_p[i][j][k]) * 4 * np.pi * r_p[i][j][k] ** 2 + 3 * np.exp(-r_c / (r_n[i][j][k] * k_factor)) * neck_area - 3 * cond_rate) * dt
                        #sublimated_mass[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k]) * np.sqrt(m_H2O / (2 * np.pi * k_B * temperature[i][j][k])) * dx[i][j][k] * dy[i][j][k]
                    #if np.isnan(sublimated_mass[i][j][k]):
                        #print(Z[i][j][k], water_particle_number[i][j][k], r_p[i][j][k], neck_area[i][j][k], cond_rate[i][j][k], delta[i][j][k], r_n[i][j][k])
                    r_n[i][j][k] = r_n[i][j][k] + dt * rate
    return r_n, r_p, sublimated_mass, areas


@njit
def sinter_neck_calculation_exp(r_n, dt, temperature_pa, temperature_su, temperature_mt, temperature_gas, a_1, b_1, c_1, d_1, omega, surface_energy, R_gas, r_grain, alpha, m_mol, density, total_passed_time, pressure, m_H2O, k_B, k_factor):
    #p_sub = 10 ** (a_1[0] + b_1[0] / temperature + c_1[0] * np.log10(temperature) + d_1[0] * temperature)
    #print(omega, surface_energy, r_grain, alpha, total_passed_time)
    p_sub_mt = 3.23E12 * np.exp(-6134.6/temperature_mt)
    p_sub_pa = 3.23E12 * np.exp(-6134.6/temperature_pa)
    p_sub_su = 3.23E12 * np.exp(-6134.6/temperature_su)
    #m_H2O, k_B Hertz-Knudsen_eq ausprobieren
    #Z = (p_sub - pressure) * (1/np.sqrt(2 * np.pi * m_mol * R_gas * temperature))
    Z_pa = (p_sub_pa) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature_pa)) - (pressure) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature_gas))
    Z_su = (p_sub_su) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature_su)) - (pressure) * np.sqrt(m_H2O/(2 * np.pi * k_B * 160))
    r_p = r_grain - (r_grain/(r_grain - (2 * m_mol * surface_energy / (density * R_gas * temperature_pa)))) * Z_pa * total_passed_time / density
    delta = r_n**2 / (2 * (r_p - r_n)) #* k_factor
    d_s = r_p * (alpha/2 + np.arctan(r_p/(r_n + delta)) - np.pi/2)
    #rate = ((omega**2 * surface_energy * p_sub_mt)/(R_gas * temperature_mt) * 1/np.sqrt(2*np.pi * m_mol * R_gas * temperature_mt) * d_s / (d_s + delta * np.arctan(r_p/(r_n + delta))) * (2/r_p + 1/delta - 1/r_n) - Z_su/density * np.exp(- (2 * m_mol * surface_energy / (density * R_gas * temperature_su)) / (delta * k_factor)))
    rate = ((omega ** 2 * surface_energy * p_sub_mt) / (R_gas * temperature_mt) * 1 / np.sqrt(2 * np.pi * m_mol * R_gas * temperature_mt) * d_s / (d_s + delta * np.arctan(r_p / (r_n + delta))) * (2 / r_p + 1 / delta - 1 / r_n) - Z_su / density * np.exp(- (2 * m_mol * surface_energy / (density * R_gas * temperature_su)) / (r_n * k_factor)))
    r_n = r_n + dt * rate
    return r_n, rate, r_p

@njit
def gas_mass_function(T, pressure, VFF, dx, dy, dz, target_mass):
    return ((10 ** (const.lh_a_1[0] + const.lh_b_1[0] / T + const.lh_c_1[0] * np.log10(T) + const.lh_d_1[0] * T)) - pressure) * np.sqrt(const.m_H2O / (2 * np.pi * const.k_boltzmann * T)) * (3 * VFF / const.r_mono * dx * dy * dz) * const.dt - target_mass


@njit
def pressure_calculation(n_x, n_y, n_z, temperature, gas_density, k_boltzmann, m_H2O, VFF, r_mono, dx, dy, dz, dt, sample_holder, sublimated_mass, water_particle_number, areas):
    pressure = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for a in range(0, n_z):
        for b in range(0, n_y):
            for c in range(0, n_x):
                if temperature[a][b][c] > 0 and sample_holder[a][b][c] != 1 and areas[a][b][c] > 0:
                    #pressure[a][b][c] = (gas_density[a][b][c] * dx[a][b][c] * dy[a][b][c] * dz[a][b][c]) * np.sqrt(2 * np.pi * k_boltzmann * temperature[a][b][c] / m_H2O) * 1 / ((3 * VFF[a][b][c] / r_mono * dx[a][b][c] * dy[a][b][c] * dz[a][b][c]) * dt)
                    pressure[a][b][c] = (gas_density[a][b][c] * dx[a][b][c] * dy[a][b][c] * dz[a][b][c]) * np.sqrt(2 * np.pi * k_boltzmann * temperature[a][b][c] / m_H2O) * (1) / (areas[a][b][c]) * 1/dt
    return pressure


@njit
def Tsinter_neck_calculation_time_dependent_diffusion(r_n, r_p, dt, temperature, a_1, b_1, c_1, d_1, omega, surface_energy, R_gas, r_grain, alpha, m_mol, density, pressure, m_H2O, k_B, k_factor, water_particle_number, blocked_voxels, n_x, n_y, n_z, sample_holder, dx, dy, dz):
    p_sub = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    sublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    areas = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] == 0:
                    sublimated_mass[i][j][k] = 10 ** (a_1[0] + b_1[0] / temperature[i][j][k] + c_1[0] * np.log10(temperature[i][j][k]) + d_1[0] * temperature[i][j][k]) * np.sqrt(m_H2O/(2 * np.pi * k_B * temperature[i][j][k])) * dx[i][j][k] * dy[i][j][k] * dt
                    areas[i][j][k] = dx[i][j][k] * dy[i][j][k]
    return r_n, r_p, sublimated_mass, areas