import numpy as np
from numba import jit, njit, prange
#import main
import constants as const
import variables_and_arrays as var


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
@njit(parallel=True)
def calculate_molecule_flux(n_x, n_y, n_z, temperature, pressure, a_1, b_1, c_1, d_1, mol_mass, R_gas, VFF, r_grain, Phi, tortuosity, dx, dy, dz, dt, surface_reduced, avogadro_constant, k_B, sample_holder, surrounding_surface):
    p_sub = 10 ** (a_1[0] + b_1[0] / temperature + c_1[0] * np.log10(temperature) + d_1[0] * temperature)
    sublimated_mass = (p_sub - pressure) * np.sqrt(mol_mass[0]/(2 * np.pi * R_gas * temperature)) * (3 * VFF / r_grain * dx * dy * dz)
    resublimated_mass = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    #Placeholder
    outgassed_mass = 0
    for each in surface_reduced:
        #Setting p_surface to zero since outgassing can be assumed to always happen towards the vacuum
        outgassed_mass += sublimated_mass[each[2]][each[1]][each[0]]
        p_sub[each[2]][each[1]][each[0]] = 0
    mass_flux = np.array(np.shape(sublimated_mass), dtype=np.float64)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] > 0:
                    diff_z = (1 - VFF[i][j][k]) * np.sqrt(1/(2 * np.pi * mol_mass[0] * R_gas * temperature[i][j][k])) * (p_sub[i-1][j][k] - p_sub[i+1][j][k])/(1 + 3 * (1 - (1 - VFF[i][j][k]))/(2 * (1 - VFF[i][j][k]) * r_grain) * Phi * tortuosity * dz[i][j][k] / 4)
                    diff_y = (1 - VFF[i][j][k]) * np.sqrt(1/(2 * np.pi * mol_mass[0] * R_gas * temperature[i][j][k])) * (p_sub[i][j-1][k] - p_sub[i][j+1][k])/(1 + 3 * (1 - (1 - VFF[i][j][k]))/(2 * (1 - VFF[i][j][k]) * r_grain) * Phi * tortuosity * dy[i][j][k] / 4)
                    diff_x = (1 - VFF[i][j][k]) * np.sqrt(1/(2 * np.pi * mol_mass[0] * R_gas * temperature[i][j][k])) * (p_sub[i][j][k-1] - p_sub[i][j][k+1])/(1 + 3 * (1 - (1 - VFF[i][j][k]))/(2 * (1 - VFF[i][j][k]) * r_grain) * Phi * tortuosity * dx[i][j][k] / 4)
                    if np.sum(sample_holder[i-1:i+1][j][k] != 0):
                        diff_z = 0
                    if np.sum(sample_holder[i][j-1:j+1][k] != 0):
                        diff_y = 0
                    if np.sum(sample_holder[i][j][k-1:k+1] != 0):
                        diff_x = 0
                    mass_flux[i-1][j][k] -= diff_z
                    mass_flux[i+1][j][k] += diff_z
                    mass_flux[i][j-1][k] -= diff_y
                    mass_flux[i][j+1][k] += diff_y
                    mass_flux[i][j][k-1] -= diff_x
                    mass_flux[i][j][k+1] += diff_x
                    if temperature[i-1][j][k] > temperature[i][j][k] and diff_z > 0:
                        resublimated_mass[i][j][k] += diff_z
                        mass_flux[i-1][j][k] = 0
                    elif temperature[i+1][j][k] > temperature[i][j][k] and diff_z < 0:
                        resublimated_mass[i][j][k] -= diff_z
                        mass_flux[i+1][j][k] = 0
                    if temperature[i][j-1][k] > temperature[i][j][k] and diff_y > 0:
                        resublimated_mass[i][j][k] += diff_y
                        mass_flux[i][j-1][k] = 0
                    elif temperature[i][j+1][k] > temperature[i][j][k] and diff_y < 0:
                        resublimated_mass[i][j][k] -= diff_y
                        mass_flux[i][j+1][k] = 0
                    if temperature[i][j][k-1] > temperature[i][j][k] and diff_x > 0:
                        resublimated_mass[i][j][k] += diff_x
                        mass_flux[i][j][k-1] = 0
                    elif temperature[i][j][k+1] > temperature[i][j][k] and diff_x < 0:
                        resublimated_mass[i][j][k] -= diff_x
                        mass_flux[i][j][k+1] = 0
                    p_sub[i-1][j][k] += mass_flux[i-1][j][k] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i-1][j][k] / dz[i-1][j][k]
                    p_sub[i+1][j][k] += mass_flux[i+1][j][k] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i+1][j][k] / dz[i+1][j][k]
                    p_sub[i][j-1][k] += mass_flux[i][j-1][k] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i][j-1][k] / dy[i][j-1][k]
                    p_sub[i][j+1][k] += mass_flux[i][j+1][k] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i][j+1][k] / dy[i][j+1][k]
                    p_sub[i][j][k-1] += mass_flux[i][j][k-1] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i][j][k-1] / dx[i][j][k-1]
                    p_sub[i][j][k+1] += mass_flux[i][j][k+1] * dt * mol_mass[0] / avogadro_constant * k_B * temperature[i][j][k+1] / dx[i][j][k+1]
    for each in surrounding_surface:
        #This should always be >= 1 since p_surface is set to 0
        outgassed_mass += max(mass_flux[each[2]][each[1]][each[0]], 0)
        p_sub[each[2]][each[1]][each[0]] = 0

    pressure = p_sub
    #Resublimation missing

    return sublimated_mass, resublimated_mass, pressure, outgassed_mass/dt
