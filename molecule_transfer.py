import numpy as np
from numba import jit
#import main
import constants as const
import variables_and_arrays as var

'''
Subroutine within calculate_molecule_flux to calcute the sublimating molecule within one layer of the system.

Input parameters:
	sub_coeff_a : float
		Empirical coefficient for the sublimation pressure calculation
	sub_coeff_b : float
		Empirical coefficient for the sublimation pressure calculation
	mass : float
	    Particle mass of the sublimating volatile
	T : float
	    Temperature of layer
	i : float
	    Number of the current numerical layer
	dx : float
	    Thickness of the numerical layer
	k_boltzmann : float
	    Boltzmann constant
	b : float
	    Coefficient used in calculating the efficiency function in Gundlach et al. (2020) equating to 4 times the diffusion scale length
	
Returns:
    Sublimating molecules of a volatile within one numerical layer dictated by the input parameters
'''
@jit
def j_leave_calculation(p_sub, mass, T, i, dx, k_boltzmann, b, reduction):
    return p_sub * np.sqrt(
                mass / (2 * np.pi * k_boltzmann * T)) * (1 + (i * dx) / b) ** (-1) * reduction # [kg/(m^2 * s)]


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
@jit
def calculate_molecule_flux(temperature, j_leave, j_leave_co2, a_H2O, b_H2O, m_H2O, k_boltzmann, b, water_content_per_layer, avogadro_constant, molar_mass_water, dt, dx, n, p_sub, co2_content_per_layer, a_CO2, b_CO2, m_CO2, molar_mass_co2, diffusion_factors, deeper_diffusion, deeper_diffusion_co2, pressure, pressure_co2):
    j_leave_overshoot = np.zeros((n_z, n_y, n_x))
    j_leave = p_sub * np.sqrt(m_H2O / (2 * np.pi * k_boltzmann * temperature)) * (1 + (depth_x + depth_y + depth_z) / b) ** (-1) # [kg/(m^2 * s)]
    check_overshoot = j_leave * dt > (water_content_per_layer / avogadro_constant) * molar_mass_water
    if check_overshoot.any():
        for indices, each in check_overshoot:
            if each:
                j_leave_overshoot[indices[0] + 1][indices[1]][indices[2]] = j_leave[indices[0]][indices[1]][indices[2]] - water_content_per_layer[indices[0]][indices[1]][indices[2]] / (avogadro_constant * dt) * molar_mass_water

    for i in range(1, n+1):
        if j_leave_overshoot[i+1] != 0:
            j_leave_overshoot[i+1] += j_leave_overshoot[i]
            j_leave_overshoot[i] = 0
        else:
            break

    j_inward = np.zeros(n + 1)
    j_inward_co2 = np.zeros(n + 1)

    for i in range(0, n + 1): #Diese for Schleife sollte redundant sein
        for m in range(0, len(diffusion_factors)):
            if i + len(diffusion_factors) < n:
                j_inward[i + 1 + m] += (j_leave[i] + j_leave_overshoot[i])/ 2 * diffusion_factors[m]
                j_inward_co2[i + 1 + m] += (j_leave_co2[i] + j_leave_co2_overshoot[i]) / 2 * diffusion_factors[m]
            else:
                if i + m < n:
                    j_inward[i + 1 + m] += (j_leave[i] + j_leave_overshoot[i]) / 2 * diffusion_factors[m]
                    j_inward_co2[i + 1 + m] += (j_leave_co2[i] + j_leave_co2_overshoot[i]) / 2 * diffusion_factors[m]
                else:
                    deeper_diffusion += (j_leave[i] + j_leave_overshoot[i]) / 2 * diffusion_factors[m]
                    deeper_diffusion_co2 += (j_leave_co2[i] + j_leave_co2_overshoot[i]) / 2 * diffusion_factors[m]

    return j_leave, j_inward, j_leave_co2, j_inward_co2, deeper_diffusion, deeper_diffusion_co2
