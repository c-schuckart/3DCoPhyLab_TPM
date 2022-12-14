import numpy as np
from numba import jit
import variables_and_arrays as var
import constants as const
import settings as sett
from thermal_parameter_functions import lambda_ice_particles, lambda_pebble


'''
Calculation of the incoming energy for the current timestep and surface energy balance

Input parameters:
	solar_constant : float
		Solar constant
	r_H : float
    	Heliocentric distance of the body at the current time step
    tilt : float
    	Angle of surface normal of the simulated plane to sun
    albedo : float
    	Albedo of the body
    dt : float
	    Length of a timestep
	axial_tilt_factor : float
		Axial tilt/obliquity factor
	day_position : float
		...
    input_energy : float
    	Constant input energy, if that setting is chosen
    sigma : flaot
    	Stefan-Boltzmann constant
    epsilon : float
    	Emissivity of the body
    temperature : ndarray
		Temperature of the system at the current time step of dimension n+1
	Lambda : ndarray
		Array containing the total heat conductivity of each layer of dimension n
	DX : ndarray
		Array containing the distances between the mid-points of the numerical layers of dimension n
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
	heat_capacity : ndarray
	    Array containing the heat capacity of each numerical layer of dimension n+1
	surface_area : float
	    Simulated surface area, standard: 1 square meter
	density : ndarray
	    Array containing the density of each numerical layer of dimension n+1
	dx : ndarray
		Array containing the thickness of the numerical layers of dimension n+1
		
Returns:
    delta_T_0 : float
	    Difference of the temperature of the previous to the current timestep in the top layer
    Energy_Increase_per_Layer_0 : float
	    Increase of energy in the top layer
	E_In + E_Rad + E_Lat : float
		Energy flow without conductive heat transfer
	E_In : float
		Energy increase of the system due to solar radiation
'''
@jit
def energy_input(solar_constant, r_H, tilt, albedo, dt, axial_tilt_factor, day_position, input_energy, sigma, epsilon, temperature, Lambda, DX, j_leave, j_inward, latent_heat_water, j_leave_co2, j_inward_co2, latent_heat_co2, heat_capacity, surface_area, density, dx, time):
	E_In = input_energy / r_H ** 2 * (1 - albedo) * dt
	#E_In = input_energy * dt
	E_Rad = - sigma * epsilon * temperature[0]**4 * dt  # [J/(m^2)]
	E_Cond = Lambda[0] * (temperature[1] - temperature[0]) / DX[0] * dt
	E_Lat = - (j_leave[0] - j_inward[0]) * latent_heat_water * dt - (j_leave_co2[0] - j_inward_co2[0]) * latent_heat_co2 * dt
	E_Energy_Increase = E_In + E_Rad + E_Cond + E_Lat
	delta_T_0 = E_Energy_Increase / (heat_capacity * density[0] * dx[0]) * surface_area
	return delta_T_0, E_Energy_Increase, E_In, E_Rad, E_Lat