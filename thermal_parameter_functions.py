import numpy as np
from numba import jit
import constants as const
import variables_and_arrays as var


@jit
def lambda_ice_particles(n, temperature, DX, dx, lambda_water_ice, poisson_ratio_par, young_modulus_par, surface_energy_par, r_mono, f_1, f_2, VFF_pack, sigma, e_1, time, temperature_ini, lambda_water_ice_change):
	lambda_net = np.zeros(n)
	lambda_rad = np.zeros(n)
	for i in range(0, n):
		T = temperature[i+1] + (temperature[i]-temperature[i+1])/DX[i] * 1/2 * dx[i+1]
		lambda_net[i] = (lambda_water_ice / T) * (9 * np.pi / 4 * (1 - poisson_ratio_par ** 2) / young_modulus_par * surface_energy_par * r_mono ** 2) ** (1 / 3) * f_1 * np.exp(f_2 * VFF_pack[i]) / r_mono
		lambda_rad[i] = 16 / 3 * sigma * T ** 3 * e_1 * (1 - VFF_pack[i]) / VFF_pack[i] * r_mono
		#if time >= (3600 * 12.5):
			#if T > temperature_ini + 1:
				#lambda_net[i] = lambda_water_ice_change
				#lambda_rad[i] = 0
	lambda_total = (lambda_net + lambda_rad)
	return lambda_total

'''
Heat conductivity calculation for the pebble case based on Gundlach and Blum (2012).
Includes network heat conductivity for a pebble system with two size scales and radiative heat conductivity

Input parameters:
	n : float
		number of numerical layers
	temperature : ndarray
		Temperature of the system at the current time step of dimension n+1
	DX : ndarray
		Array containing the distances between the mid-points of the numerical layers of dimension n
	dx : ndarray
		Array containing the thickness of the numerical layers of dimension n+1
	VFF_agg_base : float
		Volume filling factor of the aggregates (VFF of the pebbles)
	poisson_ratio_agg : float
		Poisson ratio of the aggregates (pebbles)
	young_modulus_agg : float
		Young's modulus of the aggregates (pebbles) 
	r_agg : float
		Radius of the aggregates (pebbles)
	lambda_water_ice : float
		Heat conductivity of water ice at lower temperatures
	poisson_ratio_par : float
		Poisson ratio of the particles (within a pebble)
	young_modulus_par : float
		Young's modulus of the particles (within a pebble) 
	surface_energy_par : float
		Surface energy of the particles (within a pebble)
	r_mono : float
		Radius of the particles (within a pebble)
	dust_ice_ratio_per_layer : ndarray
		Array containing the ratio of dust to ices for each layer of dimension n+1
	lambda_solid : float
		Heat conductivity of the dust component of cometary material
	co2_h2o_ratio_per_layer : ndarray
		Array containing the ratio of CO2 ice to water ice for each layer of dimension n+1
	lambda_co2_ice : float
		Heat conductivity of the CO2 ice at low temperatures
	f_1 : float
		Fit coefficient for the packing structure coefficient in Gundlach and Blum (2012)
	f_2 : float
		Fit coefficient for the packing structure coefficient in Gundlach and Blum (2012)
	VFF_pack : ndarray
		Array containing the volume filling vactors of the packing (between the pebbles)
	sigma : float
		Stefan-Boltzmann constant
	e_1 : float
		Mean free path coefficient in Gundlach and Blum (2012)
		
Returns:
	lambda_total : ndarray
		Array containing the total heat conductivity of each layer of dimension n
'''
@jit
def lambda_pebble(n, temperature, DX, dx, VFF_agg_base, poisson_ratio_agg, young_modulus_agg, r_agg, lambda_water_ice, poisson_ratio_par, young_modulus_par, surface_energy_par, r_mono, dust_ice_ratio_per_layer, lambda_solid, co2_h2o_ratio_per_layer, lambda_co2_ice, f_1, f_2, VFF_pack, sigma, e_1):
	lambda_net = np.zeros(n)
	lambda_rad = np.zeros(n)
	for i in range(0, n):
		#Temperature calculated between the layers
		T = temperature[i + 1] + (temperature[i] - temperature[i + 1]) / DX[i] * 1/2 * dx[i + 1]
		surface_energy_agg = VFF_agg_base * surface_energy_par ** (5 / 3) * (
			9 * np.pi * (1 - poisson_ratio_agg ** 2) / (r_mono * young_modulus_par)) ** (2 / 3)
		lambda_agg = ((1 - dust_ice_ratio_per_layer[i]) * lambda_solid + (dust_ice_ratio_per_layer[i] * (1 - co2_h2o_ratio_per_layer[i])) * (lambda_water_ice / T) + (dust_ice_ratio_per_layer[i] * co2_h2o_ratio_per_layer[i]) * lambda_co2_ice) * (9 * np.pi / 4 * (
				1 - poisson_ratio_par ** 2) / young_modulus_par * surface_energy_par * r_mono ** 2) ** (
							 1 / 3) * f_1 * np.exp(f_2 * VFF_agg_base) / r_mono
		lambda_net[i] = lambda_agg * (9 * np.pi / 4 * (1 - poisson_ratio_agg ** 2) / young_modulus_agg * surface_energy_agg * r_agg ** 2) ** (1 / 3) * f_1 * np.exp(f_2 * VFF_pack[i]) / r_agg
		lambda_rad[i] = 16 / 3 * sigma * T ** 3 * e_1 * (1 - VFF_pack[i]) / VFF_pack[i] * r_agg
	lambda_total = lambda_net + lambda_rad
	return lambda_total

@jit
def lambda_constant(n_x, n_y, n_z, lambda_constant):
	return np.full((const.n_z, const.n_y, const.n_x, 6), lambda_constant, dtype=np.float64)

