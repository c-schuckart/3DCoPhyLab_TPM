import numpy as np
from numba import jit, njit, prange
import constants as const
import variables_and_arrays as var


@njit(parallel=True)
def lambda_granular(n_x, n_y, n_z, temperature, Dr, dx, dy, dz, lambda_water_ice, poisson_ratio_par, young_modulus_par, surface_energy_par, r_mono, f_1, f_2, VFF_pack, sigma, e_1, sample_holder, lambda_sample_holder, r_n):
	lambda_total = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	lambda_cond = np.zeros(6, dtype=np.float64)
	interface_temperatures = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	for i in prange(1, n_z-1):
		for j in range(1, n_y-1):
			for k in range(1, n_x-1):
				if temperature[i][j][k] > 0:
					#Temperature calculated between the layers
					T_x_pos = temperature[i][j][k + 1] + (temperature[i][j][k] - temperature[i][j][k + 1]) / Dr[i][j][k][4] * 1/2 * dx[i][j][k + 1]
					T_x_neg = temperature[i][j][k] + (temperature[i][j][k - 1] - temperature[i][j][k]) / Dr[i][j][k][5] * 1 / 2 * dx[i][j][k]
					T_y_pos = temperature[i][j + 1][k] + (temperature[i][j][k] - temperature[i][j + 1][k]) / Dr[i][j][k][2] * 1 / 2 * dy[i][j + 1][k]
					T_y_neg = temperature[i][j][k] + (temperature[i][j - 1][k] - temperature[i][j][k]) / Dr[i][j][k][3] * 1 / 2 * dy[i][j][k]
					T_z_pos = temperature[i + 1][j][k] + (temperature[i][j][k] - temperature[i + 1][j][k]) / Dr[i][j][k][0] * 1 / 2 * dz[i + 1][j][k]
					T_z_neg = temperature[i][j][k] + (temperature[i - 1][j][k] - temperature[i][j][k]) / Dr[i][j][k][1] * 1 / 2 * dz[i][j][k]
					temps = np.array([T_z_pos, T_z_neg, T_y_pos, T_y_neg, T_x_pos, T_x_neg])
					interface_temperatures[i][j][k] = temps
					'''if i == 1 and j == 50 and k == 50:
						print(T_x_pos)
					if i == 1 and j == 50 and k == 51:
						print(T_x_neg)'''
					if r_n[i][j][k] == 0:
						lambda_grain = (lambda_water_ice / temps) * (9 * np.pi / 4 * (1 - poisson_ratio_par ** 2) / young_modulus_par * surface_energy_par * r_mono ** 2) ** (1 / 3) * f_1 * np.exp(f_2 * VFF_pack[i][j][k]) / r_mono
					else:
						lambda_grain = (lambda_water_ice / temps) * (3 / 4 * (1 - poisson_ratio_par ** 2) / young_modulus_par * np.sqrt(3/2 * np.pi * surface_energy_par * 2/3 * young_modulus_par * 1/(1 - poisson_ratio_par**2))) ** (1 / 3) * f_1 * np.exp(f_2 * VFF_pack[i][j][k]) * r_n[i][j][k]**(1/2) / r_mono**(2/3)
					lambda_cond = lambda_grain
					for a in range(0, len(lambda_grain)):
						if sample_holder[i + var.n_z_lr[a]][j + var.n_y_lr[a]][k + var.n_x_lr[a]] == 1 or sample_holder[i][j][k] == 1:
							#lambda_cond[a] = 0.01
							lambda_cond[a] = ((lambda_grain[a] / (Dr[i][j][k][a] / 2) * lambda_sample_holder / (Dr[i][j][k][a] / 2)) / (lambda_grain[a] / (Dr[i][j][k][a] / 2) + lambda_sample_holder / (Dr[i][j][k][a] / 2))) * Dr[i][j][k][a]
						lambda_total[i][j][k][a] = lambda_cond[a] + 16 / 3 * sigma * temps[a] ** 3 * e_1 * (1 - VFF_pack[i][j][k]) / VFF_pack[i][j][k] * r_mono
	return lambda_total, interface_temperatures


@njit(parallel=True)
def lambda_ice_block(n_x, n_y, n_z, temperature, Dr, dx, dy, dz, lambda_water_ice, r_mono, VFF_pack, sigma, e_1, sample_holder, lambda_sample_holder):
	lambda_total = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	lambda_cond = np.zeros(6, dtype=np.float64)
	for i in prange(1, n_z-1):
		for j in range(1, n_y-1):
			for k in range(1, n_x-1):
				if temperature[i][j][k] > 0:
					#Temperature calculated between the layers
					T_x_pos = temperature[i][j][k + 1] + (temperature[i][j][k] - temperature[i][j][k + 1]) / Dr[i][j][k][4] * 1/2 * dx[i][j][k + 1]
					T_x_neg = temperature[i][j][k] + (temperature[i][j][k - 1] - temperature[i][j][k]) / Dr[i][j][k][5] * 1 / 2 * dx[i][j][k]
					T_y_pos = temperature[i][j + 1][k] + (temperature[i][j][k] - temperature[i][j + 1][k]) / Dr[i][j][k][2] * 1 / 2 * dy[i][j + 1][k]
					T_y_neg = temperature[i][j][k] + (temperature[i][j - 1][k] - temperature[i][j][k]) / Dr[i][j][k][3] * 1 / 2 * dy[i][j][k]
					T_z_pos = temperature[i][j][k] + (temperature[i - 1][j][k] - temperature[i][j][k]) / Dr[i][j][k][1] * 1 / 2 * dz[i][j][k]
					T_z_neg = temperature[i + 1][j][k] + (temperature[i][j][k] - temperature[i + 1][j][k]) / Dr[i][j][k][0] * 1 / 2 * dz[i + 1][j][k]
					temps = np.array([T_z_pos, T_z_neg, T_y_pos, T_y_neg, T_x_pos, T_x_neg])
					lambda_grain = (lambda_water_ice / temps)
					lambda_cond = lambda_grain
					for a in range(0, len(lambda_grain)):
						if sample_holder[i + var.n_z_lr[a]][j + var.n_y_lr[a]][k + var.n_x_lr[a]] == 1:
							lambda_cond[a] = ((lambda_grain[a] / (Dr[i][j][k][a] / 2) * lambda_sample_holder / (Dr[i][j][k][a] / 2)) / (lambda_grain[a] / (Dr[i][j][k][a] / 2) + lambda_sample_holder / (Dr[i][j][k][a] / 2))) * Dr[i][j][k][a]
						lambda_total[i][j][k][a] = lambda_cond[a] + 16 / 3 * sigma * temps[a] ** 3 * e_1 * (1 - VFF_pack[i][j][k]) / VFF_pack[i][j][k] * r_mono
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
'''@jit
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
	return lambda_total'''

@jit
def lambda_constant(n_x, n_y, n_z, lambda_constant):
	return np.full((n_z, n_y, n_x, 6), lambda_constant, dtype=np.float64)


@jit
def calculate_heat_capacity(temperature):
	return (7.5 * temperature + 90) # [J/kg/K]


@jit
def calculate_latent_heat(temperature, b_1, c_1, d_1, R_gas, m_mol):
	return ((-b_1[0] * np.log(10) + (c_1[0] - 1) * temperature + d_1[0] * np.log(10) * temperature**2) * R_gas / (m_mol[0])) # [J/kg]


@njit
def calculate_density(temperature, VFF):
	density_grain = 918 - 0.13783 * temperature - 2.53451E-4 * temperature**2
	return density_grain, density_grain * VFF


@njit
def thermal_functions(temperature, b_1, c_1, d_1, R_gas, m_mol, VFF):
	density_grain = 918 - 0.13783 * temperature - 2.53451E-4 * temperature ** 2
	return (7.5 * temperature + 90), ((-b_1[0] * np.log(10) + (c_1[0] - 1) * temperature + d_1[0] * np.log(10) * temperature**2) * R_gas / (m_mol[0])), density_grain, density_grain * VFF

@njit(parallel=True)
def lambda_sand(n_x, n_y, n_z, temperature, Dr, lambda_sand, sample_holder, lambda_sample_holder):
	lambda_total = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	lambda_s = np.full(6, lambda_sand, dtype=np.float64)
	for i in prange(1, n_z-1):
		for j in range(1, n_y-1):
			for k in range(1, n_x-1):
				if temperature[i][j][k] > 0:
					for a in range(0, len(lambda_s)):
						if sample_holder[i + var.n_z_lr[a]][j + var.n_y_lr[a]][k + var.n_x_lr[a]] == 1 or sample_holder[i][j][k] == 1:
							lambda_total[i][j][k][a] = (lambda_s[a] / (Dr[i][j][k][a] / 2) * lambda_sample_holder / (Dr[i][j][k][a] / 2) / (lambda_s[a] / (Dr[i][j][k][a] / 2) + lambda_sample_holder / (Dr[i][j][k][a] / 2))) * Dr[i][j][k][a]
						else:
							lambda_total[i][j][k][a] = lambda_s[a]
	return lambda_total

@jit
def lambda_test(n_x, n_y, n_z, temperature, Dr, lambda_sand, sample_holder, lambda_sample_holder):
	lambda_total = np.zeros(np.shape(Dr), dtype=np.float64)
	lambda_s = np.full(6, lambda_sand, dtype=np.float64)
	lambda_s[2],lambda_s[3], lambda_s[4], lambda_s[5] = 0, 0, 0, 0
	for i in range(1, n_z-1):
		for j in range(0, 1):
			for k in range(0, 1):
				if temperature[i][j][k] > 0:
					for a in range(0, len(lambda_s)):
							lambda_total[i][j][k][a] = lambda_s[a]
	return lambda_total




