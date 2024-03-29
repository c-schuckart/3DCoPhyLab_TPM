import numpy as np
from numba import jit, njit, prange
import constants as const
import variables_and_arrays as var


@njit(parallel=True)
def lambda_granular(n_x, n_y, n_z, temperature, Dr, dx, dy, dz, lambda_water_ice, poisson_ratio_par, young_modulus_par, activation_energy, R_gas, r_mono, f_1, f_2, VFF_pack, sigma, e_1, sample_holder, lambda_sample_holder, r_n):
	'''
	Heat conductivity calculation for granular media based on Gundlach and Blum (2012).
	Includes network heat conductivity for a granular system and radiative heat conductivity

	Input parameters:
		n_x : float
			number of numerical layers in x-direction
		n_y : float
			number of numerical layers in y-direction
		n_z : float
			number of numerical layers in x-direction
		temperature : ndarray
			Temperature of the system at the current time step of dimension n_z * n_y * n_x
		Dr : ndarray
			Array containing the distances between the mid-points of the numerical layers of dimension n_z * n_y * n_x * 6
		dx : ndarray
			Array containing the thickness of the numerical layers in x-direction of dimension n_z * n_y * n_x
		dy : ndarray
			Array containing the thickness of the numerical layers in y-direction of dimension n_z * n_y * n_x
		dz : ndarray
			Array containing the thickness of the numerical layers in z-direction of dimension n_z * n_y * n_x
		lambda_water_ice : float
			Heat conductivity of water ice grains at lower temperatures (Petrenko and Whitworth 1999)
		poisson_ratio_par : float
			Poisson ratio of the aggregates
		young_modulus_agg : float
			Young's modulus of the granular particles
		activation_energy : float
			Parameter needed to calculate the surface energy of water (Jabaud et al. 2023)
		R_gas : float
			Universal gas constant
		r_mono : ndarray
			Radius of the particles within a voxel of dimensions n_z * n_y * n_x
		f_1 : float
			Fit coefficient for the packing structure coefficient in Gundlach and Blum (2012)
		f_2 : float
			Fit coefficient for the packing structure coefficient in Gundlach and Blum (2012)
		VFF_pack : ndarray
			Array containing the volume filling factors of the voxels of dimension n_z * n_y * n_x
		sigma : float
			Stefan-Boltzmann constant
		e_1 : float
			Mean free path coefficient in Gundlach and Blum (2012)
		sample_holder : ndarray
			Array marking if a voxel belongs to the sample holder or not of  dimension n_z * n_y * n_x
		lambda_sample_holder : float
			Heat conductivity of the sample holder
		r_n : ndarray
			Array containing the sinter neck radii within each voxel of dimension n_z * n_y * n_x

	Returns:
		lambda_total : ndarray
			Array containing the total heat conductivity of each layer of dimension n_z * n_y * n_x * 6
		interface_temperatures : ndarray
			Array containing the interface temperatures between all voxel surfaces of dimension n_z * n_y * n_x * 6
	'''

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
					surface_energy_par = 0.17 * np.exp(- activation_energy / (R_gas * temps))
					if r_n[i][j][k] == 0:
						lambda_grain = (lambda_water_ice / temps) * (9 * np.pi / 4 * (1 - poisson_ratio_par ** 2) / young_modulus_par * surface_energy_par * r_mono[i][j][k] ** 2) ** (1 / 3) * f_1 * np.exp(f_2 * VFF_pack[i][j][k]) / r_mono[i][j][k]
					elif r_n[i][j][k] < r_mono[i][j][k]:
						lambda_grain = (lambda_water_ice / temps) * (3 / 4 * (1 - poisson_ratio_par ** 2) / young_modulus_par * np.sqrt(3/2 * np.pi * surface_energy_par * 2/3 * young_modulus_par * 1/(1 - poisson_ratio_par**2))) ** (1 / 3) * f_1 * np.exp(f_2 * VFF_pack[i][j][k]) * r_n[i][j][k]**(1/2) / r_mono[i][j][k]**(2/3)
					else:
						lambda_grain = (lambda_water_ice / temps) * (3 / 4 * (1 - poisson_ratio_par ** 2) / young_modulus_par * np.sqrt(3/2 * np.pi * surface_energy_par * 2/3 * young_modulus_par * 1/(1 - poisson_ratio_par**2))) ** (1 / 3) * f_1 * np.exp(f_2 * VFF_pack[i][j][k]) * r_n[i][j][k]**(1/2) / r_mono[i][j][k]**(2/3)
						#lambda_grain = (lambda_water_ice / temps)
					lambda_cond = lambda_grain
					for a in range(0, len(lambda_grain)):
						if sample_holder[i + var.n_z_lr[a]][j + var.n_y_lr[a]][k + var.n_x_lr[a]] == 1 or sample_holder[i][j][k] == 1:
							#lambda_cond[a] = 0.01
							lambda_cond[a] = ((lambda_grain[a] / (Dr[i][j][k][a] / 2) * lambda_sample_holder / (Dr[i][j][k][a] / 2)) / (lambda_grain[a] / (Dr[i][j][k][a] / 2) + lambda_sample_holder / (Dr[i][j][k][a] / 2))) * Dr[i][j][k][a]
						lambda_total[i][j][k][a] = lambda_cond[a] + 16 / 3 * sigma * temps[a] ** 3 * e_1 * (1 - VFF_pack[i][j][k]) / VFF_pack[i][j][k] * r_mono[i][j][k]
	return lambda_total, interface_temperatures


@njit(parallel=True)
def lambda_granular_periodic(n_x, n_y, n_z, temperature, Dr, dx, dy, dz, lambda_water_ice, poisson_ratio_par, young_modulus_par, surface_energy_par, r_mono, f_1, f_2, VFF_pack, sigma, e_1, sample_holder, lambda_sample_holder, r_n, enable_full_sintering, chi):
	lambda_total = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	lambda_centre = np.zeros((n_z, n_y, n_x), dtype=np.float64)
	lambda_cond = np.zeros(6, dtype=np.float64)
	lambda_grain = np.zeros(6, dtype=np.float64)
	r_n_interface = np.zeros(6, dtype=np.float64)
	r_mono_interface = np.zeros(6, dtype=np.float64)
	interface_temperatures = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	interface_rns = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	interface_rmonos = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	for i in prange(1, n_z-1):
		for j in range(1, n_y-1):
			for k in range(1, n_x-1):
				if temperature[i][j][k] > 0:
					#Temperature calculated between the layers
					if sample_holder[i][j][k] != 0:
						lambda_centre[i][j][k] = lambda_sample_holder
					elif r_n[i][j][k] > r_mono[i][j][k] and enable_full_sintering:
						lambda_centre[i][j][k] = lambda_water_ice/temperature[i][j][k]
					else:
						lambda_centre[i][j][k] = (lambda_water_ice / temperature[i][j][k]) * (3 / 4 * (1 - poisson_ratio_par ** 2) / young_modulus_par * np.sqrt(3 / 2 * np.pi * surface_energy_par * 2 / 3 * young_modulus_par * 1 / (1 - poisson_ratio_par ** 2))) ** (1 / 3) * chi[i][j][k] * r_n[i][j][k] ** (1 / 2) / r_mono[i][j][k] ** (2 / 3)
	#print(lambda_centre[1][25][25], lambda_centre[2][25][25], lambda_centre[3][25][25])
	for i in prange(1, n_z-1):
		for j in range(1, n_y-1):
			for k in range(1, n_x-1):
				if k == n_x-2:
					lambda_total[i][j][k][4] = 2 * (lambda_centre[i][j][1] * lambda_centre[i][j][k]) / (lambda_centre[i][j][1] + lambda_centre[i][j][k])
				else:
					lambda_total[i][j][k][4] = 2 * (lambda_centre[i][j][k + 1] * lambda_centre[i][j][k]) / (lambda_centre[i][j][k+1] + lambda_centre[i][j][k])
				if k == 1:
					lambda_total[i][j][k][5] = 2 * (lambda_centre[i][j][k] * lambda_centre[i][j][n_x - 2]) / (lambda_centre[i][j][k] + lambda_centre[i][j][n_x-2])
				else:
					lambda_total[i][j][k][5] = 2 * (lambda_centre[i][j][k] * lambda_centre[i][j][k - 1]) / (lambda_centre[i][j][k] + lambda_centre[i][j][k-1])
				if j == n_y-2:
					lambda_total[i][j][k][2] = 2 * (lambda_centre[i][1][k] * lambda_centre[i][j][k]) / (lambda_centre[i][1][k] + lambda_centre[i][j][k])
				else:
					lambda_total[i][j][k][2] = 2 * (lambda_centre[i][j + 1][k] * lambda_centre[i][j][k]) / (lambda_centre[i][j+1][k] + lambda_centre[i][j][k])
				if j == 1:
					lambda_total[i][j][k][3] = 2 * (lambda_centre[i][j][k] * lambda_centre[i][n_y - 2][k]) / (lambda_centre[i][j][k] + lambda_centre[i][n_y-2][k])
				else:
					lambda_total[i][j][k][3] = 2 * (lambda_centre[i][j][k] * lambda_centre[i][j - 1][k]) / (lambda_centre[i][j][k] + lambda_centre[i][j-1][k])
				lambda_total[i][j][k][0] = 2 * (lambda_centre[i + 1][j][k] * lambda_centre[i][j][k]) / (lambda_centre[i+1][j][k] + lambda_centre[i][j][k])
				lambda_total[i][j][k][1] = 2 * (lambda_centre[i][j][k] * lambda_centre[i - 1][j][k]) / (lambda_centre[i][j][k] + lambda_centre[i-1][j][k])
				for a in range(len(lambda_total[i][j][k])):
					if np.isnan(lambda_total[i][j][k][a]):
						lambda_total[i][j][k][a] = 0
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

@njit(parallel=False)
def thermal_conductivity_moon_regolith(n_x, n_y, n_z, temperature, dx, dy, dz, Dr, VFF, radius, fc1, fc2, fc3, fc4, fc5, mu, E, gamma, f1, f2, e1, chi, sigma, epsilon, water_mass, dust_mass, lambda_water_ice, lambda_sample_holder, sample_holder):
	#thermal conductivity is an interface property, evaluated at the layer boundaries
	#get temperature and vff at layer boundary
	lambda_total = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	lambda_solid = np.zeros(6, dtype=np.float64)
	lambda_net = np.zeros(6, dtype=np.float64)
	lambda_rad = np.zeros(6, dtype=np.float64)
	interface_temperatures = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	interface_VFFs = np.zeros((n_z, n_y, n_x, 6), dtype=np.float64)
	for i in prange(1, n_z-1):
		for j in range(1, n_y-1):
			for k in range(1, n_x-1):
				if temperature[i][j][k] > 0:
					T_x_pos = temperature[i][j][k + 1] + (temperature[i][j][k] - temperature[i][j][k + 1]) / Dr[i][j][k][4] * 1/2 * dx[i][j][k + 1]
					T_x_neg = temperature[i][j][k] + (temperature[i][j][k - 1] - temperature[i][j][k]) / Dr[i][j][k][5] * 1 / 2 * dx[i][j][k]
					T_y_pos = temperature[i][j + 1][k] + (temperature[i][j][k] - temperature[i][j + 1][k]) / Dr[i][j][k][2] * 1 / 2 * dy[i][j + 1][k]
					T_y_neg = temperature[i][j][k] + (temperature[i][j - 1][k] - temperature[i][j][k]) / Dr[i][j][k][3] * 1 / 2 * dy[i][j][k]
					T_z_pos = temperature[i + 1][j][k] + (temperature[i][j][k] - temperature[i + 1][j][k]) / Dr[i][j][k][0] * 1 / 2 * dz[i + 1][j][k]
					T_z_neg = temperature[i][j][k] + (temperature[i - 1][j][k] - temperature[i][j][k]) / Dr[i][j][k][1] * 1 / 2 * dz[i][j][k]
					temps = np.array([T_z_pos, T_z_neg, T_y_pos, T_y_neg, T_x_pos, T_x_neg])
					interface_temperatures[i][j][k] = temps
					VFF_x_pos = VFF[i][j][k + 1] + (VFF[i][j][k] - VFF[i][j][k + 1]) / Dr[i][j][k][4] * 1/2 * dx[i][j][k + 1]
					VFF_x_neg = VFF[i][j][k] + (VFF[i][j][k - 1] - VFF[i][j][k]) / Dr[i][j][k][5] * 1 / 2 * dx[i][j][k]
					VFF_y_pos = VFF[i][j + 1][k] + (VFF[i][j][k] - VFF[i][j + 1][k]) / Dr[i][j][k][2] * 1 / 2 * dy[i][j + 1][k]
					VFF_y_neg = VFF[i][j][k] + (VFF[i][j - 1][k] - VFF[i][j][k]) / Dr[i][j][k][3] * 1 / 2 * dy[i][j][k]
					VFF_z_pos = VFF[i + 1][j][k] + (VFF[i][j][k] - VFF[i + 1][j][k]) / Dr[i][j][k][0] * 1 / 2 * dz[i + 1][j][k]
					VFF_z_neg = VFF[i][j][k] + (VFF[i - 1][j][k] - VFF[i][j][k]) / Dr[i][j][k][1] * 1 / 2 * dz[i][j][k]
					cardinalVFF = np.array([VFF_z_pos, VFF_z_neg, VFF_y_pos, VFF_y_neg, VFF_x_pos, VFF_x_neg])
					#get temperature dependence of solid thermal conductivity
					for a in range(0, len(lambda_solid)):
						lambda_solid[a] = fc1*temps[a] + fc2*(temps[a])**2 + fc3*(temps[a])**3 + fc4*(temps[a])**4 + fc5*(temps[a])**5
						lambda_solid[a] = lambda_solid[a] * dust_mass[i][j][k] / (dust_mass[i][j][k] + water_mass[i][j][k]) + lambda_water_ice/temps[a] * water_mass[i][j][k] / (dust_mass[i][j][k] + water_mass[i][j][k])
						#network and radiative part of thermal conductivity
						lambda_net[a] = lambda_solid[a] * (9*np.pi/4*(1-mu**2)/E*gamma/radius)**(1/3)*(f1*np.exp(f2*cardinalVFF[a]))*chi
						lambda_rad[a] = 8*sigma*epsilon*temps[a]**3*e1*(1-cardinalVFF[a])/cardinalVFF[a]*radius
						if sample_holder[i + var.n_z_lr[a]][j + var.n_y_lr[a]][k + var.n_x_lr[a]] == 1 or sample_holder[i][j][k] == 1:
							lambda_net[a] = ((lambda_net[a] / (Dr[i][j][k][a] / 2) * lambda_sample_holder / (Dr[i][j][k][a] / 2)) / (lambda_net[a] / (Dr[i][j][k][a] / 2) + lambda_sample_holder / (Dr[i][j][k][a] / 2))) * Dr[i][j][k][a]
							if np.isnan(lambda_net[a]):
								lambda_net[a] = 0
							lambda_rad[a] = 0
						lambda_total[i][j][k][a] = lambda_net[a] + lambda_rad[a]
	return lambda_total, interface_temperatures


@njit()
def heat_capacity_moon_regolith(n_x, n_y, n_z, temperature, c0, c1, c2, c3, c4, water_mass, dust_mass, heat_capacity_sample_holder, sample_holder):
	heat_capacity = np.zeros((n_z, n_y, n_x), dtype=np.float64)
	for i in prange(1, n_z - 1):
		for j in range(1, n_y - 1):
			for k in range(1, n_x - 1):
				if temperature[i][j][k] > 0 and sample_holder[i][j][k] == 0:
					heat_capacity[i][j][k] = c0 + c1 * temperature[i][j][k] + c2 * temperature[i][j][k]**2 + c3 * temperature[i][j][k]**3 + c4 * temperature[i][j][k]**4
					heat_capacity[i][j][k] = heat_capacity[i][j][k] * dust_mass[i][j][k] / (dust_mass[i][j][k] + water_mass[i][j][k]) + (7.5 * temperature[i][j][k] + 90) * water_mass[i][j][k] / (dust_mass[i][j][k] + water_mass[i][j][k])
				elif sample_holder[i][j][k] == 1:
					heat_capacity[i][j][k] = heat_capacity_sample_holder
				else:
					heat_capacity[i][j][k] = 1
	return heat_capacity

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
	density_grain = 916.7 - 0.175 * (temperature - 273.15) - 5.0E-4 * (temperature - 273.15) ** 2
	return density_grain, density_grain * VFF


@njit
def calculate_bulk_density_and_VFF(temperature, VFF, dust_mass, water_mass, density_dust, dx, dy, dz):
	water_ice_grain_density = calculate_density(temperature, VFF)[0]
	bulk_density = (dust_mass + water_mass) / (dx * dy * dz)
	#VFF = ((water_mass + dust_mass) / water_ice_grain_density + (dust_mass + water_mass) / density_dust) / (dx * dy * dz)
	VFF = bulk_density / (water_ice_grain_density * (water_mass / (water_mass + dust_mass)) + density_dust * (dust_mass / (water_mass + dust_mass)))
	return bulk_density, VFF, water_ice_grain_density


@njit
def calculate_water_grain_radius(n_x, n_y, n_z, uniform_water_masses, water_ice_grain_density, water_particle_number, r_mono_water):
	for i in range(0, n_z):
		for j in range(0, n_y):
			for k in range(0, n_x):
				if uniform_water_masses[i][j][k] > 0:
					r_mono_water[i][j][k] = (uniform_water_masses[i][j][k] / (water_ice_grain_density[i][j][k] * water_particle_number[i][j][k] * 4 / 3 * np.pi)) ** (1 / 3)
				else:
					r_mono_water[i][j][k] = 0
	return r_mono_water
@njit
def thermal_functions(temperature, b_1, c_1, d_1, R_gas, m_mol, VFF):
	density_grain = 918 - 0.175 * (temperature - 273.15) - 5.0E-4 * (temperature - 273.15) ** 2
	return (7.5 * temperature + 90), ((-b_1[0] * np.log(10) + (c_1[0] - 1) * temperature + d_1[0] * np.log(10) * temperature**2) * R_gas / (m_mol[0])), density_grain, density_grain * VFF

@njit(parallel=True)
def lambda_sand(n_x, n_y, n_z, temperature, Dr, lambda_sand, sample_holder, lambda_sample_holder, sensor_positions):
	sample_holder = sample_holder + sensor_positions
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


@njit
def calculate_Q_sensor(n_x, n_y, n_z, lambda_copper, A, l, temperature, temperature_plug):
	Q = np.zeros((n_z, n_y, n_x), dtype=np.float64)
	Q[20][n_y // 2][n_x // 2] = - 2 * lambda_copper * A / l * (temperature[20][n_y // 2][n_x // 2] - temperature_plug)
	Q[40][n_y // 2][n_x // 2] = - 2 * lambda_copper * A / l * (temperature[40][n_y // 2][n_x // 2] - temperature_plug)
	Q[70][n_y // 2][n_x // 2] = - 2 * lambda_copper * A / l * (temperature[70][n_y // 2][n_x // 2] - temperature_plug)
	Q[110][n_y // 2][n_x // 2] = - 2 * lambda_copper * A / l * (temperature[110][n_y // 2][n_x // 2] - temperature_plug)
	Q[180][n_y // 2][n_x // 2] = - 2 * lambda_copper * A / l * (temperature[180][n_y // 2][n_x // 2] - temperature_plug)
	'''Q[5][n_y // 2][n_x // 2] = - 2 * lambda_copper * A / l * (temperature[5][n_y // 2][n_x // 2] - temperature_plug)
	Q[10][n_y//2][n_x//2] = - 2 * lambda_copper * A / l * (temperature[10][n_y//2][n_x//2] - temperature_plug)
	Q[17][n_y//2][n_x//2] = 2 * (- lambda_copper * A / l * (temperature[17][n_y//2][n_x//2] - temperature_plug)) / 2
	Q[18][n_y // 2][n_x // 2] = 2 * (- lambda_copper * A / l * (temperature[18][n_y // 2][n_x // 2] - temperature_plug)) / 2
	Q[27][n_y // 2][n_x // 2] = 2 * (- lambda_copper * A / l * (temperature[27][n_y // 2][n_x // 2] - temperature_plug)) / 2
	Q[28][n_y // 2][n_x // 2] = 2 * (- lambda_copper * A / l * (temperature[28][n_y // 2][n_x // 2] - temperature_plug)) / 2
	Q[45][n_y // 2][n_x // 2] = - 2 * lambda_copper * A / l * (temperature[45][n_y // 2][n_x // 2] - temperature_plug)'''
	return Q




