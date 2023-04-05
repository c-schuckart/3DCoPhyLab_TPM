import numpy as np
from numba import jit, njit, prange
from scipy import integrate
import variables_and_arrays as var
import constants as const
import settings as sett


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
def energy_input(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_reduced, delta_T):
	Energy_Increase_in_surface = 0
	E_In_in_surface = 0
	E_Rad_in_surface = 0
	delta_T_0 = np.zeros(np.shape(delta_T))
	E_Lat_in_surface = 0
	for each in surface_reduced:
		#input energy durch input_energy[each[2]][each[1]][each[0]] ersetzen, sobald genaue Abstrahlcharakteristik der Lampe berechnet
		#E_In = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * surface[each[2]][each[1]][each[0]][1]
		E_In = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * dt * surface[each[2]][each[1]][each[0]][1]
		#E_In = input_energy * dt
		E_Rad = - sigma * epsilon * temperature[each[2]][each[1]][each[0]]**4 * dt * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5])) # [J/(m^2)]
		E_Cond_z_pos = Lambda[each[2]][each[1]][each[0]][0] * (temperature[each[2] + 1][each[1]][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][0] * dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][0])
		E_Cond_z_neg = Lambda[each[2]][each[1]][each[0]][1] * (temperature[each[2] - 1][each[1]][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][1] * dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][1])
		E_Cond_y_pos = Lambda[each[2]][each[1]][each[0]][2] * (temperature[each[2]][each[1] + 1][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][2] * dt * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][2])
		E_Cond_y_neg = Lambda[each[2]][each[1]][each[0]][3] * (temperature[each[2]][each[1] - 1][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][3] * dt * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][3])
		E_Cond_x_pos = Lambda[each[2]][each[1]][each[0]][4] * (temperature[each[2]][each[1]][each[0] + 1] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][4] * dt * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][4])
		E_Cond_x_neg = Lambda[each[2]][each[1]][each[0]][5] * (temperature[each[2]][each[1]][each[0] - 1] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][5] * dt * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][5])
		E_Lat = - (sublimated_mass[each[2]][each[1]][each[0]] - resublimated_mass[each[2]][each[1]][each[0]]) * latent_heat_water[each[2]][each[1]][each[0]] * dt
		#E_Lat = 0
		E_Energy_Increase = E_In + E_Rad + E_Cond_z_pos + E_Cond_z_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_x_pos + E_Cond_x_neg + E_Lat
		delta_T_0[each[2]][each[1]][each[0]] = E_Energy_Increase / (heat_capacity[each[2]][each[1]][each[0]] * density[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])
		Energy_Increase_in_surface += E_Energy_Increase
		E_In_in_surface += E_In
		E_Rad_in_surface += E_Rad
		E_Lat_in_surface += E_Lat
	return delta_T_0, Energy_Increase_in_surface, E_In_in_surface, E_Rad_in_surface, E_Lat_in_surface


@jit
def test(r_H, albedo, dt, input_energy, dx, dy, surface, surface_reduced):
	for each in surface_reduced:
		print(each[0], each[1], each[2])
		print(input_energy / r_H ** 2 * (1 - albedo) * dt)
		print(dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]])
		print(surface[each[2]][each[1]][each[0]])


@njit(parallel=False)
def energy_input_data(dt, input_temperature, sigma, epsilon, temperature, Lambda, Dr, n_x, n_y, n_z, heat_capacity, density, dx, dy, dz, surface, surface_reduced, delta_T):
	Energy_Increase_in_surface = 0
	E_In_in_surface = 0
	E_Rad_in_surface = 0
	#delta_T_0 = np.zeros(np.shape(delta_T))
	delta_T_0 = np.zeros((n_z, n_y, n_x))
	E_Lat_in_surface = 0
	a = n_x // 2
	a_rad = (n_x - 16) // 2
	b = n_y // 2
	b_rad = (n_y - 16) // 2
	for each in surface_reduced:
		'''#input energy durch input_energy[each[2]][each[1]][each[0]] ersetzen, sobald genaue Abstrahlcharakteristik der Lampe berechnet
		#E_In = input_energy * dt
		E_Rad = - sigma * epsilon * temperature[each[2]][each[1]][each[0]]**4 * dt * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5])) # [J/(m^2)]
		E_Cond_z_pos = Lambda[each[2]][each[1]][each[0]][0] * (temperature[each[2] + 1][each[1]][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][0] * dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][0])
		E_Cond_z_neg = Lambda[each[2]][each[1]][each[0]][1] * (temperature[each[2] - 1][each[1]][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][1] * dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][1])
		E_Cond_y_pos = Lambda[each[2]][each[1]][each[0]][2] * (temperature[each[2]][each[1] + 1][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][2] * dt * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][2])
		E_Cond_y_neg = Lambda[each[2]][each[1]][each[0]][3] * (temperature[each[2]][each[1] - 1][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][3] * dt * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][3])
		E_Cond_x_pos = Lambda[each[2]][each[1]][each[0]][4] * (temperature[each[2]][each[1]][each[0] + 1] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][4] * dt * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][4])
		E_Cond_x_neg = Lambda[each[2]][each[1]][each[0]][5] * (temperature[each[2]][each[1]][each[0] - 1] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][5] * dt * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][5])
		#E_Lat = - (j_leave[0] - j_inward[0]) * latent_heat_water * dt - (j_leave_co2[0] - j_inward_co2[0]) * latent_heat_co2 * dt
		E_Lat = 0
		E_Energy_Increase = 0 + E_Rad + E_Cond_z_pos + E_Cond_z_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_x_pos + E_Cond_x_neg + E_Lat
		delta_T_0[each[2]][each[1]][each[0]] = E_Energy_Increase / (heat_capacity[each[2]][each[1]][each[0]] * density[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])
		if each[2] == 1 and (((each[0] - a) / a_rad) ** 2 + ((each[1] - b) / b_rad) ** 2 <= 1):
		#if each[2] == 1 and each[1] == n_y//2 and each[0] == n_x//2:
			delta_T_0[each[2]][each[1]][each[0]] = input_temperature - temperature[each[2]][each[1]][each[0]]
		Energy_Increase_in_surface += E_Energy_Increase
		E_In_in_surface += 0
		E_Rad_in_surface += E_Rad
		E_Lat_in_surface += E_Lat'''
		delta_T_0[each[2]][each[1]][each[0]] = input_temperature - temperature[each[2]][each[1]][each[0]]
	return delta_T_0, Energy_Increase_in_surface, E_In_in_surface, E_Rad_in_surface, E_Lat_in_surface


@njit(parallel=False)
def sample_holder_data(n_x, n_y, n_z, sample_holder, temperature, temp_sample_holder):
	for i in prange(0, n_z):
		for j in range(0, n_y):
			for k in range(0, n_x):
				if sample_holder[i][j][k] != 0:
					temperature[i][j][k] = temp_sample_holder
	return temperature


@njit
def S_chamber_cal_curve(volt):
	return - 0.11726 + 0.19387*volt + 0.02832*volt**2		# [kW/m^2]


@njit
def twoD_gaussian_polar_int(x):
	sigma = const.var_lamp_profile
	return 1/(sigma**2) * np.exp(-1/2 * x**2/sigma**2) * x


def amplitude_lamp(solar_constant):
	factor = integrate.quad(twoD_gaussian_polar_int, 0, 4.5E-3)[0]
	amplitude = 15 * solar_constant * (4.5E-3)**2 * np.pi / factor
	return amplitude


@njit
def twoD_gaussian(y, x, sigma, amplitude):
	return amplitude * 1/(2*np.pi*sigma**2) * np.exp(-1/2 * (x**2 + y**2)/sigma**2)
	'''gaussian = np.zeros((len(y), len(x)), dtype=np.float64)
	for j in range(0, len(y)):
		for k in range(0, len(x)):
			gaussian[j][k] = amplitude * 1/(2*np.pi*sigma**2) * np.exp(-1/2 * (x[k]**2 + y[j]**2)/sigma**2)
	return gaussian'''


def get_energy_input_lamp(n_x, n_y, n_z, dx, dy, amplitude, sigma, temperature, a, b):
	lamp_power = np.zeros((n_z, n_y, n_x), dtype=np.float64)
	for i in range(0, n_z):
		for j in range(0, n_y):
			for k in range(0, n_x):
				if temperature[i][j][k] != 0:
					'''y = np.linspace(j*dy[i][j][k] - (np.sum([dy[i][val][k] for val in range(0, b)]) + dy[i][b][k]/2), (j+1)*dy[i][j][k] - (np.sum([dy[i][val][k] for val in range(0, b)]) + dy[i][b][k]/2), 20)
					x = np.linspace(k*dx[i][j][k] - (np.sum(dx[i][j][0:a]) + dx[i][j][a]/2), (k+1)*dx[i][j][k] - (np.sum(dx[i][j][0:a]) + dx[i][j][a]/2), 20)
					if i == 1 and j == n_y//2 and k == n_x//2:
						print(y, x)
						print(twoD_gaussian(y, x, sigma, amplitude))'''
					lamp_power[i][j][k] = integrate.dblquad(twoD_gaussian, k*dx[i][j][k] - (np.sum(dx[i][j][0:a]) + dx[i][j][a]/2), (k+1)*dx[i][j][k] - (np.sum(dx[i][j][0:a]) + dx[i][j][a]/2), j*dy[i][j][k] - (np.sum([dy[i][val][k] for val in range(0, b)]) + dy[i][b][k]/2), (j+1)*dy[i][j][k] - (np.sum([dy[i][val][k] for val in range(0, b)]) + dy[i][b][k]/2), args=(const.var_lamp_profile, amplitude))[0]
					#lamp_power[i][j][k] = np.average(twoD_gaussian(y, x, sigma, amplitude))

	return lamp_power

