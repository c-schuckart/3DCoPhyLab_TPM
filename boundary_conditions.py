import numpy as np
from numba import jit, njit, prange
from scipy import integrate
import variables_and_arrays as var
import constants as const
import settings as sett
from read_images import GCD, convolve


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
@njit
def energy_input_periodic(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_reduced, delta_T, n_x, n_y):
	Energy_Increase_in_surface = 0
	E_In_in_surface = 0
	E_Rad_in_surface = 0
	delta_T_0 = np.zeros(np.shape(delta_T))
	E_Lat_in_surface = 0
	E_cond_in_surface = 0
	Energy_conduction_per_Layer = np.zeros((const.n_z, const.n_y, const.n_x, 6), dtype=np.float64)
	for each in surface_reduced:
		#input energy durch input_energy[each[2]][each[1]][each[0]] ersetzen, sobald genaue Abstrahlcharakteristik der Lampe berechnet
		#E_In = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * surface[each[2]][each[1]][each[0]][1]
		E_In = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * dt * surface[each[2]][each[1]][each[0]][1]
		#E_In = input_energy * dt
		E_Rad = - sigma * epsilon * temperature[each[2]][each[1]][each[0]]**4 * dt * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5])) # [J/(m^2)]
		E_Cond_z_pos = Lambda[each[2]][each[1]][each[0]][0] * (temperature[each[2] + 1][each[1]][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][0] * dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][0])
		E_Cond_z_neg = Lambda[each[2]][each[1]][each[0]][1] * (temperature[each[2] - 1][each[1]][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][1] * dt * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][1])
		if each[1] == 1:
			E_Cond_y_neg = Lambda[each[2]][each[1]][each[0]][3] * (temperature[each[2]][n_y - 2][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][3] * dt * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][3])
		else:
			E_Cond_y_neg = Lambda[each[2]][each[1]][each[0]][3] * (temperature[each[2]][each[1] - 1][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][3] * dt * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][3])
		if each[1] == n_y-2:
			E_Cond_y_pos = Lambda[each[2]][each[1]][each[0]][2] * (temperature[each[2]][1][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][2] * dt * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][2])
		else:
			E_Cond_y_pos = Lambda[each[2]][each[1]][each[0]][2] * (temperature[each[2]][each[1] + 1][each[0]] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][2] * dt * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][2])
		if each[0] == 1:
			E_Cond_x_neg = Lambda[each[2]][each[1]][each[0]][5] * (temperature[each[2]][each[1]][n_x - 2] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][5] * dt * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][5])
		else:
			E_Cond_x_neg = Lambda[each[2]][each[1]][each[0]][5] * (temperature[each[2]][each[1]][each[0] - 1] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][5] * dt * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][5])
		if each[0] == n_x-2:
			E_Cond_x_pos = Lambda[each[2]][each[1]][each[0]][4] * (temperature[each[2]][each[1]][1] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][4] * dt * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][4])
		else:
			E_Cond_x_pos = Lambda[each[2]][each[1]][each[0]][4] * (temperature[each[2]][each[1]][each[0] + 1] - temperature[each[2]][each[1]][each[0]]) / Dr[each[2]][each[1]][each[0]][4] * dt * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (1 - surface[each[2]][each[1]][each[0]][4])
		Energy_conduction_per_Layer[each[2]][each[1]][each[0]][0], Energy_conduction_per_Layer[each[2]][each[1]][each[0]][1], Energy_conduction_per_Layer[each[2]][each[1]][each[0]][2], Energy_conduction_per_Layer[each[2]][each[1]][each[0]][3],Energy_conduction_per_Layer[each[2]][each[1]][each[0]][4], Energy_conduction_per_Layer[each[2]][each[1]][each[0]][5] = E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg
		E_Lat = - (sublimated_mass[each[2]][each[1]][each[0]] - resublimated_mass[each[2]][each[1]][each[0]]) * latent_heat_water[each[2]][each[1]][each[0]]
		#E_Lat = 0
		E_Energy_Increase = E_In + E_Rad + E_Cond_z_pos + E_Cond_z_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_x_pos + E_Cond_x_neg + E_Lat
		delta_T_0[each[2]][each[1]][each[0]] = E_Energy_Increase / (heat_capacity[each[2]][each[1]][each[0]] * density[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])
		Energy_Increase_in_surface += E_Energy_Increase
		E_In_in_surface += E_In
		E_Rad_in_surface += E_Rad
		E_Lat_in_surface += E_Lat
		#E_cond_in_surface += (E_Cond_z_pos + E_Cond_z_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_x_pos + E_Cond_x_neg)
		'''if each[2] == 1 and each[1] == 3 and each[0] == 49:
			print('SURFACE IN: ', E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg)'''
		'''if E_Cond_z_pos + E_Cond_z_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_x_pos + E_Cond_x_neg != 0:
			print('Position: ', each[2], each[1], each[0])
			print(E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg)
			print(temperature[each[2]][each[1]][each[0]], temperature[each[2] + 1][each[1]][each[0]], temperature[each[2] - 1][each[1]][each[0]], temperature[each[2]][each[1] + 1][each[0]], temperature[each[2]][each[1] - 1][each[0]], temperature[each[2]][each[1]][each[0] + 1], temperature[each[2]][each[1]][each[0] - 1])'''
	return delta_T_0, Energy_Increase_in_surface, E_In_in_surface, E_Rad_in_surface, E_Lat_in_surface, Energy_conduction_per_Layer


@njit
def energy_input(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_reduced, delta_T):
	Energy_Increase_in_surface = 0
	E_In_in_surface = 0
	E_Rad_in_surface = 0
	delta_T_0 = np.zeros(np.shape(delta_T))
	E_Lat_in_surface = 0
	E_cond_in_surface = 0
	Energy_conduction_per_Layer = np.zeros((const.n_z, const.n_y, const.n_x, 6), dtype=np.float64)
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
		Energy_conduction_per_Layer[each[2]][each[1]][each[0]][0], Energy_conduction_per_Layer[each[2]][each[1]][each[0]][1], Energy_conduction_per_Layer[each[2]][each[1]][each[0]][2], Energy_conduction_per_Layer[each[2]][each[1]][each[0]][3],Energy_conduction_per_Layer[each[2]][each[1]][each[0]][4], Energy_conduction_per_Layer[each[2]][each[1]][each[0]][5] = E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg
		E_Lat = - (sublimated_mass[each[2]][each[1]][each[0]] - resublimated_mass[each[2]][each[1]][each[0]]) * latent_heat_water[each[2]][each[1]][each[0]]
		#E_Lat = 0
		E_Energy_Increase = E_In + E_Rad + E_Cond_z_pos + E_Cond_z_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_x_pos + E_Cond_x_neg + E_Lat
		delta_T_0[each[2]][each[1]][each[0]] = E_Energy_Increase / (heat_capacity[each[2]][each[1]][each[0]] * density[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])
		Energy_Increase_in_surface += E_Energy_Increase
		E_In_in_surface += E_In
		E_Rad_in_surface += E_Rad
		E_Lat_in_surface += E_Lat
		#E_cond_in_surface += (E_Cond_z_pos + E_Cond_z_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_x_pos + E_Cond_x_neg)
		'''if each[2] == 1 and each[1] == 3 and each[0] == 49:
			print('SURFACE IN: ', E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg)'''
		'''if E_Cond_z_pos + E_Cond_z_neg + E_Cond_y_pos + E_Cond_y_neg + E_Cond_x_pos + E_Cond_x_neg != 0:
			print('Position: ', each[2], each[1], each[0])
			print(E_Cond_z_pos, E_Cond_z_neg, E_Cond_y_pos, E_Cond_y_neg, E_Cond_x_pos, E_Cond_x_neg)
			print(temperature[each[2]][each[1]][each[0]], temperature[each[2] + 1][each[1]][each[0]], temperature[each[2] - 1][each[1]][each[0]], temperature[each[2]][each[1] + 1][each[0]], temperature[each[2]][each[1] - 1][each[0]], temperature[each[2]][each[1]][each[0] + 1], temperature[each[2]][each[1]][each[0] - 1])'''
	return delta_T_0, Energy_Increase_in_surface, E_In_in_surface, E_Rad_in_surface, E_Lat_in_surface, Energy_conduction_per_Layer

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
		#Wouldn't work for a sample with dissolving surface voxels, once an exposed voxel would be covered by another one on top - Not a problem for sand of course
		delta_T_0[each[2]][each[1]][each[0]] = input_temperature[each[1]][each[0]] - temperature[each[2]][each[1]][each[0]]
		Energy_Increase_in_surface += delta_T_0[each[2]][each[1]][each[0]] * (heat_capacity[each[2]][each[1]][each[0]] * density[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])
		E_In_in_surface += delta_T_0[each[2]][each[1]][each[0]] * (heat_capacity[each[2]][each[1]][each[0]] * density[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]])
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
def sample_holder_test(n_x, n_y, n_z, sample_holder, temperature):
	for i in prange(0, n_z):
		for j in range(0, n_y):
			for k in range(0, n_x):
				temps = np.zeros(6, dtype=np.float64)
				if sample_holder[i][j][k] != 0:
					if sample_holder[i+1][j][k] == 0 and temperature[i+1][j][k] != 0:
						temps[0] = temperature[i+1][j][k]
					if sample_holder[i-1][j][k] == 0 and temperature[i-1][j][k] != 0:
						temps[1] = temperature[i-1][j][k]
					if sample_holder[i][j+1][k] == 0 and temperature[i][j+1][k] != 0:
						temps[2] = temperature[i][j+1][k]
					if sample_holder[i][j-1][k] == 0 and temperature[i][j-1][k] != 0:
						temps[3] = temperature[i][j-1][k]
					if sample_holder[i][j][k+1] == 0 and temperature[i][j][k+1] != 0:
						temps[4] = temperature[i][j][k+1]
					if sample_holder[i][j][k-1] == 0 and temperature[i][j][k-1] != 0:
						temps[5] = temperature[i][j][k-1]
					temperature[i][j][k] = np.max(temps)
	return temperature


@njit
def calculate_deeper_layer_source(n_x, n_y, n_z, input_energy, r_H, albedo, surface, dx, dy, dz):
	S_c = np.zeros((n_z, n_y, n_x), dtype=np.float64)
	Q = input_energy / r_H ** 2 * (1 - albedo)
	S_c[2:const.n_z] = Q[2:const.n_z] / (dx[2:const.n_z] * dy[2:const.n_z] * dz[2:const.n_z])
	return S_c


@njit
def day_night_cycle(lamp_power, S_c, period, current_time):
	time_factor = np.sin(2 * current_time / period * np.pi)
	if time_factor >= 0:
		lamp_power = lamp_power * time_factor
		S_c = S_c * time_factor
	else:
		lamp_power = lamp_power * 0
		S_c = S_c * 0
	return lamp_power, S_c

@njit
def sample_holder_test_2(n_x, n_y, n_z, sample_holder, temperature, target_temp, target_height):
	for i in prange(target_height, target_height+1):
		for j in range(0, n_y):
			for k in range(0, n_x):
				if sample_holder[i][j][k] != 0:
					temperature[i][j][k] = target_temp
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


def get_L_chamber_lamp_power(sample_holder):
	X=np.array([[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-3,-2,-1,0,1,2,3,4,5]])
	Y=np.array([[-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5],[-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4],[-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3],[-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],[0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3,3,3,3],[4,4,4,4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5,5,5,5]])
	Z=np.array([[0,0,5,10,10,20,10,5,5,0,0],[5,20,100,600,900,1200,900,600,100,20,5],[10,40,400,1900,2350,2500,2350,1900,400,40,10],[10,60,650,2200,2600,2800,2600,2200,650,60,10],[20,90,950,2400,3200,3300,3200,2400,950,90,20],[30,150,1200,2600,3400,3600,3400,2600,1200,150,30],[20,100,1000,2500,3300,3400,3300,2500,1000,100,20],[10,70,700,2400,2835,3000,2835,2400,720,55,10],[10,50,500,650,800,900,800,650,500,50,10],[5,40,50,70,100,150,100,70,50,40,5],[5,5,10,10,15,30,15,10,10,5,5]], dtype=np.float64)
	if sample_holder == 'M':
		new_Z = np.zeros((15, 15), dtype=np.float64)
		for i in range(1, 16):
			for j in range(1, 16):
				if i % 2 == 0 and j % 2 == 0:
					new_Z[i-1][j-1] = Z[i//2 + 1][j//2 + 1]
				if i % 2 == 1 and j % 2 == 0:
					new_Z[i-1][j-1] = (Z[i//2 + 1][j//2 + 1] + Z[(i+1)//2 + 1][j//2 + 1])/2
				if i % 2 == 0 and j % 2 == 1:
					new_Z[i-1][j-1] = (Z[i//2 + 1][j//2 + 1] + Z[i//2 + 1][(j+1)//2 + 1])/2
				if i % 2 == 1 and j % 2 == 1:
					new_Z[i-1][j-1] = (Z[i // 2 + 1][j // 2 + 1] + Z[i // 2 + 1][(j + 1) // 2 + 1] + Z[(i+1) // 2 + 1][j // 2 + 1] + Z[(i+1) // 2 + 1][(j + 1) // 2 + 1]) / 4
		Z = new_Z
	if sample_holder == 'L':
		new_Z = np.zeros((27, 27), dtype=np.float64)
		new_Z[8:19, 8:19] = Z
		Z = new_Z
	Lamp_power_per_m2 = Z * 1/0.48
	return Lamp_power_per_m2

def calculate_L_chamber_lamp_bd(Volt, sample_holder, n_x, n_y, n_z, min_dx, min_dy, min_dz, depth_absorption, absorption_scale_length):
	Surface_powers = get_L_chamber_lamp_power(sample_holder) * (min_dx * min_dy) * S_chamber_cal_curve(Volt)/S_chamber_cal_curve(24)
	ggT = GCD(len(Surface_powers[0]), const.n_x)
	length = len(Surface_powers[0])//ggT
	convolved = convolve(Surface_powers, length, const.n_x, len(Surface_powers[0]), n_x, n_y)[0]
	for i in range(2, n_y-2):
		convolved[i] = convolved[i+1]
	lamp_energy = np.zeros((n_z, n_y, n_x), dtype=np.float64)
	lamp_energy[:] = convolved
	if depth_absorption:
		for i in range(0, n_z):
			lamp_energy[i] = - lamp_energy[i] * (np.exp(- (i+1)*min_dz/absorption_scale_length) - np.exp(- i*min_dz/absorption_scale_length))
			if i * min_dz > 5*absorption_scale_length:
				lamp_energy[i] = lamp_energy[i] * 0
	return lamp_energy


