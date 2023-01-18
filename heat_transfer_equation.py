import numpy as np
from numba import jit
import variables_and_arrays as var
import constants as const

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
@jit
def hte_calculate(n_x, n_y, n_z, surface, delta_T_0, temperature, Lambda, Dr, dx, dy, dz, dt, density, heat_capacity, j_leave, j_inward, latent_heat_water, j_leave_co2, j_inward_co2, latent_heat_co2):
    delta_T = np.zeros((n_x, n_y, n_z), dtype=np.float64) + delta_T_0
    Energy_Increase_per_Layer = np.zeros((n_x, n_y, n_z), dtype=np.float64)
    Latent_Heat_per_Layer = np.zeros((n_x, n_y, n_z), dtype=np.float64)
    #Fourier_number = np.zeros(n_x, n_y, n_z)
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if np.sum(surface[i][j][k]) == 0 and temperature[i][j][k] > 0:
                    # Standard Thermal Diffusivity Equation 1D explicit
                    delta_T[i][j][k] = ((((temperature[i][j][k - 1] - temperature[i][j][k]) * Lambda[i][j][k][0] / (Dr[i][j][k][0])) - ((temperature[i][j][k] - temperature[i][j][k + 1]) * Lambda[i][j][k][1] / (Dr[i][j][k][1]))) / dx[i][j][k]) * dt / (
                                             density[i][j][k] * heat_capacity[i][j][k]) + \
                                       ((((temperature[i][j + 1][k] - temperature[i][j][k]) * Lambda[i][j][k][2] / (Dr[i][j][k][2]))
                                         - ((temperature[i][j][k] - temperature[i][j - 1][k]) * Lambda[i][j][k][3] / (
                                               Dr[i][j][k][3]))) / dy[i][j][k]) * dt / (
                                               density[i][j][k] * heat_capacity[i][j][k]) + \
                                       ((((temperature[i + 1][j][k] - temperature[i][j][k]) * Lambda[i][j][k][4] / (Dr[i][j][k][4]))
                                         - ((temperature[i][j][k] - temperature[i - 1][j][k]) * Lambda[i][j][k][5] / (
                                               Dr[i][j][k][5]))) / dz[i][j][k]) * dt / (
                                               density[i][j][k] * heat_capacity[i][j][k])
                    #- (j_leave[i] - j_inward[i]) * latent_heat_water * dt / (density[i] * heat_capacity[i] * dx[i]) - (j_leave_co2[i] - j_inward_co2[i]) * latent_heat_co2 * dt / (density[i] * heat_capacity[i] * dx[i])  # [K]
                    #Fourier_number[i] = Lambda[i] / (density[i] * heat_capacity[i]) * dt / dx[i] ** 2  # [-]
                    #Latent_Heat_per_Layer[i] = - (j_leave[i] - j_inward[i]) * latent_heat_water * dt - (j_leave_co2[i] - j_inward_co2[i]) * latent_heat_co2 * dt
                    Energy_Increase_per_Layer[i][j][k] = heat_capacity[i][j][k] * density[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] * delta_T[i][j][k]  # [J]
    return delta_T, Energy_Increase_per_Layer, Latent_Heat_per_Layer

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
@jit
def update_thermal_arrays(n_x, n_y, n_z, temperature, water_content_per_layer, co2_content_per_layer,  delta_T, Energy_Increase_per_Layer, j_leave, j_inward, j_leave_co2, j_inward_co2, dt, avogadro_constant, molar_mass_water, molar_mass_co2, heat_capacity, heat_capacity_water_ice, heat_capacity_co2_ice, EIpL_0, Latent_Heat_per_Layer, E_Lat_0, E_Rad, E_In):
    temperature_o = temperature + delta_T
    outgassed_molecules_per_time_step = 0
    outgassed_molecules_per_time_step_co2 = 0
    Energy_Increase_per_Layer[0] = EIpL_0
    Latent_Heat_per_Layer[0] = E_Lat_0
    Energy_Increase_Total_per_time_Step = 0
    Latent_Heat_per_time_step = 0
    dust_ice_ratio_per_layer = 0
    co2_h2o_ratio_per_layer = 0
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
    Energy_Increase_Total_per_time_Step = np.sum(Energy_Increase_per_Layer)
    E_conservation = Energy_Increase_Total_per_time_Step - E_Rad - Latent_Heat_per_time_step - E_In
    # Set Energy Loss per Timestep = 0 -> Differential Counting of Energy Loss
    return temperature_o, water_content_per_layer, co2_content_per_layer, outgassed_molecules_per_time_step, outgassed_molecules_per_time_step_co2, heat_capacity, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, E_conservation, Energy_Increase_Total_per_time_Step, E_Rad, Latent_Heat_per_time_step, E_In