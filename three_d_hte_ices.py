import numpy as np
import constants as const
import variables_and_arrays as var
import settings as sett
from tqdm import tqdm
import json
from thermal_parameter_functions import lambda_ice_particles, lambda_pebble, lambda_constant
from orbit_calculation_and_input_energy import energy_input, orbit_calculation, day_night_regular
from initialisation import dx_initialisation
from molecule_transfer import calculate_molecule_flux
from heat_transfer_equation import hte_calculate, update_thermal_arrays
from pressure_and_ejections import pressure_calculation, ejections, check_drained_layers
from save_and_load import data_store, data_save, save_current_arrays, load_from_save

dx_initialisation(sett.dx_switch)

#work arrays
Lambda = np.full(const.n, const.lambda_constant, dtype=np.float64)
heat_capacity = var.heat_capacity
temperature = np.full(const.n + 1, const.temperature_ini, dtype=np.float64)
for i in range(const.second_temp_layer_depth, const.n + 1):
    temperature[i] = const.second_temp_layer
j_leave = np.zeros(const.n + 1, dtype=np.float64)
j_inward = np.zeros(const.n + 1, dtype=np.float64)
j_leave_co2 = np.zeros(const.n + 1, dtype=np.float64)
j_inward_co2 = np.zeros(const.n + 1, dtype=np.float64)
delta_T = np.zeros(const.n + 1, dtype=np.float64)
pressure = np.zeros(const.n + 1, dtype=np.float64)
pressure_co2 = np.zeros(const.n + 1, dtype=np.float64)
highest_pressure = np.zeros(const.n + 1, dtype=np.float64)
highest_pressure_co2 = np.zeros(const.n + 1, dtype=np.float64)
Fourier_number = var.Fourier_number
surface_temperature = var.surface_temperature
water_content_per_layer = var.water_content_per_layer
co2_content_per_layer = var.co2_content_per_layer
outgassed_molecules_per_time_step = var.outgassed_molecules_per_time_step
outgassed_molecules_per_time_step_co2 = var.outgassed_molecules_per_timestep_co2
Energy_con_div = 0
Energy_total = 0
dT_0 = 0
day_counter = const.day_counter
day_position = var.day_position
day_store = var.day_store
day_start = var.day_start
r_H_tab = var.r_H_tab
r_H = const.r_H
dust_ice_ratio_per_layer = var.dust_ice_ratio_per_layer
co2_h2o_ratio_per_layer = var.co2_h2o_ratio_per_layer
total_ejection_events = var.total_ejection_events
axial_tilt_factor = var.axial_tilt_factor
#Arrays for data gathering
#temperature_save = np.zeros((const.k//434, const.n+1))
temperature_save = np.zeros((const.k//sett.data_reduction, const.n+1))
water_content_save = np.zeros((const.k//sett.data_reduction, const.n+1))
co2_content_save = np.zeros((const.k//sett.data_reduction, const.n+1))
'''temperature_save = np.zeros((const.k//sett.data_reduction, 3))
water_content_save = np.zeros((const.k//sett.data_reduction, 3))
co2_content_save = np.zeros((const.k//sett.data_reduction, 3))'''
outgassing_save = np.zeros(const.k)
outgassing_co2_save = np.zeros(const.k)
lambda_save = np.zeros((const.k//sett.data_reduction, const.n))
energy_diff_save = np.zeros((const.k//sett.data_reduction, const.n))
E_In_save = np.zeros(const.k)
r_H_save = np.zeros(const.k)
day_position_save = np.zeros(const.k)
ejection_times = np.zeros(const.k)
E_conservation = var.E_conservation
time_passed = var.time_passed
drained_layers = 0
Energy_Increase_Total_per_time_Step_arr, E_Rad_arr, Latent_Heat_per_time_step_arr, E_In_arr = np.zeros(const.k), np.zeros(const.k), np.zeros(const.k), np.zeros(const.k)

#print(temperature)
#temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, time_passed = load_from_save()
'''
Main Loop of the model. Comment out/Uncomment function calls to disable/enable features
'''
for j in tqdm(range(0, const.k)):
    #if j % 8500 == 0:
        #temperature_save[j // 434] = temperature
        #lambda_save[j // 434] = Lambda
        #energy_diff_save[j // 434] = Energy_con_div
        #print(temperature)
        #print(water_content_per_layer)
        #print(r_H, total_ejection_events)
    #r_H, r_H_tab = orbit_calculation(j, const.dt, time_passed,day_counter, const.orbit_period, const.perihelion, const.eccentricity, const.semi_major_axis, r_H_tab)
    #var.day_position_store, day_position, var.declination_store, var.day_length_store, day_counter, day_store, day_start, axial_tilt_factor = day_night_regular(j, const.orbit_period, const.equinox, const.axial_tilt, const.latitude, var.axial_tilt_store, const.Period, day_store, day_start, day_counter, day_position, const.dt, r_H, var.Max_Fourier_number, var.declination_store, var.day_length_store, var.day_position_store)
    #day_position_save[j] = day_position
    Lambda = lambda_ice_particles(const.n, temperature, var.DX, var.dx, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.surface_energy_par, const.r_mono, const.f_1, const.f_2, var.VFF_pack, const.sigma, const.e_1, j * const.dt, const.temperature_ini, const.lambda_water_ice_change)
    #Lambda = lambda_constant(const.n, const.lambda_constant)
    j_leave, j_inward, j_leave_co2, j_inward_co2, var.deeper_diffusion, var.deeper_diffusion_co2 = calculate_molecule_flux(temperature, j_leave, j_leave_co2, const.a_H2O, const.b_H2O, const.m_H2O, const.k_boltzmann, const.b, water_content_per_layer, const.avogadro_constant, const.molar_mass_water, const.dt, var.dx, const.n, co2_content_per_layer, const.a_CO2, const.b_CO2, const.m_CO2, const.molar_mass_co2, var.diffusion_factors, var.deeper_diffusion, var.deeper_diffusion_co2)
    dT_0, EIpL_0, E_In, E_Rad, E_Lat_0 = energy_input(const.solar_constant, r_H, const.tilt, const.albedo, const.dt, axial_tilt_factor, day_position, const.input_energy, const.sigma, const.epsilon, temperature, Lambda, var.DX,  j_leave, j_inward, const.latent_heat_water, j_leave_co2, j_inward_co2, const.latent_heat_co2, heat_capacity[0], const.surface_area, var.density, var.dx, j * const.dt)
    delta_T, Fourier_number, Energy_Increase_per_Layer, surface_temperature, Latent_Heat_per_Layer = hte_calculate(j, const.n, dT_0, temperature, Lambda, var.DX, var.dx, const.dt, var.density, heat_capacity, j_leave, j_inward, const.latent_heat_water, j_leave_co2, j_inward_co2, const.latent_heat_co2, var.Fourier_number, var.Energy_Increase_per_Layer, var.Latent_Heat_per_Layer, const.surface_area, var.surface_temperature)
    temperature, water_content_per_layer, co2_content_per_layer, outgassed_molecules_per_time_step, outgassed_molecules_per_time_step_co2, heat_capacity, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, E_conservation[j], Energy_Increase_Total_per_time_Step_arr[j], E_Rad_arr[j], Latent_Heat_per_time_step_arr[j], E_In_arr[j] = update_thermal_arrays(const.n, temperature, water_content_per_layer, co2_content_per_layer, outgassed_molecules_per_time_step, outgassed_molecules_per_time_step_co2, delta_T, Energy_Increase_per_Layer, const.surface_area, j_inward, j_leave, j_leave_co2, j_inward_co2, const.dt, const.avogadro_constant, const.molar_mass_water, const.molar_mass_co2, heat_capacity, const.heat_capacity_dust, const.heat_capacity_water_ice, const.heat_capacity_co2_ice, var.dust_mass_in_dust_ice_layers, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, EIpL_0, Latent_Heat_per_Layer, E_Lat_0, E_Rad, E_In)
    pressure, pressure_co2, highest_pressure, highest_pressure_co2 = pressure_calculation(j_leave, j_leave_co2, pressure, pressure_co2, temperature, var.dx, const.a_H2O, const.b_H2O, const.a_CO2, const.b_CO2, const.b, highest_pressure, highest_pressure_co2)
    temperature, pressure, pressure_co2, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, total_ejection_events, ejection_times, drained_layers = check_drained_layers(const.n, temperature, pressure, pressure_co2, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, total_ejection_events, var.layer_strength, const.second_temp_layer, var.base_water_particle_number, var.base_co2_particle_number, const.dust_ice_ratio_global, const.co2_h2o_ratio_global, heat_capacity, const.heat_capacity_dust, const.heat_capacity_water_ice, const.heat_capacity_co2_ice, ejection_times, j, drained_layers)
    temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save = data_store(j, temperature, water_content_per_layer, co2_content_per_layer, outgassed_molecules_per_time_step/const.dt, outgassed_molecules_per_time_step_co2/const.dt, temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save, sett.data_reduction)
    if j * const.dt <= (3600 * 10):
        temperature[0] = const.temperature_ini
'''    if total_ejection_events > 1:
        break
    if np.abs(E_conservation[j]) > 2e-13:
        print(temperature)
        print(total_ejection_events)
        break'''


#Data saving and output
save_current_arrays(temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, ejection_times, var.time_passed + const.dt * const.k)
data_save(temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save, E_conservation, Energy_Increase_Total_per_time_Step_arr, E_Rad_arr, Latent_Heat_per_time_step_arr, E_In_arr, 'base_case')
print(temperature)
print(water_content_per_layer)
print(total_ejection_events)
'''#ts = [temperature_save[i].tolist() for i in range (const.k//434)]
ts = [temperature_save[i].tolist() for i in range (const.k)]
ls = [lambda_save[i].tolist() for i in range (const.k//434)]
ei = [E_In_save[i].tolist() for i in range (const.k)]
rh = [r_H_save[i].tolist() for i in range (const.k)]
dp = [day_position_save[i].tolist() for i in range (const.k)]
dict = {'Temperature': ts, 'Heat Conductivity': ls, 'Energy': ei, 'r_H': rh, 'day_position': dp}
with open('test_comet_model_3.json', 'w') as outfile:
	json.dump(dict, outfile)'''


