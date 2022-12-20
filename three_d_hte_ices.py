import numpy as np
import constants as const
import variables_and_arrays as var
import settings as sett
from tqdm import tqdm
import json
from surface_detection import create_equidistant_mesh, DEBUG_print_3D_arrays, find_surface
from thermal_parameter_functions import lambda_constant
from boundary_conditions import energy_input, test
from heat_transfer_equation import hte_calculate, update_thermal_arrays

#work arrays
temperature, dx, dy, dz, Dr = create_equidistant_mesh(const.n_x, const.n_y, const.n_z, const.temperature_ini, const.min_dx, const.min_dy, const.min_dz)
heat_capacity = var.heat_capacity
density = var.density
delta_T = var.delta_T
print(np.shape(temperature))
#DEBUG_print_3D_arrays(const.n_x, const.n_y, const.n_z, temperature)
surface, surface_reduced = find_surface(const.n_x, const.n_y, const.n_z, 0, 0, 0, temperature, var.surface)
print(surface[1][1][25])
j_leave = np.zeros(const.n + 1, dtype=np.float64)
j_inward = np.zeros(const.n + 1, dtype=np.float64)
j_leave_co2 = np.zeros(const.n + 1, dtype=np.float64)
j_inward_co2 = np.zeros(const.n + 1, dtype=np.float64)
water_content_per_layer = var.water_content_per_layer
co2_content_per_layer = var.co2_content_per_layer
E_conservation = var.E_conservation

test(const.r_H, const.albedo, const.dt, const.Input_Intensity, dx, dy, surface, surface_reduced)



#print(temperature)
#temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, time_passed = load_from_save()
'''
Main Loop of the model. Comment out/Uncomment function calls to disable/enable features
'''
for j in tqdm(range(0, const.k)):
    #Lambda = lambda_ice_particles(const.n, temperature, var.DX, var.dx, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.surface_energy_par, const.r_mono, const.f_1, const.f_2, var.VFF_pack, const.sigma, const.e_1, j * const.dt, const.temperature_ini, const.lambda_water_ice_change)
    Lambda = lambda_constant(const.n_x, const.n_y, const.n_z, const.lambda_constant)
    #j_leave, j_inward, j_leave_co2, j_inward_co2, var.deeper_diffusion, var.deeper_diffusion_co2 = calculate_molecule_flux(temperature, j_leave, j_leave_co2, const.a_H2O, const.b_H2O, const.m_H2O, const.k_boltzmann, const.b, water_content_per_layer, const.avogadro_constant, const.molar_mass_water, const.dt, var.dx, const.n, co2_content_per_layer, const.a_CO2, const.b_CO2, const.m_CO2, const.molar_mass_co2, var.diffusion_factors, var.deeper_diffusion, var.deeper_diffusion_co2)
    dT_0, EIis_0, E_In, E_Rad, E_Lat_0 = energy_input(const.r_H, const.albedo, const.dt, const.Input_Intensity, const.sigma, const.epsilon, temperature, Lambda, Dr, j_leave, j_inward, const.latent_heat_water, j_leave_co2, j_inward_co2, const.latent_heat_co2, heat_capacity, density, dx, dy, dz, surface, surface_reduced, delta_T)
    delta_T, Energy_Increase_per_Layer = hte_calculate(const.n_x, const.n_y, const.n_z, surface, dT_0, temperature, Lambda, Dr, dx, dy, dz, const.dt, density, heat_capacity, j_leave, j_inward, const.latent_heat_water, j_leave_co2, j_inward_co2, const.latent_heat_co2, Fourier_number, Latent_Heat_per_Layer, surface_temperature)
    temperature, water_content_per_layer, co2_content_per_layer, outgassed_molecules_per_time_step, outgassed_molecules_per_time_step_co2, heat_capacity, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, E_conservation[j], Energy_Increase_Total_per_time_Step_arr[j], E_Rad_arr[j], Latent_Heat_per_time_step_arr[j], E_In_arr[j] = update_thermal_arrays(const.n_x, const.n_y, const.n_z, temperature, water_content_per_layer, co2_content_per_layer,  delta_T, Energy_Increase_per_Layer, j_inward, j_leave, j_leave_co2, j_inward_co2, const.dt, const.avogadro_constant, const.molar_mass_water, const.molar_mass_co2, heat_capacity, const.heat_capacity_water_ice, const.heat_capacity_co2_ice, EIis_0, E_Lat_0, E_Rad, E_In)
    #pressure, pressure_co2, highest_pressure, highest_pressure_co2 = pressure_calculation(j_leave, j_leave_co2, pressure, pressure_co2, temperature, var.dx, const.a_H2O, const.b_H2O, const.a_CO2, const.b_CO2, const.b, highest_pressure, highest_pressure_co2)
    #temperature, pressure, pressure_co2, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, total_ejection_events, ejection_times, drained_layers = check_drained_layers(const.n, temperature, pressure, pressure_co2, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, total_ejection_events, var.layer_strength, const.second_temp_layer, var.base_water_particle_number, var.base_co2_particle_number, const.dust_ice_ratio_global, const.co2_h2o_ratio_global, heat_capacity, const.heat_capacity_dust, const.heat_capacity_water_ice, const.heat_capacity_co2_ice, ejection_times, j, drained_layers)
    #temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save = data_store(j, temperature, water_content_per_layer, co2_content_per_layer, outgassed_molecules_per_time_step/const.dt, outgassed_molecules_per_time_step_co2/const.dt, temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save, sett.data_reduction)


#Data saving and output
#save_current_arrays(temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, ejection_times, var.time_passed + const.dt * const.k)
#data_save(temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save, E_conservation, Energy_Increase_Total_per_time_Step_arr, E_Rad_arr, Latent_Heat_per_time_step_arr, E_In_arr, 'base_case')




