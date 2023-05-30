import numpy as np
import constants as const
import variables_and_arrays as var
import settings as sett
from tqdm import tqdm
import json
from surface_detection import create_equidistant_mesh, DEBUG_print_3D_arrays, find_surface, surrounding_checker, update_surface_arrays
from thermal_parameter_functions import lambda_constant, lambda_granular, calculate_heat_capacity, lambda_sand, calculate_latent_heat, calculate_density, thermal_functions, lambda_ice_block
from boundary_conditions import energy_input, test, energy_input_data, sample_holder_data, amplitude_lamp, get_energy_input_lamp, calculate_L_chamber_lamp_bd, sample_holder_test, sample_holder_test_2
from heat_transfer_equation import hte_calculate, update_thermal_arrays, test_E_cond, test_e_cond_2
from molecule_transfer import calculate_molecule_flux, calculate_molecule_surface
from data_input import getPath, read_temperature_data, transform_temperature_data
from save_and_load import data_store, data_store_sensors, data_save_sensors, data_save


#work arrays and mesh creation + surface detection
temperature, dx, dy, dz, Dr, a, a_rad, b, b_rad = create_equidistant_mesh(const.n_x, const.n_y, const.n_z, const.temperature_ini, const.min_dx, const.min_dy, const.min_dz)
#temperature, dx, dy, dz, Dr, Lambda = one_d_test(const.n_x, const.n_y, const.n_z, const.min_dx, const.min_dy, const.min_dz, 'y')
heat_capacity = var.heat_capacity
density = var.density * const.VFF_pack_const
delta_T = var.delta_T
print(np.shape(temperature))
#DEBUG_print_3D_arrays(const.n_x, const.n_y, const.n_z, temperature)
surface, surface_reduced, sample_holder = find_surface(const.n_x, const.n_y, const.n_z, 0, 0, 0, const.n_x, const.n_y, const.n_z, temperature, var.surface, a, a_rad, b, b_rad, True)
surrounding_surface = surrounding_checker(surface_reduced, surface, var.n_x_lr, var.n_y_lr, var.n_z_lr, temperature)
uniform_water_masses = density * dx * dy * dz
for i in range(0, const.n_z):
    for j in range(0, const.n_y):
        for k in range(0, const.n_x):
            if temperature[i][j][k] == 0 or sample_holder[i][j][k] == 1:
                uniform_water_masses[i][j][k] = 0
print(np.sum(uniform_water_masses[1]))
#surface = var.surface
#surface_reduced = np.array([])
#print(surface[1][1][25])
sublimated_mass = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
resublimated_mass = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
outgassing_rate = var.outgassing_rate
pressure = var.pressure
water_content_per_layer = var.water_content_per_layer
co2_content_per_layer = var.co2_content_per_layer
E_conservation = var.E_conservation
Energy_Increase_Total_per_time_Step_arr = var.Energy_Increase_Total_per_time_Step_arr
E_Rad_arr = var.E_Rad_arr
Latent_Heat_per_time_step_arr = var.Latent_Heat_per_time_step_arr
E_In_arr = var.E_In_arr

#test(const.r_H, const.albedo, const.dt, const.Input_Intensity, dx, dy, surface, surface_reduced)



#print(temperature)
#temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, time_passed = load_from_save()
#2023-02-15 17:45:01 2023-02-15 17:45:02
#2023-02-15 20:45:41
#time_deltas_data_surface, surface_temp = read_temperature_data(getPath(), '2023-02-15 16:45:00', '2023-02-15 17:45:02', [1], [])
#max_k, surface_temp = transform_temperature_data(const.k, const.dt, np.array(time_deltas_data_surface), [], surface_temp)

#lamp_power = calculate_L_chamber_lamp_bd(15.33, 'M', const.n_x, const.n_y, const.n_z)
lamp_power = np.full((const.n_z, const.n_y, const.n_x), const.solar_constant*const.min_dx*const.min_dy, dtype=np.float64)
max_k = const.k
time_deltas_data_interior, sample_holder_temp = read_temperature_data('D:/Laboratory_data/temps_ice.txt', '2023-02-22 11:27:02', '2023-02-22 13:00:56', [6], [])
max_k_2, sample_holder_temp = transform_temperature_data(const.k, const.dt, np.array(time_deltas_data_interior), [], sample_holder_temp)

temperature_save = np.zeros((min(const.k, max_k, max_k_2)//sett.data_reduce + 1, const.n_z, const.n_y, const.n_x))
sensor_10mm = np.zeros(min(const.k, max_k, max_k_2), dtype=np.float64)
sensor_20mm = np.zeros(min(const.k, max_k, max_k_2), dtype=np.float64)
sensor_35mm = np.zeros(min(const.k, max_k, max_k_2), dtype=np.float64)
sensor_55mm = np.zeros(min(const.k, max_k, max_k_2), dtype=np.float64)
sensor_90mm = np.zeros(min(const.k, max_k, max_k_2), dtype=np.float64)
#max_k, max_k_2 = const.k, const.k
#temperature_save = np.zeros((min(const.k, max_k, max_k_2)//sett.data_reduce + 1, const.n_z, const.n_y, const.n_x))
water_content_save = np.zeros((min(const.k, max_k, max_k_2)//sett.data_reduce + 1, const.n_z, const.n_y, const.n_x))
sublimated_mass_save = np.zeros((min(const.k, max_k, max_k_2)//sett.data_reduce + 1, const.n_z, const.n_y, const.n_x))
Max_Fourier_number = np.zeros(min(const.k, max_k, max_k_2), dtype=np.float64)
surface_temp = np.full(const.k, 201, dtype=np.float64)
r_n = np.zeros((const.n_x, const.n_y, const.n_x), dtype=np.float64)
#sample_holder_temp = np.full(const.k, 140, dtype=np.float64)

'''with open('lamp_input_S_chamber.json') as json_file:
    data_e_in = json.load(json_file)
lamp_power = np.array(data_e_in['Lamp Power'])
json_file.close()'''

#np.savetxt("D:/Masterarbeit_data/surface_temp.csv", surface_temp, delimiter=",")
#np.savetxt("D:/Masterarbeit_data/sample_holder_temp.csv", sample_holder_temp, delimiter=",")
latent_heat_water = np.full((const.n_z, const.n_y, const.n_x), const.latent_heat_water, dtype=np.float64)
'''
Main Loop of the model. Comment out/Uncomment function calls to disable/enable features
'''
'''sample_holder = np.zeros((const.n_z, const.n_y, const.n_x))
sample_holder[const.n_z-1][0][0] = 1
surface = np.zeros((const.n_z, const.n_y, const.n_x, 6))'''
temperature = sample_holder_data(const.n_x, const.n_y, const.n_z, sample_holder, temperature, 110)
#Lambda = lambda_sand(const.n_x, const.n_y, const.n_z, temperature, Dr, const.lambda_sand, sample_holder, const.lambda_sample_holder)
#Lambda = lambda_granular(const.n_x, const.n_y, const.n_z, temperature, Dr, dx, dy, dz, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.surface_energy_par, const.r_mono, const.f_1, const.f_2, var.VFF_pack, const.sigma, const.e_1, sample_holder, const.lambda_sample_holder)
#print(Lambda[15][0][0])
Delta_cond_ges = 0

#for j in tqdm(range(0, min(const.k, max_k, max_k_2))):
for j in tqdm(range(0, 1000)):
    Lambda = lambda_granular(const.n_x, const.n_y, const.n_z, temperature, Dr, dx, dy, dz, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.surface_energy_par, const.r_mono, const.f_1, const.f_2, var.VFF_pack, const.sigma, const.e_1, sample_holder, const.lambda_sample_holder, r_n)[0] * const.lambda_scaling_factor
    #Lambda = lambda_ice_block(const.n_x, const.n_y, const.n_z, temperature, Dr, dx, dy, dz, const.lambda_water_ice, const.r_mono, var.VFF_pack, const.sigma, const.e_1, sample_holder, const.lambda_sample_holder) * const.lambda_scaling_factor
    #Lambda = lambda_sand(const.n_x, const.n_y, const.n_z, temperature, Dr, const.lambda_sand, sample_holder, const.lambda_sample_holder)
    #Lambda = lambda_constant(const.n_x, const.n_y, const.n_z, const.lambda_constant)
    '''density = calculate_density(temperature, const.VFF_pack_const)[1]
    heat_capacity = calculate_heat_capacity(temperature)
    latent_heat_water = calculate_latent_heat(temperature, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.R, const.m_mol)'''
    heat_capacity, latent_heat_water, density_grain, density = thermal_functions(temperature, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.R, const.m_mol, const.VFF_pack_const)
    #sublimated_mass, resublimated_mass, pressure, outgassing_rate[j], empty_voxels = calculate_molecule_flux(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_mol, const.R, var.VFF_pack, const.r_mono, const.Phi, const.tortuosity, dx, dy, dz, const.dt, surface_reduced, const.avogadro_constant, const.k_boltzmann, sample_holder, uniform_water_masses, var.n_x_lr, var.n_y_lr, var.n_z_lr, Dr)
    sublimated_mass, resublimated_mass, pressure, outgassing_rate[j], empty_voxels = calculate_molecule_surface(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_H2O, const.R, var.VFF_pack, const.r_mono, const.Phi_Asaeda, const.tortuosity, dx, dy, dz, const.dt, surface_reduced, const.avogadro_constant, const.k_boltzmann, sample_holder, uniform_water_masses, var.n_x_lr, var.n_y_lr, var.n_z_lr, Dr, const.surface_reduction_factor)
    #temperature = sample_holder_test_2(const.n_x, const.n_y, const.n_z, sample_holder, temperature, 110, 50)
    dT_0, EIis_0, E_In, E_Rad, E_Lat_0, Econd_0, Energy_conduction_per_Layer = energy_input(const.r_H, const.albedo, const.dt, lamp_power, const.sigma, const.epsilon, temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_reduced, delta_T)
    #dT_0, EIis_0, E_In, E_Rad, E_Lat_0 = energy_input_data(const.dt, surface_temp[j], const.sigma, const.epsilon, temperature, Lambda, Dr, const.n_x, const.n_y, const.n_z, heat_capacity, density, dx, dy, dz, surface, surface_reduced, delta_T)
    #temperature = sample_holder_data(const.n_x, const.n_y, const.n_z, sample_holder, temperature, 110)
    #temperature = sample_holder_test(const.n_x, const.n_y, const.n_z, sample_holder, temperature)
    delta_T, Energy_Increase_per_Layer, Latent_Heat_per_Layer, Max_Fourier_number[j], E_sh, EcoPL = hte_calculate(const.n_x, const.n_y, const.n_z, surface, dT_0, temperature, Lambda, Dr, dx, dy, dz, const.dt, density, heat_capacity, sublimated_mass, resublimated_mass, latent_heat_water, sample_holder)
    #Delta_cond_ges += test_E_cond(const.n_x, const.n_y, const.n_z, surface, dT_0, temperature, Lambda, Dr, dx, dy, dz, const.dt, density, heat_capacity, sublimated_mass, resublimated_mass, latent_heat_water, sample_holder, Energy_conduction_per_Layer)
    #test_e_cond_2(const.n_x, const.n_y, const.n_z, Energy_Increase_per_Layer, EcoPL, 50, sample_holder)
    temperature, uniform_water_masses, heat_capacity, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, E_conservation[j], Energy_Increase_Total_per_time_Step_arr[j], E_Rad_arr[j], Latent_Heat_per_time_step_arr[j], E_In_arr[j] = update_thermal_arrays(const.n_x, const.n_y, const.n_z, temperature, uniform_water_masses, delta_T, Energy_Increase_per_Layer, sublimated_mass, resublimated_mass, const.dt, const.avogadro_constant, const.molar_mass_water, const.molar_mass_co2, heat_capacity, const.heat_capacity_water_ice, const.heat_capacity_co2_ice, EIis_0, Latent_Heat_per_Layer, E_Lat_0, E_Rad, E_In, E_sh)
    #temperature = sample_holder_test(const.n_x, const.n_y, const.n_z, sample_holder, temperature)
    #temperature = sample_holder_data(const.n_x, const.n_y, const.n_z, sample_holder, temperature, 77)
    #if len(empty_voxels) != 0:
        #surface, surface_reduced = update_surface_arrays(empty_voxels, surface, surface_reduced, temperature, const.n_x, const.n_y, const.n_z, a, a_rad, b, b_rad)
    sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, temperature_save = data_store_sensors(j, const.n_x, const.n_y, const.n_z, temperature, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, sett.data_reduce, temperature_save)
    #pressure, pressure_co2, highest_pressure, highest_pressure_co2 = pressure_calculation(j_leave, j_leave_co2, pressure, pressure_co2, temperature, var.dx, const.a_H2O, const.b_H2O, const.a_CO2, const.b_CO2, const.b, highest_pressure, highest_pressure_co2)
    #temperature, pressure, pressure_co2, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, total_ejection_events, ejection_times, drained_layers = check_drained_layers(const.n, temperature, pressure, pressure_co2, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, total_ejection_events, var.layer_strength, const.second_temp_layer, var.base_water_particle_number, var.base_co2_particle_number, const.dust_ice_ratio_global, const.co2_h2o_ratio_global, heat_capacity, const.heat_capacity_dust, const.heat_capacity_water_ice, const.heat_capacity_co2_ice, ejection_times, j, drained_layers)
    #print('---------- STEP ' + str(j) + ' ----------')
    if Max_Fourier_number[j] > 1/2:
        print('Instability warning')
        print('Fourier number: ' + str(Max_Fourier_number[j]))
        break
    if j % 50 == 0:
        print(E_conservation[j], E_In_arr[j], E_Rad_arr[j], Latent_Heat_per_time_step_arr[j], Energy_Increase_Total_per_time_Step_arr[j], EIis_0, E_sh, Econd_0)
        temperature_save[0] = temperature
        #print(temperature[1, 1:20, 40:60])
        #print(temperature[2, 1:20, 40:60])
    '''if j % sett.data_reduce == 0 or j == 0:
        #temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save = data_store(j, temperature, water_content_per_layer, co2_content_per_layer, outgassed_molecules_per_time_step/const.dt, outgassed_molecules_per_time_step_co2/const.dt, temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save, sett.data_reduction)
        #sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, temperature_save = data_store_sensors(j, const.n_x, const.n_y, const.n_z, temperature, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, sett.data_reduce, temperature_save)
        temperature_save[j//sett.data_reduce] = temperature
        water_content_save[j // sett.data_reduce] = uniform_water_masses
        sublimated_mass_save[j // sett.data_reduce] = sublimated_mass'''

#Data saving and output
#save_current_arrays(temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, ejection_times, var.time_passed + const.dt * const.k)
temperature_save[len(temperature_save)-1] = temperature
water_content_save[len(water_content_save)-1] = uniform_water_masses
sublimated_mass_save[len(sublimated_mass_save)-1] = sublimated_mass
#data_save(temperature_save, water_content_save, outgassing_rate, sublimated_mass_save, 'D:/Masterarbeit_data/' + 'Albedo_' + str(albedo) + '_just_surface_area')
#data_save(np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64), water_content_save, outgassing_rate, sublimated_mass_save,'D:/Masterarbeit_data/' + 'Albedo_0.80_Lambda_wi_' + str(lambda_water_ice) + '_lamp_test')
#print('Albedo: ' + str(albedo) + '\n Outgassed mass: ')
#print(np.sum([outgassing_rate[b] * const.dt for b in range(len(outgassing_rate))]))
#if np.sum([outgassing_rate[b] * const.dt for b in range(len(outgassing_rate))]) < 0.006E-3:
    #fac = -1
#print(sensor_10mm[1000:1100])
#data_save_sensors(temperature_save, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, 'D:/Masterarbeit_data/3D_temps_lamp_test_Albedo_0.80_Lambda_wi_' + str(lambda_water_ice), 'D:/Masterarbeit_data/sensor_temp_lamp_test_Albedo_0.80_Lambda_wi_' + str(lambda_water_ice))
data_dict = {'Temperature': temperature_save.tolist(), 'Surface': surface.tolist(), 'RSurface': surface_reduced.tolist(), 'HC': Lambda.tolist(), 'SH': sample_holder.tolist()}
with open('test_ec.json', 'w') as outfile:
    json.dump(data_dict, outfile)
print(np.max(Max_Fourier_number))
print('done')




