import numpy as np
import constants as const
import variables_and_arrays as var
import settings as sett
from tqdm import tqdm
import json
from surface_detection import create_equidistant_mesh, DEBUG_print_3D_arrays, find_surface, create_equidistant_mesh_2_layer
from thermal_parameter_functions import lambda_constant, lambda_granular, calculate_heat_capacity, lambda_sand
from boundary_conditions import energy_input, test, energy_input_data, sample_holder_data, calculate_L_chamber_lamp_bd, calculate_deeper_layer_source, day_night_cycle
from heat_transfer_equation_DG_ADI import hte_implicit_DGADI
from data_input import getPath, read_temperature_data, transform_temperature_data
from save_and_load import data_store, data_store_sensors, data_save_sensors
from utility_functions import thermal_reservoir, prescribe_temp_profile_from_data


'''albedo_arr = [0.8, 0.5, 0.3]
lambda_arr = [0.002, 0.003, 0.0074, 0.015, 0.2, 0.27]
absorption_depth_arr = [0.5E-3, 1E-3, 2E-3, 3E-3]'''
print('here')
albedo_arr = [0.95]
lambda_arr = [0.003]
absorption_depth_arr = [1E-3]
ambient_temperature_arr = [840]
ambient_temperature = 300
epsilon_arr = [0.95]
epsilon_ambient_arr = [0.95]
#abs_depth = absorption_depth_arr[0]
heat_capacity_sand = ambient_temperature_arr[0]
for albedo in albedo_arr:
    for lambda_sand_c in lambda_arr:
        for abs_depth in absorption_depth_arr:
            #for heat_capacity_sand in ambient_temperature_arr:
                for epsilon in epsilon_arr:
                    for epsilon_ambient in epsilon_ambient_arr:
                        print(albedo, lambda_sand_c, abs_depth, epsilon, epsilon_ambient)
                        temp_max_const = np.zeros(const.k, dtype=np.float64)
                        temp_max_daynight = np.zeros(const.k,dtype=np.float64)
                        for type in [1]:
                            #work arrays and mesh creation + surface detection
                            #temperature, dx, dy, dz, Dr, a, a_rad, b, b_rad = create_equidistant_mesh(const.n_x, const.n_y, const.n_z, const.temperature_ini, const.min_dx, const.min_dy, const.min_dz, False)
                            temperature, dx, dy, dz, Dr, a, a_rad, b, b_rad = create_equidistant_mesh_2_layer(const.n_x, const.n_y, const.n_z, const.temperature_ini, const.min_dx, const.min_dy, const.min_dz, 21, 10)
                            #temperature, dx, dy, dz, Dr, Lambda = one_d_test(const.n_x, const.n_y, const.n_z, const.min_dx, const.min_dy, const.min_dz, 'y')
                            heat_capacity = np.full(np.shape(temperature), heat_capacity_sand, dtype=np.float64)
                            density = var.density_sand
                            delta_T = var.delta_T
                            uniform_water_masses = 0 * dx * dy * dz
                            print(np.shape(temperature))
                            #DEBUG_print_3D_arrays(const.n_x, const.n_y, const.n_z, temperature)
                            surface, surface_reduced, sample_holder, mesh_shape_positive, mesh_shape_negative = find_surface(const.n_x, const.n_y, const.n_z, 0, 0, 0, const.n_x, const.n_y, const.n_z, temperature, var.surface, a, a_rad, b, b_rad, True, False)
                            #surface = var.surface
                            for i in range(0, const.n_z):
                                for j in range(0, const.n_y):
                                    for k in range(0, const.n_x):
                                        if sample_holder[i][j][k] == 1:
                                            density[i][j][k] = const.density_sample_holder_L
                                            heat_capacity[i][j][k] = const.heat_capacity_sample_holder_L
                            #surface_reduced = np.array([])
                            #print(surface[1][1][25])
                            j_leave = np.zeros(const.n + 1, dtype=np.float64)
                            j_inward = np.zeros(const.n + 1, dtype=np.float64)
                            j_leave_co2 = np.zeros(const.n + 1, dtype=np.float64)
                            j_inward_co2 = np.zeros(const.n + 1, dtype=np.float64)
                            water_content_per_layer = var.water_content_per_layer
                            co2_content_per_layer = var.co2_content_per_layer
                            E_conservation = var.E_conservation
                            Energy_Increase_Total_per_time_Step_arr = var.Energy_Increase_Total_per_time_Step_arr
                            E_Rad_arr = var.E_Rad_arr
                            Latent_Heat_per_time_step_arr = var.Latent_Heat_per_time_step_arr
                            E_In_arr = var.E_In_arr
                            temperature_save = np.zeros((const.k//sett.data_reduce, const.n_z, const.n_y, const.n_x))
                            sensor_10mm = np.zeros(const.k, dtype=np.float64)
                            sensor_20mm = np.zeros(const.k, dtype=np.float64)
                            sensor_35mm = np.zeros(const.k, dtype=np.float64)
                            sensor_55mm = np.zeros(const.k, dtype=np.float64)
                            sensor_90mm = np.zeros(const.k, dtype=np.float64)
                            Max_Fourier_number = np.zeros(const.k, dtype=np.float64)

                            #test(const.r_H, const.albedo, const.dt, const.Input_Intensity, dx, dy, surface, surface_reduced)
                            print(np.sum(Dr[1:const.n_z-1, const.n_y//2, const.n_x//2, 0]))
                            print(Dr[1:const.n_z-1, const.n_y//2, const.n_x//2, 0])
                            print(np.sum(dy[const.n_z//2, 1:const.n_y-1, const.n_x//2]))
                            #print(temperature)
                            #temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, time_passed = load_from_save()
                            #np.savetxt("D:/Masterarbeit_data/surface_temp.csv", surface_temp, delimiter=",")
                            #np.savetxt("D:/Masterarbeit_data/sample_holder_temp.csv", sample_holder_temp, delimiter=",")
                            lamp_power = calculate_L_chamber_lamp_bd(24, 'L', const.n_x, const.n_y, const.n_z, const.min_dx, const.min_dy, const.min_dz, True, abs_depth)
                            S_p = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
                            S_c = calculate_deeper_layer_source(const.n_x, const.n_y, const.n_z, lamp_power, const.r_H, albedo, surface, dx, dy, dz)
                            #data_save_file = 'C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Sand_sim_thesis_' + str(albedo) + '_Absdepth_' + str(abs_depth) + '_Lambda_' + str(lambda_sand_c) +'.json'
                            #data_save_file = 'D:/TPM_Data/Big_sand/Thesis_run/Periodic_sand_sim_thesis_' + str(albedo) + '_Absdepth_' + str(abs_depth) + '_Lambda_' + str(lambda_sand_c) +'.json'
                            #data_save_file = 'D:/TPM_Data/Big_sand/Thesis_run/varying_epsilon' + str(epsilon) + '_ambient_epsilon' + str(epsilon_ambient) + '_test_308K.json'
                            data_save_file = 'D:/TPM_Data/Big_sand/Thesis_run/const_illum_best_fit_am_300K.json'
                            #data_save_file_2 = 'D:/TPM_data/Big_sand/Sand_sim_thesis_' + str(albedo) + '_Absdepth_' + str(abs_depth) + '_Lambda_' + str(lambda_sand_c) + '.json'
                            #data_save_file_2 = 'D:/TPM_Data/Big_sand/Thesis_run/Periodic_surface_sand_sim_thesis_' + str(albedo) + '_Absdepth_' + str(abs_depth) + '_Lambda_' + str(lambda_sand_c) +'.json'
                            data_save_file_2 = 'D:/TPM_Data/Big_sand/Thesis_run/const_illum_surface_best_fit_am_300K.json'
                            middle_slices = np.zeros((const.k, const.n_z), dtype=np.float64)
                            sensors = np.zeros((const.k, 5), dtype=np.float64)
                            outer_sensors = np.zeros((const.k, 5), dtype=np.float64)
                            temp_surface = np.zeros((const.n_y-2, const.n_x-2), dtype=np.float64)
                            #temperature = prescribe_temp_profile_from_data(const.n_x, const.n_y, const.n_z, temperature, 305, 301, 'C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/temps_sandy_randy.txt', '2023-07-20 13:53:07', 11, 21, 22, 24, 27, const.min_dx, const.min_dy, 0.125)
                            temperature = sample_holder_data(const.n_x, const.n_y, const.n_z, sample_holder, temperature, 301)
                            '''
                            Main Loop of the model. Comment out/Uncomment function calls to disable/enable features
                            '''
                            Lambda = lambda_sand(const.n_x, const.n_y, const.n_z, temperature, Dr, lambda_sand_c, sample_holder, const.lambda_sample_holder_L, var.sensor_positions)
                            #print(temperature[1, 0:const.n_y, const.n_x // 2])
                            #print(np.isnan(temperature).any())
                            #print(1)
                            for j in tqdm(range(0, const.k)):
                                middle_slices[j] = temperature[0:const.n_z, const.n_y//2, const.n_x//2].copy()
                                '''sensors[j][0] = temperature[11][const.n_y//2][const.n_x//2]
                                sensors[j][1] = temperature[21][const.n_y//2+3][const.n_x//2]
                                sensors[j][2] = temperature[22][const.n_y//2-3][const.n_x//2]
                                sensors[j][3] = temperature[24][const.n_y//2][const.n_x//2]
                                sensors[j][4] = temperature[27][const.n_y//2+3][const.n_x//2-1]
                                outer_sensors[j][0] = temperature[11][const.n_y//2-8][const.n_x//2-8]
                                outer_sensors[j][1] = temperature[21][const.n_y//2+3-8][const.n_x//2-8]
                                outer_sensors[j][2] = temperature[22][const.n_y//2-3-8][const.n_x//2-8]
                                outer_sensors[j][3] = temperature[24][const.n_y//2-8][const.n_x//2-8]
                                outer_sensors[j][4] = temperature[27][const.n_y//2+3-8][const.n_x//2-8]'''
                                sensors[j][0] = temperature[11][const.n_y//2][const.n_x//2]
                                sensors[j][1] = temperature[21][const.n_y//2+3][const.n_x//2]
                                sensors[j][2] = temperature[22][const.n_y//2-3][const.n_x//2] * 4/5 + temperature[23][const.n_y//2-3][const.n_x//2] * 1/5
                                sensors[j][3] = temperature[24][const.n_y//2][const.n_x//2]
                                sensors[j][4] = temperature[27][const.n_y//2+3][const.n_x//2-1]
                                outer_sensors[j][0] = temperature[11][const.n_y//2-8][const.n_x//2-8] * 3/4 + temperature[11][const.n_y//2-7][const.n_x//2-7] * 1/4
                                outer_sensors[j][1] = temperature[21][const.n_y//2+3-8][const.n_x//2-8] * 3/4 + temperature[21][const.n_y//2+3-7][const.n_x//2-7] * 1/4
                                outer_sensors[j][2] = (temperature[22][const.n_y//2-3-8][const.n_x//2-8] * 3/4 + temperature[22][const.n_y//2-3-7][const.n_x//2-7] * 1/4) * 4/5 + (temperature[23][const.n_y//2-3-8][const.n_x//2-8] * 3/4 + temperature[23][const.n_y//2-3-7][const.n_x//2-7] * 1/4) * 1/5
                                outer_sensors[j][3] = temperature[24][const.n_y//2-8][const.n_x//2-8] * 3/4 + temperature[24][const.n_y//2-7][const.n_x//2-7] * 1/4
                                outer_sensors[j][4] = temperature[27][const.n_y//2+3-8][const.n_x//2-8] * 3/4 + temperature[27][const.n_y//2+3-8][const.n_x//2-8] * 1/4
                                if j * const.dt == 120000:
                                    temp_surface = temperature[1, 1:const.n_y-1, 1:const.n_x-1].copy()
                                #sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm = data_store_sensors(j, const.n_x, const.n_y, const.n_z, temperature, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, sett.data_reduce)
                                #data_save_sensors(j * const.dt, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, data_save_file)
                                #Lambda = lambda_granular(const.n_x, const.n_y, const.n_z, temperature, Dr, dx, dy, dz, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.surface_energy_par, const.r_mono, const.f_1, const.f_2, var.VFF_pack, const.sigma, const.e_1, sample_holder, const.lambda_sample_holder)
                                #Lambda = lambda_sand(const.n_x, const.n_y, const.n_z, temperature, Dr, const.lambda_sand, sample_holder, const.lambda_sample_holder)
                                #heat_capacity = calculate_heat_capacity(temperature)
                                #lamp_power_dn, S_c_dn = day_night_cycle(lamp_power, S_c, 3800, j*const.dt)
                                if type == 1:
                                    temperature = hte_implicit_DGADI(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, albedo, const.dt, lamp_power, const.sigma, epsilon, epsilon_ambient, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder, ambient_temperature)
                                else:
                                    lamp_power_dn, S_c_dn = day_night_cycle(lamp_power, S_c, 3800, j * const.dt)
                                    temperature = hte_implicit_DGADI(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, albedo, const.dt, lamp_power_dn, const.sigma, epsilon, epsilon_ambient, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c_dn, S_p, sample_holder, ambient_temperature)
                                #print(temperature[1, 0:const.n_y, const.n_x//2])
                                #temperature = hte_implicit_DGADI(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, albedo, const.dt, lamp_power, const.sigma, const.epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder, ambient_temperature)
                                #print(temperature[0:const.n_z, const.n_y//2-20, const.n_x//2])
                                #print(temperature[0:const.n_z, 3, 23])
                                #temperature = thermal_reservoir(const.n_x, const.n_y, 1, temperature, 300, sample_holder)
                                #print(temperature[0:const.n_z, const.n_y // 2 - 20, const.n_x // 2])
                                if type == 1:
                                    temp_max_const[j] = np.max(temperature[1, 1:const.n_y-1, 1:const.n_x-1])
                                else:
                                    temp_max_daynight[j] = np.max(temperature[1, 1:const.n_y-1, 1:const.n_x-1])

                            #Data saving and output
                            data_dict = {'Temperature': sensors.tolist(), 'Temperature Outer': outer_sensors.tolist()}
                            with open(data_save_file, 'w') as outfile:
                                json.dump(data_dict, outfile)
                            #print(np.max(Max_Fourier_number))
                            #print('done')
            
                            data_dict_2 = {'Temperature': middle_slices.tolist(), 'Temperature Surface': temp_surface.tolist()}
                            with open(data_save_file_2, 'w') as outfile:
                                json.dump(data_dict_2, outfile)
                            # print(np.max(Max_Fourier_number))
                            print('done')
            
                            print(temperature[0:const.n_z, const.n_y//2, const.n_x//2])
                        '''data_dict = {'Temp Const': temp_max_const.tolist(), 'Temp DN': temp_max_daynight.tolist()}
                        with open(data_save_file, 'w') as outfile:
                            json.dump(data_dict, outfile)'''