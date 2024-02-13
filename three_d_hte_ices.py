import numpy as np
import constants as const
import variables_and_arrays as var
import settings as sett
from tqdm import tqdm
import json
from os import listdir
from surface_detection import create_equidistant_mesh, DEBUG_print_3D_arrays, find_surface, surrounding_checker_moon, update_surface_arrays_slow, update_surface_arrays, get_sample_holder_adjacency, create_equidistant_mesh_2_layer
from thermal_parameter_functions import calculate_latent_heat, calculate_density, thermal_functions, calculate_bulk_density_and_VFF, thermal_conductivity_moon_regolith, heat_capacity_moon_regolith, calculate_water_grain_radius, calculate_heat_capacity, lambda_granular
from molecule_transfer import calculate_molecule_surface, diffusion_parameters_moon, calculate_source_terms, pressure_calculation, calculate_source_terms_linearised, calculate_molecule_flux_moon_test, sinter_neck_calculation_time_dependent, sintered_surface_checker
from heat_transfer_equation_DG_ADI import hte_implicit_DGADI, hte_implicit_DGADI_zfirst
from diffusion_equation_DG_ADI import de_implicit_DGADI, de_implicit_DGADI_zfirst
from boundary_conditions import sample_holder_data, day_night_cycle, calculate_L_chamber_lamp_bd
from ray_tracer import generate_topography, trace_rays_MC, get_temperature_vector
from utility_functions import save_sensors_L_sample_holder, save_sensors_L_sample_holder_high_res, prescribe_temp_profile_from_data, prescribe_crater, artificial_crater_heating, save_mean_temps_light_spot

#albedo_arr = [0.95, 0.8, 0.85, 0.8, 0.75]
#lambda_arr = [0.16, 0.14, 0.12, 0.1, 0.8]
albedo_arr = [0.85]
sinter_temp_reduce_arr = [1E-4]
for run in range(0, 1):
    for albedo in albedo_arr:
        for sinter_temp_reduce in sinter_temp_reduce_arr:
            #work arrays and mesh creation + surface detection
            temperature, dx, dy, dz, Dr, a, a_rad, b, b_rad = create_equidistant_mesh(const.n_x, const.n_y, const.n_z, const.temperature_ini, const.min_dx, const.min_dy, const.min_dz, False)
            #temperature, dx, dy, dz, Dr, a, a_rad, b, b_rad = create_equidistant_mesh_2_layer(const.n_x, const.n_y, const.n_z, const.temperature_ini, const.min_dx, const.min_dy, const.min_dz, 21, 10)

            '''data_file_sensors = 'C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Ice/Thesis/Agilent_L_chamber_30_11_2023_14_20_44.txt'
            height_list_sensors = np.array([2, 3, 4, 5, 6, 7, 9, 11, 16, 20])
            surface, surface_reduced, sample_holder, mesh_shape_positive, mesh_shape_negative = find_surface(const.n_x, const.n_y, const.n_z, 0, 0, 0, const.n_x, const.n_y, const.n_z, temperature, var.surface, a, a_rad, b, b_rad, True, False)
            temperature = prescribe_temp_profile_from_data(const.n_x, const.n_y, const.n_z, temperature, 45770, 153, 145, data_file_sensors, height_list_sensors, sample_holder)
            '''
            temperature = prescribe_crater(const.n_x, const.n_y, const.n_z, temperature, Dr, 0.024, 9E-3, const.min_dz)

            '''data_dict = {'Temperature': temperature.tolist()}
            with open('test.json', 'w') as outfile:
                json.dump(data_dict, outfile)'''
            #temperature, dx, dy, dz, Dr, Lambda = one_d_test(const.n_x, const.n_y, const.n_z, const.min_dx, const.min_dy, const.min_dz, 'y')
            heat_capacity = var.heat_capacity
            #density = var.density * const.VFF_pack_const
            #density = var.density_sand
            delta_T = var.delta_T
            print(np.shape(temperature))
            #DEBUG_print_3D_arrays(const.n_x, const.n_y, const.n_z, temperature)
            surface, surface_reduced, sample_holder, mesh_shape_positive, mesh_shape_negative = find_surface(const.n_x, const.n_y, const.n_z, 0, 0, 0, const.n_x, const.n_y, const.n_z, temperature, var.surface, a, a_rad, b, b_rad, True, False)
            print(len(surface_reduced))
            water_ice_grain_density = calculate_density(temperature, var.VFF_pack)[0]
            density = water_ice_grain_density * (1 / (const.dust_ice_ratio_global + 1)) * var.VFF_pack + const.density_TUBS_M * np.ones((const.n_z, const.n_y, const.n_x), dtype=np.float64) * (const.dust_ice_ratio_global / (const.dust_ice_ratio_global + 1)) * var.VFF_pack
            uniform_water_masses = density * dx * dy * dz * (1 / (const.dust_ice_ratio_global + 1))
            uniform_dust_masses = density * dx * dy * dz * (const.dust_ice_ratio_global / (const.dust_ice_ratio_global + 1))
            for i in range(0, const.n_z):
                for j in range(0, const.n_y):
                    for k in range(0, const.n_x):
                        if temperature[i][j][k] == 0: #or sample_holder[i][j][k] == 1:
                            uniform_water_masses[i][j][k] = 0
                            uniform_dust_masses[i][j][k] = 0

            water_particle_number = uniform_water_masses / (4/3 * np.pi * const.r_mono_water**3 * water_ice_grain_density)
            print(np.sum(uniform_water_masses[1]))
            density, VFF = calculate_bulk_density_and_VFF(temperature, var.VFF_pack, uniform_dust_masses, uniform_water_masses, const.density_TUBS_M, dx, dy, dz)[0:2]
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
            Q = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            r_mono_water = np.full((const.n_z, const.n_y, const.n_x), const.r_mono_water, dtype=np.float64)

            #test(const.r_H, const.albedo, const.dt, const.Input_Intensity, dx, dy, surface, surface_reduced)



            #print(temperature)
            #temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, time_passed = load_from_save()
            #2023-02-15 17:45:01 2023-02-15 17:45:02
            #2023-02-15 20:45:41
            #time_deltas_data_surface, surface_temp = read_temperature_data(getPath(), '2023-02-15 16:45:00', '2023-02-15 17:45:02', [1], [])
            #max_k, surface_temp = transform_temperature_data(const.k, const.dt, np.array(time_deltas_data_surface), [], surface_temp)

            #lamp_power = calculate_L_chamber_lamp_bd(15.33, 'M', const.n_x, const.n_y, const.n_z)
            '''lamp_power = np.full((const.n_z, const.n_y, const.n_x), const.solar_constant*const.min_dx*const.min_dy, dtype=np.float64)'''
            max_k = const.k
            '''with open('lamp_input_S_chamber.json') as json_file:
                data_e_in = json.load(json_file)
            lamp_power = np.array(data_e_in['Lamp Power'])
            json_file.close()'''

            #np.savetxt("D:/Masterarbeit_data/surface_temp.csv", surface_temp, delimiter=",")
            #np.savetxt("D:/Masterarbeit_data/sample_holder_temp.csv", sample_holder_temp, delimiter=",")
            latent_heat_water = np.full((const.n_z, const.n_y, const.n_x), const.latent_heat_water, dtype=np.float64)
            '''sample_holder = np.zeros((const.n_z, const.n_y, const.n_x))
            sample_holder[const.n_z-1][0][0] = 1
            surface = np.zeros((const.n_z, const.n_y, const.n_x, 6))'''
            #temperature = sample_holder_data(const.n_x, const.n_y, const.n_z, sample_holder, temperature, 110)
            #Lambda = lambda_granular(const.n_x, const.n_y, const.n_z, temperature, Dr, dx, dy, dz, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.surface_energy_par, const.r_mono, const.f_1, const.f_2, var.VFF_pack, const.sigma, const.e_1, sample_holder, const.lambda_sample_holder)
            #print(Lambda[15][0][0])
            Delta_cond_ges = 0
            outgassed_mass_complete = 0

            #file = open('D:/Masterarbeit_data/Sand_no_tubes/Results/sensor_data_lambda_' + str(round(lambda_sand_val, 5)) + '.csv', 'a')
            #file = open('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Sand(no_tubes)/sand_lambda_' + str(const.lambda_sand) + '.csv', 'a')
            #Lambda = lambda_sand(const.n_x, const.n_y, const.n_z, temperature, Dr, const.lambda_sand, sample_holder, const.lambda_sample_holder, var.sensor_positions)
            #for j in tqdm(range(0, min(const.k, max_k, max_k_2))):
            temperature_previous = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            gas_density = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            S_c = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            S_p = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            sub_gas_begin = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            sub_gasdens_begin = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            gas_density_previous = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            temp_begin =np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)

            #mesh_shape_positive[1] = np.zeros((const.n_y, const.n_x), dtype=np.int32)
            #mesh_shape_negative = 1 - mesh_shape_positive
            #print(sh_adjacent_voxel[0:2, 10:20, 10:20])
            max_temp = np.zeros(const.k, dtype=np.float64)
            r_n_start = (const.R_JKL*const.K_JKL*(const.P_JKL + 3*const.surface_energy_par*np.pi*const.R_JKL + np.sqrt(6*const.surface_energy_par*np.pi*const.R_JKL*const.P_JKL + (3*const.surface_energy_par*np.pi*const.R_JKL)**2)))**(1/3)
            r_n = np.full((const.n_z, const.n_y, const.n_x), r_n_start, dtype=np.float64)
            S_c_deeper = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            lamp_power = calculate_L_chamber_lamp_bd(24, 'L', const.n_x, const.n_y, const.n_z, const.min_dx, const.min_dy, const.min_dz, False, 0)
            lamp_power_dn = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            blocked_voxels = sintered_surface_checker(const.n_x, const.n_y, const.n_z, r_n, r_mono_water)

            r_mono_base = r_mono_water.copy()
            r_n_base = r_n.copy()
            max_temps = np.zeros(const.k, dtype=np.float64)
            sublimated_mass_mid = np.zeros(const.k, dtype=np.float64)
            surface_topography_polygons = np.empty(0)
            reradiated_heat = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            sensors_right = np.zeros((const.k, 11), dtype=np.float64)
            sensors_rear = np.zeros((const.k, 11), dtype=np.float64)
            data_file_sensors = 'C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Ice/Thesis/Agilent_L_chamber_30_11_2023_14_20_44.txt'
            #height_list_sensors = np.array([2, 3, 4, 5, 6, 7, 9, 11, 16, 20])
            height_list_sensors = np.array([11, 21, 31, 41, 51, 61, 81, 101, 151, 200])
            #height_list_sensors = np.array([6, 11, 16, 21, 26, 31, 41, 51, 76, 100])

            '''latent_heat_water = calculate_latent_heat(temperature, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.R, const.m_mol)
            heat_capacity = calculate_heat_capacity(temperature)
            
            #print(temperature[0:const.n_z, const.n_y//2, const.n_x//2])
            #print(temperature[0:const.n_z, 1, const.n_x//2])
            density = calculate_density(temperature, VFF)[1]
            density = density + sample_holder * (const.density_copper - density[const.n_z-2, const.n_y//2, const.n_x//2])
            
            Lambda = np.full((const.n_z, const.n_y, const.n_x, 6), 0.1, dtype=np.float64)
            
            temp = lamp_power.copy()
            lamp_power = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            #lamp_power[1, const.n_y//2-1:const.n_y//2+2, const.n_x//2-1:const.n_x//2+2] = temp[1, const.n_y//2-1:const.n_y//2+2, const.n_x//2-1:const.n_x//2+2]
            lamp_power[1, const.n_y//2-1:const.n_y//2+2, const.n_x//2-1:const.n_x//2+2] = np.full((3, 3), temp[1, const.n_y//2, const.n_x//2] * 2)'''

            sinter_time_reducer = sinter_temp_reduce
            #sinter_time_reducer = 80
            if run == 0:
                name_string = 'CD_Final_Granular_ice_L_albedo_' + str(albedo) + '_sinter_reduction_factor_' + str(sinter_temp_reduce) + '.json'
            if run == 1:
                name_string = 'CD_Final_Granular_ice_L_albedo_' + str(albedo) + '_sinter_reduction_factor_' + str(sinter_temp_reduce) + '_wall_0.90_.json'

            '''
            Main Loop of the model. Comment out/Uncomment function calls to disable/enable features
            '''

            for j in tqdm(range(0, const.k)):
                if j % 100 == 0:
                    print(temperature[0:const.n_z, const.n_y // 2, const.n_x // 2])
                    print(lamp_power_dn[1][26][26])
                    #s_1 = sinter_neck_calculation_time_dependent(r_n, r_mono_water, const.dt, temperature, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.molecular_volume_H2O, const.surface_energy_par, const.R, const.r_mono, const.packing_geometry_factor, const.molar_mass_water, density, pressure, const.m_H2O, const.k_boltzmann, 0.001, water_particle_number, blocked_voxels, const.n_x, const.n_y, const.n_z, sample_holder, dx, dy, dz, surface)[2]
                    #print(s_1[0:5, const.n_y//2, const.n_x//2])
                    #s_2 = calculate_molecule_surface(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_H2O, dx, dy, dz, const.dt, surface_reduced, const.k_boltzmann, uniform_water_masses)[0]
                    #print(s_2[0:5, const.n_y//2, const.n_x//2])
                    if j != 0:
                        print(Lambda[5:10, const.n_y//2, const.n_x//2])
                        print(blocked_voxels[5, 0:const.n_y, const.n_x // 2])
                        print(uniform_water_masses[5:10, const.n_y//2, const.n_x//2])
                        print(np.sum(sublimated_mass_mid[j-100:j]))
                    #print(np.sum(uniform_water_masses) + outgassed_mass_complete)
                    #np.save('D:/TPM_Data/Luwex/only_temps_equilibriated/only_temperature_sim_' + str(j * const.dt) + '.npy', temperature)
                    #np.save('D:/TPM_Data/Luwex/sublimation_test/sublimation_test' + str(j * const.dt) + '.npy', temperature)
                    #np.save('D:/TPM_Data/Luwex/sublimation_test/WATERsublimation_test' + str(j * const.dt) + '.npy', uniform_water_masses)
                    #np.save('D:/TPM_Data/Luwex/sublimation_and_diffusion/GASsublimation_and_diffusion' + str(j * const.dt) + '.npy', gas_density * dx * dy * dz)
                #temperature_previous = temperature[0:const.n_z, 0:const.n_y, 0:const.n_x]
                #save_mean_temps_light_spot(const.n_x, const.n_y, const.n_z, temperature, 'D:/TPM_Data/Ice/a_0.85_srf_0.0001_std_rt.csv')
                #sensors_right, sensors_rear = save_sensors_L_sample_holder(const.n_x, const.n_y, const.n_z, temperature, sensors_right, sensors_rear, j)
                sensors_right, sensors_rear = save_sensors_L_sample_holder_high_res(const.n_x, const.n_y, const.n_z, temperature, sensors_right, sensors_rear, j, height_list_sensors, sf=1)
                if sett.enable_ray_tracing and len(surface_topography_polygons) != 0:
                    reradiated_heat = get_temperature_vector(const.n_x, const.n_y, const.n_z, temperature, surface, surface_reduced, len(surface_topography_polygons), view_factor_matrix, const.sigma, const.epsilon, albedo, lamp_power_dn, dx, dy, dz)[0]
                lamp_power_dn, S_c_deeper, activity_factor = day_night_cycle(lamp_power, S_c_deeper, 3 * 3600, j * const.dt, const.activity_threshold, const.activity_split)
                #density, VFF, water_ice_grain_density = calculate_bulk_density_and_VFF(temperature, VFF, uniform_dust_masses, uniform_water_masses, const.density_TUBS_M, dx, dy, dz)
                density = calculate_density(temperature, VFF)[1]
                r_n, r_mono_water, sublimated_mass, areas = sinter_neck_calculation_time_dependent(r_n, r_mono_water, const.dt, temperature, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.molecular_volume_H2O, const.activation_energy_water_ice, const.R, const.r_mono, const.packing_geometry_factor, const.molar_mass_water, density, pressure, const.m_H2O, const.k_boltzmann, 0.001, water_particle_number, blocked_voxels, const.n_x, const.n_y, const.n_z, sample_holder, dx, dy, dz, surface, sinter_time_reducer)
                #sublimated_mass, empty_voxels = calculate_molecule_surface(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_H2O, dx, dy, dz, const.dt, surface_reduced, const.k_boltzmann, uniform_water_masses)[0:2]
                blocked_voxels = sintered_surface_checker(const.n_x, const.n_y, const.n_z, r_n, r_mono_water)
                #r_mono_water = calculate_water_grain_radius(const.n_x, const.n_y, const.n_z, uniform_water_masses, water_ice_grain_density, water_particle_number, r_mono_water)
                latent_heat_water = calculate_latent_heat(temperature, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.R, const.m_mol)
                density = density + sample_holder * (const.density_copper - density[const.n_z-2, const.n_y//2, const.n_x//2])
                #Lambda, interface_temperatures = thermal_conductivity_moon_regolith(const.n_x, const.n_y, const.n_z, temperature, dx, dy, dz, Dr, VFF, const.r_mono, const.fc1, const.fc2, const.fc3, const.fc4, const.fc5, const.mu, const.E, const.gamma, const.f1, const.f2, const.e1, const.chi_maria, const.sigma, const.epsilon, uniform_water_masses, uniform_dust_masses, const.lambda_water_ice, const.lambda_sample_holder, sample_holder)
                Lambda, interface_temperatures = lambda_granular(const.n_x, const.n_y, const.n_z, temperature, Dr, dx, dy, dz, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.activation_energy_water_ice, const.R, r_mono_water, const.f_1, const.f_2, VFF, const.sigma, const.e_1, sample_holder, const.lambda_copper, r_n)
                #Lambda = np.full((const.n_z, const.n_y, const.n_x, 6), lambda_const, dtype=np.float64)
                #for j in range (1, const.n_y-1):
                    #for k in range(1, const.n_x-1):
                        #if ((k - a) / a_rad) ** 2 + ((j - b) / b_rad) ** 2 >= 0.8 and ((k - a) / a_rad) ** 2 + ((j - b) / b_rad) ** 2 < 1:
                            #Lambda[1:const.n_z-1, j, k] = np.full((const.n_z-2, 6), 0.01, dtype=np.float64)
                #print(Lambda[1, 0:const.n_y, const.n_x//2, 1])
                #heat_capacity = heat_capacity_moon_regolith(const.n_x, const.n_y, const.n_z, temperature, const.c0, const.c1, const.c2, const.c3, const.c4, uniform_water_masses, uniform_dust_masses, const.heat_capacity_sample_holder, sample_holder)
                heat_capacity = calculate_heat_capacity(temperature)
                S_c, S_p, Scde, Spde, empty_voxels = calculate_source_terms(const.n_x, const.n_y, const.n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, const.dt, surface_reduced, uniform_water_masses, latent_heat_water, surface)
                #S_c, S_p = calculate_source_terms(const.n_x, const.n_y, const.n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, const.dt, surface_reduced, uniform_water_masses, latent_heat_water, surface)[0:2]
                #S_c, S_p, sublimated_mass, outgassed_mass_timestep = calculate_molecule_flux_moon_test(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_H2O, dx, dy, dz, const.dt, const.k_boltzmann, sample_holder, uniform_water_masses, latent_heat_water, water_particle_number, r_mono_water)
                #print(Lambda[2, 0:const.n_y, const.n_x // 2])
                temperature = hte_implicit_DGADI_zfirst(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, albedo, const.dt, lamp_power_dn, const.sigma, const.epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder, np.full(6, const.ambient_radiative_temperature, dtype=np.float64), reradiated_heat)
                #print(temperature[0:const.n_z, const.n_y // 2, const.n_x // 2])
                outgassed_mass_complete += np.sum(sublimated_mass)
                uniform_water_masses = uniform_water_masses - sublimated_mass * activity_factor
                uniform_water_masses = np.maximum(uniform_water_masses, np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64))
                outgassing_rate[j] = np.sum(sublimated_mass)/const.dt
                #print(temperature[1, const.n_y // 2, const.n_x // 2], uniform_water_masses[1, const.n_y // 2, const.n_x // 2], sublimated_mass[1, const.n_y // 2, const.n_x // 2], empty_voxels, density[1, const.n_y // 2, const.n_x // 2], VFF[1, const.n_y // 2, const.n_x // 2])
                if len(empty_voxels) != 0 or j == 0:
                    #surface, surface_reduced, temperature, r_n, r_mono_water = update_surface_arrays_slow(empty_voxels, surface, surface_reduced, temperature, const.n_x, const.n_y, const.n_z, a, a_rad, b, b_rad, False, r_n, r_mono_water)
                    #print(np.isnan(temperature).any())
                    surface, surface_reduced = update_surface_arrays(empty_voxels, surface, surface_reduced, temperature, const.n_x, const.n_y, const.n_z, a, a_rad, b, b_rad, False)
                    if sett.enable_ray_tracing:
                        surface_topography_polygons = generate_topography(surface, surface_reduced, dx, dy, dz)
                        print('Tracing rays')
                        view_factor_matrix = trace_rays_MC(surface_topography_polygons, 3000, True, surface)
                        print('Finished')
                max_temps[j] = np.max(temperature)
                sublimated_mass_mid[j] = np.sum(sublimated_mass[0:const.n_z, const.n_y//2, const.n_x//2])
                #if j == 2646:
                    #np.save('D:/TPM_Data/Ice/a_0.85_srf_0.0001_13th_cycle', temperature)
                #if (j + 162) % 216 == 0:
                    #np.save('D:/TPM_Data/Ice/no_rt_cycle' + str((j + 162)//216), temperature)
                '''if j >= 50 and j <= 59:
                    np.save('D:/TPM_Data/Ice/rt_j_' + str(j), temperature)
                if j >= 260:
                    np.save('D:/TPM_Data/Ice/rt_j_' + str(j), temperature)
                #print(temperature[0:const.n_z - 2, const.n_y // 2 - 4:const.n_y // 2 + 5, const.n_x // 2 - 4:const.n_x // 2 + 5])
                if run == 1:
                    temperature = artificial_crater_heating(const.n_x, const.n_y, const.n_z, temperature, surface_reduced, 0.90)'''
                #print(np.sum(sublimated_mass))
                #if np.max(np.abs(temperature - temperature_previous)) < 50E-6:
                    #break


            #Data saving and output
            #save_current_arrays(temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, ejection_times, var.time_passed + const.dt * const.k)
            print(name_string)
            data_dict = {'Right': sensors_right.tolist(), 'Rear': sensors_rear.tolist(), 'Outgassing rate': outgassing_rate.tolist(), 'Temperature': temperature.tolist()}
            with open('D:/TPM_Data/Ice/' + name_string, 'w') as outfile:
                json.dump(data_dict, outfile)

'''#data_save_sensors(const.k * const.dt, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, file)
#file.close()
data_dict = {'Temperature': temperature.tolist(), 'Surface': surface.tolist(), 'RSurface': surface_reduced.tolist(), 'HC': Lambda.tolist(), 'SH': sample_holder.tolist()}
with open('test_no_ray_trace.json', 'w') as outfile:
    json.dump(data_dict, outfile)
print('done')
'''

'''
Configuration for the heat transfer + gas diffusion
'''

'''
blocked_voxels = np.full((const.n_z, const.n_y, const.n_x), 1, dtype=np.float64)
#print(uniform_water_masses[0:const.n_z, const.n_y//2, const.n_x//2])
passed_time = 0
next_step = 0.0
dyn_dt = const.dt
j_stop = const.k
#S_c_light = calculate_deeper_layer_source_from_values(const.n_x, const.n_y, const.n_z, lamp_power, const.r_H, const.albedo, temperature, dx, dy, dz, absorption_factors)
inner_structure = np.zeros((const.n_z, const.n_y, const.n_x, 6), dtype=np.float64)

for j in tqdm(range(0, const.k)):
    #print(temperature[0:const.n_z, 1, const.n_x//2])
    if j % 50 == 0:
        print(temperature[0:const.n_z, const.n_y // 2, const.n_x // 2])
        print(temperature[1, 0:const.n_y, const.n_x // 2])
        #print(sublimated_mass[0:const.n_z, const.n_y // 2, const.n_x // 2])
        #print(sublimated_mass[1, 0:const.n_y, const.n_x // 2])
        print('G:', gas_density[0:const.n_z, const.n_y // 2, const.n_x // 2])
        print('P:', pressure[0:const.n_z, const.n_y // 2, const.n_x // 2])
        print(np.max(gas_density), np.min(gas_density[2:const.n_z-1, 1:const.n_y-1, 1:const.n_x-1]))
        print(np.max(temperature), np.min(temperature[2:const.n_z-1, 1:const.n_y-1, 1:const.n_x-1]))
        #print(lamp_power_dn[0:const.n_z, const.n_y // 2, const.n_x // 2])
    if passed_time >= next_step or True:
        np.save('D:/TPM_Data/Ice/Diffusion/full_model/temperatures_' + str(round(passed_time, 1)) + '.npy', temperature)
        np.save('D:/TPM_Data/Ice/Diffusion/full_model/water_masses_' + str(round(passed_time, 1)) + '.npy', uniform_water_masses)
        #np.save('D:/TPM_Data/Noah/diffusion_sh/r_n_' + str(round(passed_time, 1)) + '.npy', r_n)
        #np.save('D:/TPM_Data/Noah/diffusion_sh/r_mono_' + str(round(passed_time, 1)) + '.npy', r_mono_water)
        #np.save('D:/TPM_Data/Noah/diffusion_sh/pressure_' + str(round(passed_time, 1)) + '.npy', pressure)
        next_step += 100
    #density, VFF, water_ice_grain_density = calculate_bulk_density_and_VFF(temperature, VFF, uniform_dust_masses, uniform_water_masses, const.density_TUBS_M, dx, dy, dz)
    #r_mono_water = calculate_water_grain_radius(const.n_x, const.n_y, const.n_z, uniform_water_masses, water_ice_grain_density, water_particle_number, r_mono_water)
    lamp_power_dn, S_c_deeper, activity_factor = day_night_cycle(lamp_power, S_c_deeper, 3 * 3600, j * const.dt, const.activity_threshold, const.activity_split)
    #S_c_light = calculate_deeper_layer_source_from_values(const.n_x, const.n_y, const.n_z, lamp_power_dn, const.r_H, const.albedo, temperature, dx, dy, dz, absorption_factors)
    r_n, r_mono_water, sublimated_mass, areas = sinter_neck_calculation_time_dependent_diffusion_t(r_n, r_mono_water, dyn_dt, temperature, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.molecular_volume_H2O, const.surface_energy_par, const.R, const.r_mono, const.packing_geometry_factor, const.molar_mass_water, density, pressure, const.m_H2O, const.k_boltzmann, 0.001, water_particle_number, blocked_voxels, const.n_x, const.n_y, const.n_z, sample_holder, dx, dy, dz, gas_density, areas, surface, sinter_reduction_factor)
    blocked_voxels = sintered_surface_checker(const.n_x, const.n_y, const.n_z, r_n, r_mono_water)
    latent_heat_water = calculate_latent_heat(temperature, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.R, const.m_mol)
    #density = density + sample_holder * const.density_sample_holder
    chi = np.full((const.n_z, const.n_y, const.n_x), 0.05, dtype=np.float64)
    Lambda, interface_temperatures = lambda_granular(const.n_x, const.n_y, const.n_z, temperature, Dr, dx, dz, dz, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.activation_energy_water_ice, const.R, r_mono_water, const.f_1, const.f_2, VFF, const.sigma, const.e_1, sample_holder, const.lambda_copper, r_n)
    heat_capacity = calculate_heat_capacity(temperature)
    diffusion_coefficient, p_sub = diffusion_parameters_sintering_periodic(const.n_x, const.n_y, const.n_z, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, temperature, const.molar_mass_water, const.R, VFF, r_mono_water, const.Phi_Guettler, pressure, const.m_H2O, const.k_boltzmann, dx, dy, dz, Dr, dyn_dt, sample_holder, blocked_voxels, var.n_x_lr, var.n_y_lr, var.n_z_lr)
    gas_density_previous = gas_density.copy()
    temperature_previous = temperature.copy()
    sub_gas_begin = sublimated_mass.copy()
    sub_gasdens_begin = gas_density.copy()
    temp_begin = temperature.copy()
    #print(sublimated_mass[0:const.n_z, 25, 25])
    #gas_density = np.maximum(gas_density, np.zeros((const.n_z,const.n_y, const.n_x), dtype=np.float64))
    #print(np.max(Lambda[0:const.n_z-3, 0:const.n_y, 0:const.n_x]))
    #print(np.nanmax(diffusion_coefficient))
    #print(np.max(temp_begin))
    for iterate in range(0, 15):
        S_c_hte, S_p_hte, S_c_de, S_p_de, empty_voxels = calculate_source_terms_sintering_diffusion(const.n_x, const.n_y, const.n_z, temperature, dx, dy, dz, dyn_dt, sample_holder, uniform_water_masses, latent_heat_water, sublimated_mass, gas_density)
        #S_c_hte = S_c_hte + S_c_light
        #print(np.max(S_c_hte))
        #print(np.isnan(temperature).any(), np.isnan(S_c_hte).any(), np.isnan(S_p_hte).any())
        temperature = hte_implicit_DGADI_zfirst(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, const.albedo, dyn_dt, lamp_power_dn, const.sigma, const.epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c_hte, S_p_hte, sample_holder, np.full(6, const.ambient_radiative_temperature, dtype=np.float64), reradiated_heat)
        #S_c_hte, S_p_hte, S_c_de, S_p_de, empty_voxels = calculate_source_terms_sintering_diffusion(const.n_x, const.n_y, const.n_z, temperature, dx, dy, dz, dyn_dt, sample_holder, uniform_water_masses, latent_heat_water, sublimated_mass, gas_density)
        gas_dens_previous = gas_density.copy()
        if const.dt > 1E-8:
            dyn_dt = 1E-8
            iteration_step = 0
            sublimated_mass_step = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
            gas_dens_last = gas_density.copy()
            for i in range(0, int(round(const.dt/1E-8, 2))):
                r_n, r_mono_water, sublimated_mass, areas = sinter_neck_calculation_time_dependent_diffusion_t(r_n, r_mono_water, dyn_dt, temperature, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.molecular_volume_H2O, const.surface_energy_par, const.R, const.r_mono, const.packing_geometry_factor, const.molar_mass_water, density, pressure, const.m_H2O, const.k_boltzmann, 0.001, water_particle_number, blocked_voxels, const.n_x, const.n_y, const.n_z, sample_holder, dx, dy, dz, gas_density, areas, surface, sinter_reduction_factor)
                S_c_hte, S_p_hte, S_c_de, S_p_de, empty_voxels = calculate_source_terms_sintering_diffusion(const.n_x, const.n_y, const.n_z, temperature, dx, dy, dz, dyn_dt, sample_holder, uniform_water_masses, latent_heat_water, sublimated_mass, gas_density)
                gas_density = de_implicit_DGADI_zfirst(const.n_x, const.n_y, const.n_z, surface_reduced, dyn_dt, gas_density, diffusion_coefficient, Dr, dx, dy, dz, surface, S_c_de, S_p_de, sh_adjacent_voxels, True, temperature, simulate_region)
                pressure = pressure_calculation_impulse(const.n_x, const.n_y, const.n_z, temperature, gas_density, const.k_boltzmann, const.m_H2O, var.VFF_pack, r_mono_water, dx, dy, dz, const.dt, sample_holder, sublimated_mass, water_particle_number, areas)
                iteration_step += 1
                if np.max(np.abs(gas_density - gas_dens_last)) <= np.max(gas_density[1:const.n_z-2, 2:const.n_y-2, 2:const.n_x-2]) * 1E-8 or i == int(round(const.dt/1E-8, 2))-1:
                    sublimated_mass_step += sublimated_mass
                    #gas_dens_out += gas_density[1]
                    break
                else:
                    if i % 1000:
                        print(np.max(np.abs(gas_density - gas_dens_last)))
                    if j == 73 and i % 1000:
                        print(gas_density[0:const.n_z, const.n_y // 2, const.n_x // 2])
                        print(sublimated_mass[0:const.n_z, const.n_y // 2, const.n_x // 2])
                        print(temperature[0:const.n_z, const.n_y // 2, const.n_x // 2])
                    sublimated_mass_step += sublimated_mass
                    if i >= 10000000:
                        break
                    gas_dens_last = gas_density.copy()
                    #gas_dens_out += gas_density[1]
                    #gas_density[1] = np.zeros((const.n_y, const.n_x), dtype=np.float64)
            sublimated_mass_step += sublimated_mass * (int(round(const.dt/1E-8)) - iteration_step)
            #gas_dens_out += gas_density[1] * (int(round(dt_arr[loops]/1E-9)) - iteration_step)
            dyn_dt = const.dt
            outgassing_rate[j] = (np.sum((gas_dens_previous - gas_density) * dx * dy * dz) + np.sum(sublimated_mass_step)) / const.dt
        else:
            #r_n, r_mono_water, sublimated_mass, areas, p_sub = Tsinter_neck_calculation_time_dependent_diffusion(r_n, r_mono_water, dyn_dt, temperature, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, r_mono_water, pressure, const.m_H2O, const.k_boltzmann, 0.001, water_particle_number, blocked_voxels, const.n_x, const.n_y, const.n_z, sample_holder, dx, dy, dz, gas_density)
            r_n, r_mono_water, sublimated_mass, areas = sinter_neck_calculation_time_dependent_diffusion(r_n, r_mono_water, dyn_dt, temperature, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.molecular_volume_H2O, const.surface_energy_par, const.R, const.r_mono, const.packing_geometry_factor, const.molar_mass_water, density, pressure, const.m_H2O, const.k_boltzmann, 0.001, water_particle_number, blocked_voxels, const.n_x, const.n_y, const.n_z, sample_holder, dx, dy, dz, gas_density, areas)
            S_c_hte, S_p_hte, S_c_de, S_p_de, empty_voxels = calculate_source_terms_sintering_diffusion(const.n_x, const.n_y, const.n_z, temperature, dx, dy, dz, dyn_dt, sample_holder, uniform_water_masses, latent_heat_water, sublimated_mass, gas_density)
            gas_density = de_implicit_DGADI_periodic(const.n_x, const.n_y, const.n_z, surface_reduced, dyn_dt, gas_density, diffusion_coefficient, Dr, dx, dy, dz, surface, S_c_de, S_p_de, sh_adjacent_voxels, False, temperature, simulate_region)
            sublimated_mass_step = sublimated_mass.copy()
            outgassing_rate[j] = (np.sum((gas_dens_previous - gas_density) * dx * dy * dz) + np.sum(sublimated_mass_step)) / const.dt
        #sublimated_mass = (gas_density - gas_density_previous) * dx * dy * dz
        sublimated_mass = sublimated_mass_step.copy()
        #print(sublimated_mass[1:3, 10:20, 10:20], 4)
        pressure = pressure_calculation_impulse(const.n_x, const.n_y, const.n_z, temperature, gas_density, const.k_boltzmann, const.m_H2O, var.VFF_pack, r_mono_water, dx, dy, dz, const.dt, sample_holder, sublimated_mass, water_particle_number, areas)
        #check_array(const.n_x, const.n_y, const.n_z, np.abs(temperature - temp_begin), 'greater', 1)
        #print(np.max(np.abs(temperature - temp_begin)))
        #print(sublimated_mass[0:const.n_z, 12, 12])
        #print(np.max(np.abs(temperature - temp_begin)))
        if len(empty_voxels) != 0:
            surface, surface_reduced = update_surface_arrays(empty_voxels, surface, surface_reduced, temperature, const.n_x, const.n_y, const.n_z, a, a_rad, b, b_rad, False)
        if np.max(np.abs(temperature - temp_begin)) < 5E-6:
            break
        sub_gas_begin = sublimated_mass.copy()
        sub_gasdens_begin = gas_density.copy()
        temp_begin = temperature.copy()
        if iterate < 29:
            gas_density = gas_density_previous.copy()
            temperature = temperature_previous.copy()
        else:
            print('Low Convergence Warning')
    #outgassing_rate[j] = np.sum(sublimated_mass)/const.dt
    outgassed_mass_complete += np.sum(sublimated_mass)
    uniform_water_masses = uniform_water_masses - sublimated_mass
    #print(np.sum(sublimated_mass))
    #surface, surface_reduced, temperature, r_n, r_mono_water = update_surface_arrays_periodic(empty_voxels, surface, surface_reduced, temperature, const.n_x, const.n_y, const.n_z, a, a_rad, b, b_rad, False, r_n, r_mono_water)
    passed_time += dyn_dt
    if np.isnan(temperature).any():
        print('NaN warning')
        break


data_dict = {'Temperature': temperature.tolist(), 'OR': outgassing_rate.tolist()}
with open('D:/TPM_Data/Noah/only_temps_volabs_sh_test/Outgassing_rate.json', 'w') as outfile:
    json.dump(data_dict, outfile)
'''