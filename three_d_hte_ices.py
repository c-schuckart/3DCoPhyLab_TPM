import numpy as np
import constants as const
import variables_and_arrays as var
import settings as sett
from tqdm import tqdm
import json
from os import listdir
from surface_detection import create_equidistant_mesh, DEBUG_print_3D_arrays, find_surface, surrounding_checker_moon, update_surface_arrays, get_sample_holder_adjacency, create_equidistant_mesh_2_layer
from thermal_parameter_functions import calculate_latent_heat, calculate_density, thermal_functions, calculate_bulk_density_and_VFF, thermal_conductivity_moon_regolith, heat_capacity_moon_regolith, calculate_water_grain_radius, calculate_heat_capacity, lambda_granular
from molecule_transfer import calculate_molecule_surface, diffusion_parameters_moon, calculate_source_terms, pressure_calculation, calculate_source_terms_linearised, calculate_molecule_flux_moon_test, sinter_neck_calculation_time_dependent, sintered_surface_checker
from heat_transfer_equation_DG_ADI import hte_implicit_DGADI, hte_implicit_DGADI_zfirst
from diffusion_equation_DG_ADI import de_implicit_DGADI, de_implicit_DGADI_zfirst
from boundary_conditions import sample_holder_data, day_night_cycle, calculate_L_chamber_lamp_bd
from ray_tracer import generate_topography, trace_rays_MC, get_temperature_vector

#work arrays and mesh creation + surface detection
#temperature, dx, dy, dz, Dr, a, a_rad, b, b_rad = create_equidistant_mesh(const.n_x, const.n_y, const.n_z, const.temperature_ini, const.min_dx, const.min_dy, const.min_dz, False)
temperature, dx, dy, dz, Dr, a, a_rad, b, b_rad = create_equidistant_mesh_2_layer(const.n_x, const.n_y, const.n_z, const.temperature_ini, const.min_dx, const.min_dy, const.min_dz, 21, 10)
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
'''time_deltas_data_interior, sample_holder_temp = read_temperature_data('D:/Laboratory_data/temps_ice.txt', '2023-03-04 17:52:03', '2023-03-04 22:00:02', [6], [])
max_k_2, sample_holder_temp = transform_temperature_data(const.k, const.dt, np.array(time_deltas_data_interior), [], sample_holder_temp)'''

#directory = 'D:/Laboratory_data/Sand_without_tubes/temp_profile/temp_profile/'
'''directory = 'D:/Masterarbeit_data/Sand_no_tubes/temp_profile/temp_profile/'
file_list = listdir(directory)
csv_file = open('D:/Masterarbeit_data/Sand_no_tubes/sand_temps(no_tubes).txt', 'r')
#csv_file = open('D:/Laboratory_data/Sand_without_tubes/sand_temps(no_tubes).txt', 'r')
surface_temperature_section, current_file, time_cur, current_surface_temp_scaled, next_segment_time = get_surface_temperatures_csv(const.n_x, const.n_y, directory, file_list, 5, 0, np.zeros(1, dtype=np.float64), const.dt, 0, True)
time_deltas_data_interior, current_index, next_segment_time_sh, sample_holder_temp = read_temperature_data_partial(csv_file, '2023-03-05 17:52:03', [6], True, 0, 0, [])
max_k_2, sample_holder_temp = transform_temperature_data_partial(int(time_deltas_data_interior.astype(int)/const.dt), const.dt, time_deltas_data_interior.astype(int), [], sample_holder_temp)

temperature_save = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
sensor_10mm = 0
sensor_20mm = 0
sensor_35mm = 0
sensor_55mm = 0
sensor_90mm = 0
#max_k, max_k_2 = const.k, const.k
#temperature_save = np.zeros((min(const.k, max_k, max_k_2)//sett.data_reduce + 1, const.n_z, const.n_y, const.n_x))
water_content_save = np.zeros((const.k//sett.data_reduce + 1, const.n_z, const.n_y, const.n_x), dtype=np.float64)
sublimated_mass_save = np.zeros((const.k//sett.data_reduce + 1, const.n_z, const.n_y, const.n_x), dtype=np.float64)
Max_Fourier_number = np.zeros(const.k, dtype=np.float64)
#surface_temp = np.full(const.k, 201, dtype=np.float64)
r_n = np.zeros((const.n_x, const.n_y, const.n_x), dtype=np.float64)
#sample_holder_temp = np.full(const.k, 140, dtype=np.float64)
previous_section_time = 0
previous_section_time_sh = 0'''
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

'''data_dict = {'SSurface': sh_adjacent_voxel.tolist(), 'Surface': surface.tolist(), 'RSurface': surface_reduced_diffusion.tolist(), 'SH': sample_holder.tolist(), 'SHD': sample_holder_diffusion.tolist()}
with open('test.json', 'w') as outfile:
    json.dump(data_dict, outfile)'''

'''for j in tqdm(range(0, const.k)):
    #if (j * const.dt) % 15 == 0:
        #print(np.sum(uniform_water_masses) + outgassed_mass_complete)
        #np.save('D:/TPM_Data/Luwex/only_temps_equilibriated/only_temperature_sim_' + str(j * const.dt) + '.npy', temperature)
        #np.save('D:/TPM_Data/Luwex/sublimation_and_diffusion_test/sublimation_and_diffusion' + str(j * const.dt) + '.npy', temperature)
        #np.save('D:/TPM_Data/Luwex/sublimation_and_diffusion_test/WATERsublimation_and_diffusion' + str(j * const.dt) + '.npy', uniform_water_masses)
        #np.save('D:/TPM_Data/Luwex/sublimation_and_diffusion_test/GASsublimation_and_diffusion' + str(j * const.dt) + '.npy', gas_density * dx * dy * dz)
    #temperature_previous = temperature[0:const.n_z, 0:const.n_y, 0:const.n_x]
    if np.isnan(temperature).any():
        print(temperature[2])
        break
    if np.greater(temperature, np.full((const.n_z, const.n_y, const.n_x), 400, dtype=np.float64)).any():
        print(temperature[2])
        print('Fluctuation')
        break
    density, VFF, water_ice_grain_density = calculate_bulk_density_and_VFF(temperature, VFF, uniform_dust_masses, uniform_water_masses, const.density_TUBS_M, dx, dy, dz)
    r_mono_water = calculate_water_grain_radius(const.n_x, const.n_y, const.n_z, uniform_water_masses, water_ice_grain_density, water_particle_number, r_mono_water)
    latent_heat_water = calculate_latent_heat(temperature, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.R, const.m_mol)
    density = density + sample_holder * const.density_sample_holder
    Lambda, interface_temperatures = thermal_conductivity_moon_regolith(const.n_x, const.n_y, const.n_z, temperature, dx, dy, dz, Dr, VFF, const.r_mono, const.fc1, const.fc2, const.fc3, const.fc4, const.fc5, const.mu, const.E, const.gamma, const.f1, const.f2, const.e1, const.chi_maria, const.sigma, const.epsilon, uniform_water_masses, uniform_dust_masses, const.lambda_water_ice, const.lambda_sample_holder, sample_holder)
    heat_capacity = heat_capacity_moon_regolith(const.n_x, const.n_y, const.n_z, temperature, const.c0, const.c1, const.c2, const.c3, const.c4, uniform_water_masses, uniform_dust_masses, const.heat_capacity_sample_holder, sample_holder)
    diffusion_coefficient, p_sub, sublimated_mass = diffusion_parameters_moon(const.n_x, const.n_y, const.n_z, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, temperature, interface_temperatures, const.m_mol[0], const.R, var.VFF_pack, const.r_mono, const.Phi_Guettler, const.tortuosity, pressure, const.m_H2O, const.k_boltzmann, dx, dy, dz, Dr, const.dt, sample_holder, sample_holder_diffusion, water_particle_number, r_mono_water)
    #print(sublimated_mass[0:10, 0:10, const.n_x//2])
    gas_density_previous[0:const.n_z, 0:const.n_y, 0:const.n_x] = gas_density[0:const.n_z, 0:const.n_y, 0:const.n_x]
    temperature_previous[0:const.n_z, 0:const.n_y, 0:const.n_x] = temperature[0:const.n_z, 0:const.n_y, 0:const.n_x]
    sub_gas_begin[0:const.n_z, 0:const.n_y, 0:const.n_x] = sublimated_mass[0:const.n_z, 0:const.n_y, 0:const.n_x]
    sub_gasdens_begin[0:const.n_z, 0:const.n_y, 0:const.n_x] = gas_density[0:const.n_z, 0:const.n_y, 0:const.n_x]
    temp_begin = temperature[0:const.n_z, 0:const.n_y, 0:const.n_x] = temperature[0:const.n_z, 0:const.n_y, 0:const.n_x]
    #print(sublimated_mass[1:3, 10:20, 10:20], 1)
    for i in range(0, 30):
        #Q_c_hte, Q_p_hte, Q_c_de, Q_p_de = calculate_source_terms(const.n_x,const.n_y, const.n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, const.dt, surface_reduced, uniform_water_masses, latent_heat_water, surface)
        Q_c_hte, Q_p_hte, Q_c_de, Q_p_de = calculate_source_terms_linearised(const.n_x, const.n_y, const.n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, Dr, const.dt, surface_reduced, uniform_water_masses, latent_heat_water, surface, const.m_H2O, const.k_boltzmann, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, sample_holder, water_particle_number, r_mono_water, diffusion_coefficient, False)
        #print((Q_c_de * dx * dy * dz * const.dt)[1:3, 10:20, 10:20], 2)
        #print(gas_density[1:3, 10:20, 10:20], 3)
        #temperature = hte_implicit_DGADI(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, const.albedo, const.dt, lamp_power, const.sigma, const.epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, Q_c_hte, Q_p_hte, sample_holder, const.ambient_radiative_temperature)
        temperature = hte_implicit_DGADI_zfirst(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, const.albedo, const.dt, lamp_power, const.sigma, const.epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, Q_c_hte, Q_p_hte, sample_holder, const.ambient_radiative_temperature)
        diffusion_coefficient, p_sub, sublimated_mass = diffusion_parameters_moon(const.n_x, const.n_y, const.n_z, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, temperature, interface_temperatures, const.m_mol[0], const.R, var.VFF_pack, const.r_mono, const.Phi_Guettler, const.tortuosity, pressure, const.m_H2O, const.k_boltzmann, dx, dy, dz, Dr, const.dt, sample_holder, sample_holder_diffusion, water_particle_number, r_mono_water)
        Q_c_hte, Q_p_hte, Q_c_de, Q_p_de = calculate_source_terms_linearised(const.n_x, const.n_y, const.n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, Dr, const.dt, surface_reduced, uniform_water_masses, latent_heat_water, surface, const.m_H2O, const.k_boltzmann, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, sample_holder, water_particle_number, r_mono_water, diffusion_coefficient, False)
        #gas_density = de_implicit_DGADI(const.n_x, const.n_y, const.n_z, surface_reduced_diffusion, const.dt, gas_density, diffusion_coefficient, Dr, dx, dy, dz, surface, Q_c_de, Q_p_de, sh_adjacent_voxel, temperature, False, surrounding_surface, j)
        gas_density = de_implicit_DGADI_zfirst(const.n_x, const.n_y, const.n_z, surface_reduced_diffusion, const.dt, gas_density, diffusion_coefficient, Dr, dx, dy, dz, surface, Q_c_de, Q_p_de, sh_adjacent_voxel, temperature, False, surrounding_surface, j)
        #gas_density = np.maximum(np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64), gas_density)
        sublimated_mass = (gas_density - gas_density_previous) * dx * dy * dz
        #print(np.max(sublimated_mass))
        #print(sublimated_mass[1:3, 10:20, 10:20], 4)
        if np.abs(np.sum(sub_gas_begin) - np.sum(sublimated_mass)) > 1E-20:
            print(np.abs(np.sum(sub_gas_begin) - np.sum(sublimated_mass)))
            print('Mass conservation warning')
        pressure = pressure_calculation(const.n_x, const.n_y, const.n_z, temperature, gas_density, const.k_boltzmann, const.m_H2O, var.VFF_pack, const.r_mono, dx, dy, dz, const.dt, sample_holder, sublimated_mass, water_particle_number)
        if np.max(np.abs(temperature - temp_begin)) < 1E-7:
            break
        sub_gas_begin[0:const.n_z, 0:const.n_y, 0:const.n_x] = sublimated_mass[0:const.n_z, 0:const.n_y, 0:const.n_x]
        sub_gasdens_begin[0:const.n_z, 0:const.n_y, 0:const.n_x] = gas_density[0:const.n_z, 0:const.n_y, 0:const.n_x]
        temp_begin[0:const.n_z, 0:const.n_y, 0:const.n_x] = temperature[0:const.n_z, 0:const.n_y, 0:const.n_x]
        if i < 29:
            gas_density[0:const.n_z, 0:const.n_y, 0:const.n_x] = gas_density_previous[0:const.n_z, 0:const.n_y, 0:const.n_x]
            temperature[0:const.n_z, 0:const.n_y, 0:const.n_x] = temperature_previous[0:const.n_z, 0:const.n_y, 0:const.n_x]
        else:
            print('Low Convergence Warning')
        #break
    #S_c, sublimated_mass, outgassed_mass_timestep = calculate_molecule_flux_moon(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_H2O, dx, dy, dz, const.dt, const.k_boltzmann, sample_holder, uniform_water_masses, latent_heat_water)
    #temperature = hte_implicit_DGADI(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, const.albedo, const.dt, lamp_power, const.sigma, const.epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder)
    #outgassed_mass_complete += outgassed_mass_timestep
    #print(diffusion_coefficient[2])
    #print(np.min(gas_density))
    outgassing_rate[j] = np.sum(sublimated_mass * mesh_shape_negative)/const.dt
    max_temp[j] = np.max(temperature[2])
    outgassed_mass_complete += np.sum(sublimated_mass * mesh_shape_negative)
    sublimated_mass = sublimated_mass * mesh_shape_positive
    gas_density = gas_density * mesh_shape_positive
    uniform_water_masses = uniform_water_masses - sublimated_mass
    #print(np.sum(sublimated_mass))
    #if np.max(np.abs(temperature - temperature_previous)) < 50E-6:
        #print('Equilibrated :3')
        #break
    if np.min(gas_density) < 0:
        print('We need roads :C')
        break
    #uniform_water_masses_implicit = update_thermal_arrays(const.n_x, const.n_y, const.n_z, temperature, uniform_water_masses_implicit, delta_T, Energy_Increase_per_Layer, sublimated_mass_implicit, resublimated_mass, const.dt, const.avogadro_constant, const.molar_mass_water, const.molar_mass_co2, heat_capacity, const.heat_capacity_water_ice, const.heat_capacity_co2_ice, EIis_0, Latent_Heat_per_Layer, E_Lat_0, E_Rad, E_In, E_sh, E_source_sink)[1]
    #print(sublimated_mass_implicit[1][12][12], temperature_implicit[1][12][12])'
    #if j % sett.data_reduce == 0 or j == 0:
        #data_save_sensors(j * const.dt, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, file)
        #temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save = data_store(j, temperature, water_content_per_layer, co2_content_per_layer, outgassed_molecules_per_time_step/const.dt, outgassed_molecules_per_time_step_co2/const.dt, temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save, sett.data_reduction)
        #sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, temperature_save = data_store_sensors(j, const.n_x, const.n_y, const.n_z, temperature, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, sett.data_reduce, temperature_save)
        #temperature_save[j//sett.data_reduce] = temperature
        #water_content_save[j // sett.data_reduce] = uniform_water_masses
        #sublimated_mass_save[j // sett.data_reduce] = sublimated_mass'''

r_mono_base = r_mono_water.copy()
r_n_base = r_n.copy()
max_temps = np.zeros(const.k, dtype=np.float64)
sublimated_mass_mid = np.zeros(const.k, dtype=np.float64)
surface_topography_polygons = np.empty(0)

for j in tqdm(range(0, const.k)):
    if j % 100 == 0:
        print(temperature[0:const.n_z, const.n_y // 2, const.n_x // 2])
        print(lamp_power_dn[1][26][26])
        #s_1 = sinter_neck_calculation_time_dependent(r_n, r_mono_water, const.dt, temperature, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.molecular_volume_H2O, const.surface_energy_par, const.R, const.r_mono, const.packing_geometry_factor, const.molar_mass_water, density, pressure, const.m_H2O, const.k_boltzmann, 0.001, water_particle_number, blocked_voxels, const.n_x, const.n_y, const.n_z, sample_holder, dx, dy, dz, surface)[2]
        #print(s_1[0:5, const.n_y//2, const.n_x//2])
        #s_2 = calculate_molecule_surface(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_H2O, dx, dy, dz, const.dt, surface_reduced, const.k_boltzmann, uniform_water_masses)[0]
        #print(s_2[0:5, const.n_y//2, const.n_x//2])
        if j != 0:
            print(Lambda[0:5, const.n_y//2, const.n_x//2])
            print(uniform_water_masses[0:5, const.n_y//2, const.n_x//2])
            print(np.sum(sublimated_mass_mid[j-100:j]))
        #print(np.sum(uniform_water_masses) + outgassed_mass_complete)
        #np.save('D:/TPM_Data/Luwex/only_temps_equilibriated/only_temperature_sim_' + str(j * const.dt) + '.npy', temperature)
        #np.save('D:/TPM_Data/Luwex/sublimation_test/sublimation_test' + str(j * const.dt) + '.npy', temperature)
        #np.save('D:/TPM_Data/Luwex/sublimation_test/WATERsublimation_test' + str(j * const.dt) + '.npy', uniform_water_masses)
        #np.save('D:/TPM_Data/Luwex/sublimation_and_diffusion/GASsublimation_and_diffusion' + str(j * const.dt) + '.npy', gas_density * dx * dy * dz)
    #temperature_previous = temperature[0:const.n_z, 0:const.n_y, 0:const.n_x]
    if sett.enable_ray_tracing and len(surface_topography_polygons) != 0:
        surface_temperatur_vector = get_temperature_vector(temperature, surface, surface_reduced, len(surface_topography_polygons))
    lamp_power_dn, S_c_deeper = day_night_cycle(lamp_power, S_c_deeper, 3 * 3600, j * const.dt)
    #density, VFF, water_ice_grain_density = calculate_bulk_density_and_VFF(temperature, VFF, uniform_dust_masses, uniform_water_masses, const.density_TUBS_M, dx, dy, dz)
    density = calculate_density(temperature, VFF)[1]
    r_n, r_mono_water, sublimated_mass, areas = sinter_neck_calculation_time_dependent(r_n, r_mono_water, const.dt, temperature, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.molecular_volume_H2O, const.surface_energy_par, const.R, const.r_mono, const.packing_geometry_factor, const.molar_mass_water, density, pressure, const.m_H2O, const.k_boltzmann, 0.001, water_particle_number, blocked_voxels, const.n_x, const.n_y, const.n_z, sample_holder, dx, dy, dz, surface)
    #sublimated_mass, empty_voxels = calculate_molecule_surface(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_H2O, dx, dy, dz, const.dt, surface_reduced, const.k_boltzmann, uniform_water_masses)[0:2]
    blocked_voxels = sintered_surface_checker(const.n_x, const.n_y, const.n_z, r_n, r_mono_water)
    #r_mono_water = calculate_water_grain_radius(const.n_x, const.n_y, const.n_z, uniform_water_masses, water_ice_grain_density, water_particle_number, r_mono_water)
    latent_heat_water = calculate_latent_heat(temperature, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.R, const.m_mol)
    density = density + sample_holder * (const.density_copper - density[const.n_z-2, const.n_y//2, const.n_x//2])
    #Lambda, interface_temperatures = thermal_conductivity_moon_regolith(const.n_x, const.n_y, const.n_z, temperature, dx, dy, dz, Dr, VFF, const.r_mono, const.fc1, const.fc2, const.fc3, const.fc4, const.fc5, const.mu, const.E, const.gamma, const.f1, const.f2, const.e1, const.chi_maria, const.sigma, const.epsilon, uniform_water_masses, uniform_dust_masses, const.lambda_water_ice, const.lambda_sample_holder, sample_holder)
    Lambda, interface_temperatures = lambda_granular(const.n_x, const.n_y, const.n_z, temperature, Dr, dx, dz, dz, const.lambda_water_ice, const.poisson_ratio_par, const.young_modulus_par, const.activation_energy_water_ice, const.R, r_mono_water, const.f_1, const.f_2, VFF, const.sigma, const.e_1, sample_holder, const.lambda_copper, r_n)
    #heat_capacity = heat_capacity_moon_regolith(const.n_x, const.n_y, const.n_z, temperature, const.c0, const.c1, const.c2, const.c3, const.c4, uniform_water_masses, uniform_dust_masses, const.heat_capacity_sample_holder, sample_holder)
    heat_capacity = calculate_heat_capacity(temperature)
    S_c, S_p, Scde, Spde, empty_voxels = calculate_source_terms(const.n_x, const.n_y, const.n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, const.dt, surface_reduced, uniform_water_masses, latent_heat_water, surface)
    #S_c, S_p = calculate_source_terms(const.n_x, const.n_y, const.n_z, temperature, gas_density, pressure, sublimated_mass, dx, dy, dz, const.dt, surface_reduced, uniform_water_masses, latent_heat_water, surface)[0:2]
    #S_c, S_p, sublimated_mass, outgassed_mass_timestep = calculate_molecule_flux_moon_test(const.n_x, const.n_y, const.n_z, temperature, pressure, const.lh_a_1, const.lh_b_1, const.lh_c_1, const.lh_d_1, const.m_H2O, dx, dy, dz, const.dt, const.k_boltzmann, sample_holder, uniform_water_masses, latent_heat_water, water_particle_number, r_mono_water)
    temperature = hte_implicit_DGADI_zfirst(const.n_x, const.n_y, const.n_z, surface_reduced, const.r_H, const.albedo, const.dt, lamp_power_dn, const.sigma, const.epsilon, temperature, Lambda, Dr, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder, np.full(6, const.ambient_radiative_temperature, dtype=np.float64))
    outgassed_mass_complete += np.sum(sublimated_mass)
    uniform_water_masses = uniform_water_masses - sublimated_mass
    outgassing_rate[j] = np.sum(sublimated_mass)/const.dt
    #print(temperature[1, const.n_y // 2, const.n_x // 2], uniform_water_masses[1, const.n_y // 2, const.n_x // 2], sublimated_mass[1, const.n_y // 2, const.n_x // 2], empty_voxels, density[1, const.n_y // 2, const.n_x // 2], VFF[1, const.n_y // 2, const.n_x // 2])
    if len(empty_voxels) != 0:
        surface, surface_reduced = update_surface_arrays(empty_voxels, surface, surface_reduced, temperature, const.n_x, const.n_y, const.n_z, a, a_rad, b, b_rad, False)
        if sett.enable_ray_tracing:
            surface_topography_polygons = generate_topography(surface, surface_reduced, dx, dy, dz)
            print('Tracing rays')
            view_factor_matrix = trace_rays_MC(surface_topography_polygons, 4000)
            print('Finished')
    max_temps[j] = np.max(temperature)
    sublimated_mass_mid[j] = np.sum(sublimated_mass[0:const.n_z, const.n_y//2, const.n_x//2])
    #print(np.sum(sublimated_mass))
    #if np.max(np.abs(temperature - temperature_previous)) < 50E-6:
        #break


#Data saving and output
#save_current_arrays(temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, highest_pressure, highest_pressure_co2, ejection_times, var.time_passed + const.dt * const.k)
'''data_dict = {'Temperature': temperature.tolist(), 'OR': outgassing_rate.tolist()}
with open('test_or_nodiff.json', 'w') as outfile:
    json.dump(data_dict, outfile)'''

#data_save_sensors(const.k * const.dt, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, file)
#file.close()
'''data_dict = {'Temperature': temperature_save.tolist(), 'Surface': surface.tolist(), 'RSurface': surface_reduced.tolist(), 'HC': Lambda.tolist(), 'SH': sample_holder.tolist()}
with open('test_ec.json', 'w') as outfile:
    json.dump(data_dict, outfile)
print(np.max(Max_Fourier_number))
print('done')'''




