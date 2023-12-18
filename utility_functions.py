import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
import csv
import pandas as pd
from os import path


@njit
def radial_x_mixing(n_x, n_y, n_z, temperature, water_mass_per_layer, outer_voxels, height):
    temp_line_storage_temps = np.zeros(n_z-2-height, dtype=np.float64)
    temp_line_storage_mass = np.zeros(n_z-2-height, dtype=np.float64)
    if n_x % 2 == 0:
        for each in outer_voxels:
            temp_line_storage_temps = temperature[height:n_z-1, each[1], n_x//2-1].copy()
            temp_line_storage_mass = water_mass_per_layer[height:n_z-1, each[1], n_x//2-1].copy()
            for a in range(n_x//2-1, each[0], -1):
                temperature[height:n_z-1, each[1], a] = temperature[height:n_z-1, each[1], a-1]
                water_mass_per_layer[height:n_z-1, each[1], a] = water_mass_per_layer[height:n_z-1, each[1], a+1]
            temperature[height:n_z-1, each[1], each[0]] = temp_line_storage_temps.copy()
            water_mass_per_layer[height:n_z-1, each[1], each[0]] = temp_line_storage_mass.copy()
            temp_line_storage_temps = temperature[height:n_z - 1, each[1], n_x//2].copy()
            temp_line_storage_mass = water_mass_per_layer[height:n_z - 1, each[1],  n_x//2].copy()
            for a in range(n_x//2, n_x-1-each[0]):
                temperature[height:n_z - 1, each[1], a] = temperature[height:n_z - 1, each[1], a + 1]
                water_mass_per_layer[height:n_z - 1, each[1], a] = water_mass_per_layer[height:n_z - 1, each[1], a + 1]
            temperature[height:n_z - 1, each[1], n_x-1-each[0]] = temp_line_storage_temps.copy()
            water_mass_per_layer[height:n_z - 1, each[1], n_x-1-each[0]] = temp_line_storage_mass.copy()
        return temperature, water_mass_per_layer
    else: #n_x % 2 == 1
        for each in outer_voxels:
            temp_line_storage_temps = temperature[height:n_z-1, each[1], n_x//2].copy()
            temp_line_storage_mass = water_mass_per_layer[height:n_z-1, each[1], n_x//2].copy()
            for a in range(n_x//2, each[0], -1):
                temperature[height:n_z-1, each[1], a] = temperature[height:n_z-1, each[1], a-1]
                water_mass_per_layer[height:n_z-1, each[1], a] = water_mass_per_layer[height:n_z-1, each[1], a-1]
            temperature[height:n_z-1, each[1], each[0]] = temp_line_storage_temps.copy()
            water_mass_per_layer[height:n_z-1, each[1], each[0]] = temp_line_storage_mass.copy()
            temp_line_storage_temps = temperature[height:n_z - 1, each[1], n_x//2+1].copy()
            temp_line_storage_mass = water_mass_per_layer[height:n_z - 1, each[1],  n_x//2+1].copy()
            for a in range(n_x//2+1, n_x-1-each[0]):
                temperature[height:n_z - 1, each[1], a] = temperature[height:n_z - 1, each[1], a + 1]
                water_mass_per_layer[height:n_z - 1, each[1], a] = water_mass_per_layer[height:n_z - 1, each[1], a + 1]
            temperature[height:n_z - 1, each[1], n_x-1-each[0]] = temp_line_storage_temps.copy()
            water_mass_per_layer[height:n_z - 1, each[1], n_x-1-each[0]] = temp_line_storage_mass.copy()
        return temperature, water_mass_per_layer


@njit
def radial_y_mixing(n_x, n_y, n_z, temperature, water_mass_per_layer, outer_voxels, height):
    temp_line_storage_temps = np.zeros(n_z-2-height, dtype=np.float64)
    temp_line_storage_mass = np.zeros(n_z-2-height, dtype=np.float64)
    if n_y % 2 == 0:
        for each in outer_voxels:
            temp_line_storage_temps = temperature[height:n_z-1, n_y//2-1, each[0]].copy()
            temp_line_storage_mass = water_mass_per_layer[height:n_z-1, n_y//2-1, each[0]].copy()
            for a in range(n_y//2-1, each[1], -1):
                temperature[height:n_z-1, a, each[0]] = temperature[height:n_z-1, a-1, each[0]]
                water_mass_per_layer[height:n_z-1, a, each[0]] = water_mass_per_layer[height:n_z-1, a-1, each[0]]
            temperature[height:n_z-1, each[1], each[0]] = temp_line_storage_temps.copy()
            water_mass_per_layer[height:n_z-1, each[1], each[0]] = temp_line_storage_mass.copy()
            temp_line_storage_temps = temperature[height:n_z - 1, n_y//2, each[0]].copy()
            temp_line_storage_mass = water_mass_per_layer[height:n_z - 1, n_y//2, each[0]].copy()
            for a in range(n_y//2, n_y-1-each[1]):
                temperature[height:n_z - 1, a, each[0]] = temperature[height:n_z - 1, a+1, each[0]]
                water_mass_per_layer[height:n_z - 1, a, each[0]] = water_mass_per_layer[height:n_z - 1, a+1, each[0]]
            temperature[height:n_z - 1, n_y-1-each[1], each[0]] = temp_line_storage_temps.copy()
            water_mass_per_layer[height:n_z - 1, n_y-1-each[1], each[0]] = temp_line_storage_mass.copy()
        return temperature, water_mass_per_layer
    else: #n_y % 2 == 1
        for each in outer_voxels:
            temp_line_storage_temps = temperature[height:n_z-1, each[1], each[0]].copy()
            temp_line_storage_mass = water_mass_per_layer[height:n_z-1, each[1], each[0]].copy()
            for a in range(each[1], n_y//2):
                temperature[height:n_z - 1, a, each[0]] = temperature[height:n_z - 1, a + 1, each[0]]
                water_mass_per_layer[height:n_z - 1, a, each[0]] = water_mass_per_layer[height:n_z - 1, a + 1, each[0]]
            temperature[height:n_z-1, n_y//2, each[0]] = temp_line_storage_temps.copy()
            water_mass_per_layer[height:n_z-1, n_y//2, each[0]] = temp_line_storage_mass.copy()
            temp_line_storage_temps = temperature[height:n_z - 1, n_y-1-each[1], each[0]].copy()
            temp_line_storage_mass = water_mass_per_layer[height:n_z - 1, n_y-1-each[1], each[0]].copy()
            for a in range(n_y-1-each[1], n_y//2+1, -1):
                temperature[height:n_z - 1, a, each[0]] = temperature[height:n_z - 1, a-1, each[0]]
                water_mass_per_layer[height:n_z - 1, a, each[0]] = water_mass_per_layer[height:n_z - 1, a-1, each[0]]
            temperature[height:n_z - 1, n_y//2+1, each[0]] = temp_line_storage_temps.copy()
            water_mass_per_layer[height:n_z - 1, n_y//2+1, each[0]] = temp_line_storage_mass.copy()
        return temperature, water_mass_per_layer

@njit
def radial_mixing(n_x, n_y, n_z, outer_voxels_x, outer_voxels_y, temperature, water_mass_per_layer, sample_holder, height):
    if np.max(outer_voxels_x) == 0 or np.max(outer_voxels_y) == 0:
        ov_counter_x = 0
        ov_counter_y = 0
        for j in range(0, n_y):
            for k in range(0, n_x):
                if sample_holder[height][j][k] == 0 and temperature[height][j][k] > 0:
                    outer_voxels_x[ov_counter_x] = np.array([k, j, height], dtype=np.int32)
                    ov_counter_x += 1
                    break
                else:
                    pass
        for k in range(0, n_x):
            for j in range(0, n_y):
                if sample_holder[height][j][k] == 0 and temperature[height][j][k] > 0:
                    outer_voxels_y[ov_counter_y] = np.array([k, j, height], dtype=np.int32)
                    ov_counter_y += 1
                    break
                else:
                    pass
        outer_voxels_x = outer_voxels_x[0:ov_counter_x]
        outer_voxels_y = outer_voxels_y[0:ov_counter_y]
    temperature, water_mass_per_layer = radial_x_mixing(n_x, n_y, n_z, temperature, water_mass_per_layer, outer_voxels_x, height)
    temperature, water_mass_per_layer = radial_y_mixing(n_x, n_y, n_z, temperature, water_mass_per_layer, outer_voxels_y, height)
    return temperature, water_mass_per_layer, outer_voxels_x, outer_voxels_y

@njit
def get_mixing_stats(n_x, n_y, temperature, sample_holder, target_temp, height, count_array):
    for j in range(0, n_y):
        for k in range(0, n_x):
            if temperature[height][j][k] == target_temp and sample_holder[height][j][k] == 0:
                count_array[j][k] += 1
    return count_array


@njit
def thermal_reservoir(n_x, n_y, surface_height, temperature, reservoir_temp, sample_holder):
    for j in range(0, n_y):
        for k in range(0, n_x):
            if temperature[surface_height][j][k] > 0 and sample_holder[surface_height][j][k] == 0:
                if temperature[surface_height][j][k] < reservoir_temp:
                    temperature[surface_height][j][k] = reservoir_temp
    return temperature


def sort_csv(path, sort_avrg, outpath):
    csvdf = pd.read_csv(path, names=['Name', 'S1_avrg', 'S1_max', 'S2_avrg', 'S2_max', 'S3_avrg', 'S3_max', 'S4_avrg', 'S4_max', 'S5_avrg', 'S5_max'])
    csvdf_mirror = csvdf.copy()
    s1 = csvdf['Name']
    s2 = csvdf['S1_avrg'] + csvdf['S2_avrg'] + csvdf['S3_avrg'] + csvdf['S4_avrg'] + csvdf['S5_avrg']
    s3 = csvdf['S1_max'] + csvdf['S2_max'] + csvdf['S3_max'] + csvdf['S4_max'] + csvdf['S5_max']
    content_arr = pd.DataFrame({'A': s1, 'B': s2, 'C': s3})
    if sort_avrg:
        content_arr.sort_values(by='B', inplace=True)
    else:
        content_arr.sort_values(by='C', inplace=True)
    #print(content_arr)
    #print(csvdf_mirror)
    for i in range(len(csvdf)):
        #print(csvdf.iloc[content_arr.iloc[[i]].index[0]][1])
        csvdf_mirror.iloc[i] = csvdf.iloc[content_arr.iloc[[i]].index[0]][0], csvdf.iloc[content_arr.iloc[[i]].index[0]][1], csvdf.iloc[content_arr.iloc[[i]].index[0]][2], csvdf.iloc[content_arr.iloc[[i]].index[0]][3], csvdf.iloc[content_arr.iloc[[i]].index[0]][4], csvdf.iloc[content_arr.iloc[[i]].index[0]][5], csvdf.iloc[content_arr.iloc[[i]].index[0]][6], csvdf.iloc[content_arr.iloc[[i]].index[0]][7], csvdf.iloc[content_arr.iloc[[i]].index[0]][8], csvdf.iloc[content_arr.iloc[[i]].index[0]][9], csvdf.iloc[content_arr.iloc[[i]].index[0]][10]
    #print(csvdf_mirror)
    csvdf_mirror.to_csv(outpath)


def auto_path(path_string):
    seperator = 0
    path_components = []
    out_path = ''
    for i in range(len(path_string)):
        if path_string[i] == r"\ "[0]:
            path_components.append(path_string[seperator:i])
            seperator = i+1
        if i == len(path_string)-1:
            path_components.append(path_string[seperator:i+1])
    for i in range(len(path_components)):
        if i == 0:
            out_path = path_components[0]
        else:
            out_path = path.join(out_path, path_components[i])
    return out_path.replace("\\", "/")


def randomize(n_x_start, n_y_start, n_z_start, n_x_end, n_y_end, n_z_end, temperture, probability, distribution):
    if distribution == 'uniform':
        generator = np.random.default_rng()
    else:
        print('Distribution type not implemented, defaulting to uniform')
        generator = np.random.default_rng()
    for i in range(n_z_start, n_z_end):
        for j in range(n_y_start, n_y_end):
            for k in range(n_x_start, n_x_end):
                if generator.random() > probability: #Inverse probability here. So p=7/10 equals <7/10> gridpoints filled.
                    temperture[i][j][k] = 0
    return temperture


def check_connections(n_x, n_y, n_z, temperature):
    cluster_array = np.full(np.shape(temperature), 1E7, dtype=np.int32)
    cluster_nr = 1
    cluster_dict = {}
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] > 0 and (temperature[i+1][j][k] > 0 or temperature[i-1][j][k] > 0 or temperature[i][j+1][k] > 0 or temperature[i][j-1][k] > 0 or temperature[i][j][k+1] > 0 or temperature[i][j][k-1] > 0):
                    nr_arr = np.array([cluster_array[i+1][j][k], cluster_array[i-1][j][k], cluster_array[i][j+1][k], cluster_array[i][j-1][k], cluster_array[i][j][k+1], cluster_array[i][j][k-1]], dtype=np.int32)
                    if np.min(nr_arr) == 1E7:
                        cluster_array[i][j][k] = cluster_nr
                        cluster_dict[str(cluster_nr)] = [np.array([k, j, i], dtype=np.int32)]
                        cluster_nr += 1
                    elif np.min(nr_arr) == cluster_nr:
                        cluster_array[i][j][k] = cluster_nr
                        cluster_dict[str(cluster_nr)].append(np.array([k, j, i], dtype=np.int32))
                    else:
                        cluster_array[i][j][k] = np.min(nr_arr)
                        cluster_dict[str(cluster_array[i][j][k])].append(np.array([k, j, i], dtype=np.int32))
                    if cluster_array[i+1][j][k] > cluster_array[i][j][k] and cluster_array[i+1][j][k] < 1E7:
                        to_replace = cluster_array[i+1][j][k]
                        for each in cluster_dict[str(cluster_array[i+1][j][k])]:
                            cluster_dict[str(cluster_array[i][j][k])].append(each)
                            cluster_array[each[2]][each[1]][each[0]] = cluster_array[i][j][k]
                        cluster_dict.pop(str(to_replace))
                    if cluster_array[i-1][j][k] > cluster_array[i][j][k] and cluster_array[i-1][j][k] < 1E7:
                        to_replace = cluster_array[i-1][j][k]
                        for each in cluster_dict[str(cluster_array[i-1][j][k])]:
                            cluster_dict[str(cluster_array[i][j][k])].append(each)
                            cluster_array[each[2]][each[1]][each[0]] = cluster_array[i][j][k]
                        cluster_dict.pop(str(to_replace))
                    if cluster_array[i][j+1][k] > cluster_array[i][j][k] and cluster_array[i][j+1][k] < 1E7:
                        to_replace = cluster_array[i][j+1][k]
                        for each in cluster_dict[str(cluster_array[i][j+1][k])]:
                            cluster_dict[str(cluster_array[i][j][k])].append(each)
                            cluster_array[each[2]][each[1]][each[0]] = cluster_array[i][j][k]
                        cluster_dict.pop(str(to_replace))
                    if cluster_array[i][j-1][k] > cluster_array[i][j][k] and cluster_array[i][j-1][k] < 1E7:
                        to_replace = cluster_array[i][j-1][k]
                        for each in cluster_dict[str(cluster_array[i][j-1][k])]:
                            cluster_dict[str(cluster_array[i][j][k])].append(each)
                            cluster_array[each[2]][each[1]][each[0]] = cluster_array[i][j][k]
                        cluster_dict.pop(str(to_replace))
                    if cluster_array[i][j][k+1] > cluster_array[i][j][k] and cluster_array[i][j][k+1] < 1E7:
                        to_replace = cluster_array[i][j][k+1]
                        for each in cluster_dict[str(cluster_array[i][j][k+1])]:
                            cluster_dict[str(cluster_array[i][j][k])].append(each)
                            cluster_array[each[2]][each[1]][each[0]] = cluster_array[i][j][k]
                        cluster_dict.pop(str(to_replace))
                    if cluster_array[i][j][k-1] > cluster_array[i][j][k] and cluster_array[i][j][k-1] < 1E7:
                        to_replace = cluster_array[i][j][k-1]
                        for each in cluster_dict[str(cluster_array[i][j][k-1])]:
                            cluster_dict[str(cluster_array[i][j][k])].append(each)
                            cluster_array[each[2]][each[1]][each[0]] = cluster_array[i][j][k]
                        cluster_dict.pop(str(to_replace))
    if len(cluster_dict) == 1:
        print('Array consists of a singular cluster')
    else:
        print('Array has non connected clusters numbering: ' + str(len(cluster_dict)))
    return cluster_array

'''temp_slice = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
temperature = np.zeros((7, 7, 7), dtype=np.float64)
for i in range(0, 7):
    temperature[0:7, 0:7, i] = temp_slice
plt.imshow(temperature[0:7, 0:7, 1])
plt.show()
ca = check_connections(7, 7, 7, temperature)
plt.imshow(ca[0:7, 0:7, 1])
plt.show()'''

@njit
def test(n_x, n_y, n_z, temperature, uniform_water_masses, uniform_dust_masses, dust_ice_ratio_global, dx, dy, dz):
    for i in range(0, n_z):
        for j in range(0, n_y):
            for k in range(0, n_x):
                if temperature[i][j][k] == 0:  # or sample_holder[i][j][k] == 1:
                    uniform_water_masses[i][j][k] = 0
                    uniform_dust_masses[i][j][k] = 0
                if temperature[i][j][k] > 0 and i < 200:
                    uniform_water_masses = (1 / 8 * 4 / 3 * np.pi) * dx * dy * dz * (1 / (dust_ice_ratio_global + 1))
    return uniform_water_masses, uniform_dust_masses


@njit
def check_array(n_x, n_y, n_z, target_array, instruction, target_nr):
    for i in range(0, n_z):
        for j in range(0, n_y):
            for k in range(0, n_x):
                if instruction == 'NaN' and np.isnan(target_array[i][j][k]):
                    print(k, j, i)
                elif instruction == 'number' and target_array[i][j][k] == target_nr:
                    print(k, j, i)
                elif instruction == 'greater' and target_array[i][j][k] > target_nr:
                    print(k, j, i)
                elif instruction == 'lesser' and target_array[i][j][k] < target_nr:
                    print(k, j, i)



'''
The sensors are in n_y//2 to n_y for the right sensors and 0 to n_x//2 for the rear sensors
'''
@njit
def save_sensors_L_sample_holder(n_x, n_y, n_z, temperature, sensors_right, sensors_rear, time):
    sensors_rear[time][0] = (temperature[2][n_y//2-2][n_x//2-15] * 3/5 + temperature[2][n_y//2-2][n_x//2-16] * 2/5)       # rear 5mm
    sensors_rear[time][1] = ((temperature[3][n_y//2] [n_x//2-13] * 1/5 + temperature[3][n_y//2+1] [n_x//2-13] * 4/5) * 4/5 + (temperature[3][n_y//2] [n_x//2-12] * 1/5 + temperature[3][n_y//2+1] [n_x//2-12] * 4/5) * 1/5)      # rear 10mm
    sensors_rear[time][2] = (temperature[3][n_y//2-12] [n_x//2-13] * 4/5 + temperature[3][n_y//2-12] [n_x//2-12] * 1/5)       # rear 10mm side
    sensors_rear[time][3] = temperature[4][n_y//2-1][n_x//2-17]        # rear 15mm
    sensors_rear[time][4] = (temperature[5][n_y//2+3][n_x//2-15] * 1/5 + temperature[5][n_y//2+3][n_x//2-14] * 4/5)      # rear 20mm
    sensors_rear[time][5] = ((temperature[6][n_y//2+1] [n_x//2-13] * 4/5 + temperature[6][n_y//2+1] [n_x//2-12] * 1/5) * 3/5 + (temperature[6][n_y//2+2] [n_x//2-13] * 4/5 + temperature[6][n_y//2+2] [n_x//2-12] * 1/5) * 2/5)       # rear 25mm
    sensors_rear[time][6] = ((temperature[7][n_y//2+3][n_x//2-17] * 4/5 + temperature[7][n_y//2+2][n_x//2-17] * 1/5) * 4/5 + (temperature[7][n_y//2+3][n_x//2-16] * 4/5 + temperature[7][n_y//2+2][n_x//2-16] * 1/5) * 1/5)     # rear 30mm
    sensors_rear[time][7] = (temperature[9][n_y//2][n_x//2-14] * 2/5 + temperature[9][n_y//2][n_x//2-13] * 3/5)         # rear 40mm
    sensors_rear[time][8] = ((temperature[11][n_y//2] [n_x//2-13] * 1/5 + temperature[11][n_y//2+1] [n_x//2-13] * 4/5) * 4/5 + (temperature[11][n_y//2] [n_x//2-12] * 1/5 + temperature[11][n_y//2+1] [n_x//2-12] * 4/5) * 1/5)       # rear 50mm
    sensors_rear[time][9] = (temperature[16][n_y//2-2][n_x//2-15] * 3/5 + temperature[16][n_y//2-2][n_x//2-16] * 2/5)       # rear 75mm
    sensors_rear[time][10] = ((temperature[20][n_y//2][n_x//2-13] * 3/5 + temperature[20][n_y//2-1][n_x//2-13] * 2/5) * 4/5 + (temperature[20][n_y//2][n_x//2-12] * 3/5 + temperature[20][n_y//2-1][n_x//2-12] * 2/5) * 1/5)     # rear 100mm (20, because 21 is already sample holder)
    sensors_right[time][0] = (temperature[2][n_y//2+14][n_x//2] * 3/5 + temperature[2][n_y//2+14][n_x//2-1] * 2/5)   # right 5mm
    sensors_right[time][1] = temperature[3][n_y//2+12][n_x//2-4]  # right 10mm
    sensors_right[time][2] = ((temperature[4][n_y//2+15][n_x//2+1] * 4/5 + temperature[4][n_y//2+15][n_x//2+2] * 1/5) * 4/5 + (temperature[4][n_y//2+16][n_x//2+1] * 4/5 + temperature[4][n_y//2+16][n_x//2+2] * 1/5) * 1/5) # right 15mm
    sensors_right[time][3] = ((temperature[4][n_y//2+15][n_x//2+10] * 4/5 + temperature[4][n_y//2+16][n_x//2+10] * 1/5) * 4/5 + (temperature[4][n_y//2+15][n_x//2+9] * 4/5 + temperature[4][n_y//2+16][n_x//2+9] * 1/5) * 1/5)  # right 15mm side
    sensors_right[time][4] = (temperature[5][n_y//2+15][n_x//2-3] * 3/5 + temperature[5][n_y//2+14][n_x//2-3] * 2/5)  # right 20mm
    sensors_right[time][5] = (temperature[6][n_y//2+13][n_x//2] * 3/5 + temperature[6][n_y//2+12][n_x//2] * 2/5)  # right 25mm
    sensors_right[time][6] = ((temperature[7][n_y//2+16][n_x//2-1] * 2/5 + temperature[7][n_y//2+15][n_x//2-1] * 3/5) * 3/5 + (temperature[7][n_y//2+16][n_x//2-2] * 2/5 + temperature[7][n_y//2+15][n_x//2-2] * 3/5) * 2/5)  # right 30mm
    sensors_right[time][7] = ((temperature[9][n_y//2+13][n_x//2+1] * 3/5 + temperature[9][n_y//2+13][n_x//2+2] * 2/5) * 3/5 + (temperature[9][n_y//2+12][n_x//2+1] * 3/5 + temperature[9][n_y//2+12][n_x//2+2] * 2/5) * 2/5) # right 40mm
    sensors_right[time][8] = (temperature[11][n_y//2+15][n_x//2-1] * 3/5 + temperature[11][n_y//2+14][n_x//2-1] * 2/5)  # right 50mm
    sensors_right[time][9] = ((temperature[16][n_y//2+16][n_x//2-2] * 4/5 + temperature[16][n_y//2+16][n_x//2-3] * 1/5) * 2/5 + (temperature[16][n_y//2+15][n_x//2-2] * 4/5 + temperature[16][n_y//2+15][n_x//2-3] * 1/5) * 3/5) # right 75mm
    sensors_right[time][10] = ((temperature[20][n_y//2+14][n_x//2+15] * 1/5 + temperature[20][n_y//2+13][n_x//2+15] * 4/5) * 3/5 + (temperature[20][n_y//2+14][n_x//2+16] * 1/5 + temperature[20][n_y//2+13][n_x//2+16] * 4/5) * 2/5)  # right 100mm side (20, because 21 is already sample holder)
    return sensors_right, sensors_rear