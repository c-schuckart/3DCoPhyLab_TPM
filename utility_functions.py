import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
import csv
import pandas as pd
from os import path
from scipy.interpolate import interp1d


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


@njit
def save_sensors_L_sample_holder_high_res(n_x, n_y, n_z, temperature, sensors_right, sensors_rear, time, hl, sf=1):
    sensors_rear[time][0] = (temperature[hl[0]][n_y//2-2*sf][n_x//2-15*sf] * 3/5 + temperature[hl[0]][n_y//2-2*sf][n_x//2-16*sf] * 2/5)       # rear 5mm
    sensors_rear[time][1] = ((temperature[hl[1]][n_y//2] [n_x//2-13*sf] * 1/5 + temperature[hl[1]][n_y//2+1*sf] [n_x//2-13*sf] * 4/5) * 4/5 + (temperature[hl[1]][n_y//2] [n_x//2-12*sf] * 1/5 + temperature[hl[1]][n_y//2+1*sf] [n_x//2-12*sf] * 4/5) * 1/5)      # rear 10mm
    sensors_rear[time][2] = (temperature[hl[1]][n_y//2-12*sf] [n_x//2-13*sf] * 4/5 + temperature[hl[1]][n_y//2-12*sf] [n_x//2-12*sf] * 1/5)       # rear 10mm side
    sensors_rear[time][3] = temperature[hl[2]][n_y//2-1*sf][n_x//2-17*sf]        # rear 15mm
    sensors_rear[time][4] = (temperature[hl[3]][n_y//2+3*sf][n_x//2-15*sf] * 1/5 + temperature[hl[3]][n_y//2+3*sf][n_x//2-14*sf] * 4/5)      # rear 20mm
    sensors_rear[time][5] = ((temperature[hl[4]][n_y//2+1*sf] [n_x//2-13*sf] * 4/5 + temperature[hl[4]][n_y//2+1*sf] [n_x//2-12*sf] * 1/5) * 3/5 + (temperature[hl[4]][n_y//2+2*sf] [n_x//2-13*sf] * 4/5 + temperature[hl[4]][n_y//2+2*sf] [n_x//2-12*sf] * 1/5) * 2/5)       # rear 25mm
    sensors_rear[time][6] = ((temperature[hl[5]][n_y//2+3*sf][n_x//2-17*sf] * 4/5 + temperature[hl[5]][n_y//2+2*sf][n_x//2-17*sf] * 1/5) * 4/5 + (temperature[hl[5]][n_y//2+3*sf][n_x//2-16*sf] * 4/5 + temperature[hl[5]][n_y//2+2*sf][n_x//2-16*sf] * 1/5) * 1/5)     # rear 30mm
    sensors_rear[time][7] = (temperature[hl[6]][n_y//2][n_x//2-14*sf] * 2/5 + temperature[hl[6]][n_y//2][n_x//2-13*sf] * 3/5)         # rear 40mm
    sensors_rear[time][8] = ((temperature[hl[7]][n_y//2] [n_x//2-13*sf] * 1/5 + temperature[hl[7]][n_y//2+1*sf] [n_x//2-13*sf] * 4/5) * 4/5 + (temperature[hl[7]][n_y//2][n_x//2-12*sf] * 1/5 + temperature[hl[7]][n_y//2+1*sf] [n_x//2-12*sf] * 4/5) * 1/5)       # rear 50mm
    sensors_rear[time][9] = (temperature[hl[8]][n_y//2-2*sf][n_x//2-15*sf] * 3/5 + temperature[hl[8]][n_y//2-2*sf][n_x//2-16*sf] * 2/5)       # rear 75mm
    sensors_rear[time][10] = ((temperature[hl[9]][n_y//2][n_x//2-13*sf] * 3/5 + temperature[hl[9]][n_y//2-1*sf][n_x//2-13*sf] * 2/5) * 4/5 + (temperature[hl[9]][n_y//2][n_x//2-12*sf] * 3/5 + temperature[hl[9]][n_y//2-1*sf][n_x//2-12*sf] * 2/5) * 1/5)     # rear 100mm (20, because 21 is already sample holder)
    sensors_right[time][0] = (temperature[hl[0]][n_y//2+14*sf][n_x//2] * 3/5 + temperature[hl[0]][n_y//2+14*sf][n_x//2-1*sf] * 2/5)   # right 5mm
    sensors_right[time][1] = temperature[hl[1]][n_y//2+12*sf][n_x//2-4*sf]  # right 10mm
    sensors_right[time][2] = ((temperature[hl[2]][n_y//2+15*sf][n_x//2+1*sf] * 4/5 + temperature[hl[2]][n_y//2+15*sf][n_x//2+2*sf] * 1/5) * 4/5 + (temperature[hl[2]][n_y//2+16*sf][n_x//2+1*sf] * 4/5 + temperature[hl[2]][n_y//2+16*sf][n_x//2+2*sf] * 1/5) * 1/5) # right 15mm
    sensors_right[time][3] = ((temperature[hl[2]][n_y//2+15*sf][n_x//2+10*sf] * 4/5 + temperature[hl[2]][n_y//2+16*sf][n_x//2+10*sf] * 1/5) * 4/5 + (temperature[hl[2]][n_y//2+15*sf][n_x//2+9*sf] * 4/5 + temperature[hl[2]][n_y//2+16*sf][n_x//2+9*sf] * 1/5) * 1/5)  # right 15mm side
    sensors_right[time][4] = (temperature[hl[3]][n_y//2+15*sf][n_x//2-3*sf] * 3/5 + temperature[hl[3]][n_y//2+14*sf][n_x//2-3*sf] * 2/5)  # right 20mm
    sensors_right[time][5] = (temperature[hl[4]][n_y//2+13*sf][n_x//2] * 3/5 + temperature[hl[4]][n_y//2+12*sf][n_x//2] * 2/5)  # right 25mm
    sensors_right[time][6] = ((temperature[hl[5]][n_y//2+16*sf][n_x//2-1*sf] * 2/5 + temperature[hl[5]][n_y//2+15*sf][n_x//2-1*sf] * 3/5) * 3/5 + (temperature[hl[5]][n_y//2+16*sf][n_x//2-2*sf] * 2/5 + temperature[hl[5]][n_y//2+15*sf][n_x//2-2*sf] * 3/5) * 2/5)  # right 30mm
    sensors_right[time][7] = ((temperature[hl[6]][n_y//2+13*sf][n_x//2+1*sf] * 3/5 + temperature[hl[6]][n_y//2+13*sf][n_x//2+2*sf] * 2/5) * 3/5 + (temperature[hl[6]][n_y//2+12*sf][n_x//2+1*sf] * 3/5 + temperature[hl[6]][n_y//2+12*sf][n_x//2+2*sf] * 2/5) * 2/5) # right 40mm
    sensors_right[time][8] = (temperature[hl[7]][n_y//2+15*sf][n_x//2-1*sf] * 3/5 + temperature[hl[7]][n_y//2+14*sf][n_x//2-1*sf] * 2/5)  # right 50mm
    sensors_right[time][9] = ((temperature[hl[8]][n_y//2+16*sf][n_x//2-2*sf] * 4/5 + temperature[hl[8]][n_y//2+16*sf][n_x//2-3*sf] * 1/5) * 2/5 + (temperature[hl[8]][n_y//2+15*sf][n_x//2-2*sf] * 4/5 + temperature[hl[8]][n_y//2+15*sf][n_x//2-3*sf] * 1/5) * 3/5) # right 75mm
    sensors_right[time][10] = ((temperature[hl[9]][n_y//2+14*sf][n_x//2+15*sf] * 1/5 + temperature[hl[9]][n_y//2+13*sf][n_x//2+15*sf] * 4/5) * 3/5 + (temperature[hl[9]][n_y//2+14*sf][n_x//2+16*sf] * 1/5 + temperature[hl[9] ][n_y//2+13*sf][n_x//2+16*sf] * 4/5) * 2/5)  # right 100mm side (20, because 21 is already sample holder)
    return sensors_right, sensors_rear


def prescribe_temp_profile_from_data(n_x, n_y, n_z, temperature, time_profile, surface_temp, bottom_temp, file, height_list, sample_holder):
    data = pd.read_csv(file,
                       names=['Time', 'pen1', 'pen2', 'pen3', 'MOT1', 'MOT2', 'Right_25', 'Rear_25', 'Right_20',
                              'Rear_20', 'Right_15', 'Rear_15', 'Right_10', 'Rear_10_side', 'Right_15_side',
                              'Rear_5', 'Right_5', 'Rear_10', 'Sidewall_55', 'Sidewall_25', 'Copperplate',
                              'Sidewall_85',
                              'Blackbody',
                              'Right_30', 'Rear_30', 'Rear_40', 'Right_40', 'Right_50', 'Rear_50',
                              'Rear_75', 'Right_75', 'Right_100_side', 'Rear_100', 'CP_tube',
                              'CS_left_top', 'CS_rear_top', 'CS_right_top', 'CS_top_plate', 'CS_left_bot',
                              'CS_rear_bot',
                              'CS_right_bot'], sep=',', skiprows=1)

    data['Time'] = pd.to_datetime(data['Time'], format='%d_%m_%Y_%H:%M:%S')

    shift = -28

    coef = [3.9083e-3, -5.775e-7, -4.183e-12]
    trange = np.linspace(60, 550, 500) - 273.15
    R = (1 + coef[0] * trange + coef[1] * trange ** 2 + coef[2] * (trange - 100) * trange ** 3) * 1000
    f = interp1d(R, trange, bounds_error=False, fill_value=np.nan)

    sensor_list = np.zeros(10, dtype=np.float64)
    count = 0
    for each in ['Rear_5', 'Rear_10', 'Rear_15', 'Rear_20', 'Rear_25', 'Rear_30', 'Rear_40', 'Rear_50', 'Rear_75',
                 'Rear_100']:
        sensor_list[count] = ((data[each] + shift).apply(f) + 273.15)[time_profile]
        count += 1

    profile = np.zeros(np.shape(temperature)[0], dtype=np.float64)
    if np.isnan(sensor_list[0]):
        sensor_1 = (surface_temp + sensor_list[1]) / 2
    for i in range(0, height_list[0] - 1):
        profile[i + 1] = surface_temp + (sensor_list[0] - surface_temp) * i / (height_list[0] - 1)
    for j in range(0, len(sensor_list) - 1):
        for i in range(height_list[j] + 1, height_list[j + 1]):
            profile[i] = sensor_list[j] + (sensor_list[j + 1] - sensor_list[j]) * (i - height_list[j]) / (
                        height_list[j + 1] - height_list[j])
    for i in range(height_list[9] + 1, n_z - 2):
        profile[i] = sensor_list[9] + (bottom_temp - sensor_list[9]) * (i - height_list[9]) / (n_z - 2 - height_list[9])
    profile[height_list[0]] = sensor_list[0]
    profile[height_list[1]] = sensor_list[1]
    profile[height_list[2]] = sensor_list[2]
    profile[height_list[3]] = sensor_list[3]
    profile[height_list[4]] = sensor_list[4]
    profile[height_list[5]] = sensor_list[5]
    profile[height_list[6]] = sensor_list[6]
    profile[height_list[7]] = sensor_list[7]
    profile[height_list[8]] = sensor_list[8]
    profile[height_list[9]] = sensor_list[9]
    profile[n_z-2] = bottom_temp
    sand_base_temp = np.full(n_z, bottom_temp, dtype=np.float64)
    for j in range(1, n_y - 1):
        for k in range(1, n_x - 1):
            # temperature[1:n_z-1, j, k] = sand_base_temp[1:n_z-1] + (profile[1:n_z-1] - sand_base_temp[1:n_z-1]) * np.exp(- (((j - n_y//2) * min_dy)**2 + ((k - n_x//2) * min_dx)**2) / (1/2*r_sh)**2)
            if np.sum(temperature[0:n_z-2, j, k]) > 0 and np.sum(sample_holder[0:n_z-2, j, k]) == 0:
                temperature[0:n_z, j, k] = profile
            if np.isnan(temperature[1, j, k]):
                print(j, k)
    # plt.plot(np.arange(1, n_z), profile[1:n_z])
    # plt.show()
    return temperature

@njit
def prescribe_crater(n_x, n_y, n_z, temperature, Dr, amplitude, sigma, min_dz):
    for j in range(0, n_y):
        for k in range(0, n_x):
            if temperature[1][j][k] > 0:
                if j <= n_y//2 and k <= n_x//2:
                    r = np.sqrt(np.sum(Dr[1, j:n_y//2, k, 2])**2 + np.sum(Dr[1, j, k:n_x//2, 4])**2)
                elif j > n_y//2 and k <= n_x//2:
                    r = np.sqrt(np.sum(Dr[1, n_y//2:j, k, 2]) ** 2 + np.sum(Dr[1, j, k:n_x // 2, 4]) ** 2)
                elif j <= n_y//2 and k > n_x//2:
                    r = np.sqrt(np.sum(Dr[1, j:n_y//2, k, 2])**2 + np.sum(Dr[1, j, n_x//2:k, 4])**2)
                else:
                    r = np.sqrt(np.sum(Dr[1, n_y//2:j, k, 2]) ** 2 + np.sum(Dr[1, j, n_x//2:k, 4]) ** 2)
                h_crater = amplitude * np.exp(- r ** 2 / (2 * sigma ** 2))
                for i in range(1, n_z):
                    if h_crater > np.sum(Dr[1:i+1, j, k, 0]) - min_dz/2:
                        temperature[i][j][k] = 0
                    else:
                        break
    return temperature


@njit
def artificial_crater_heating(n_x, n_y, n_z, temperature, surface_reduced, factor):
    max_temp_crater = np.nanmax(temperature[0:n_z-2, n_y//2-4:n_y//2+5, n_x//2-4:n_x//2+5])
    for each in surface_reduced:
        if each[2] > 1:
            temperature[each[2]][each[1]][each[0]] = np.maximum(max_temp_crater * factor, temperature[each[2]][each[1]][each[0]])
    return temperature


def sort_csv_ice(path, sort_avrg, outpath):
    csvdf = pd.read_csv(path, names=['Name', 'S1_avrg', 'S1_max', 'S2_avrg', 'S2_max', 'S3_avrg', 'S3_max', 'S4_avrg', 'S4_max', 'S5_avrg', 'S5_max', 'S6_avrg', 'S6_max', 'S7_avrg', 'S7_max', 'S8_avrg', 'S8_max', 'S9_avrg', 'S9_max', 'S10_avrg', 'S10_max', 'S11_avrg', 'S11_max'])
    csvdf_mirror = csvdf.copy()
    s1 = csvdf['Name']
    s2 = csvdf['S1_avrg'] + csvdf['S2_avrg'] + csvdf['S3_avrg'] + csvdf['S4_avrg'] + csvdf['S5_avrg'] + csvdf['S6_avrg'] + csvdf['S7_avrg'] + csvdf['S8_avrg'] + csvdf['S9_avrg'] + csvdf['S10_avrg'] + csvdf['S11_avrg']
    s3 = csvdf['S1_max'] + csvdf['S2_max'] + csvdf['S3_max'] + csvdf['S4_max'] + csvdf['S5_max'] + csvdf['S6_max'] + csvdf['S7_max'] + csvdf['S8_max'] + csvdf['S9_max'] + csvdf['S10_max'] + csvdf['S11_max']
    content_arr = pd.DataFrame({'A': s1, 'B': s2, 'C': s3})
    if sort_avrg:
        content_arr.sort_values(by='B', inplace=True)
    else:
        content_arr.sort_values(by='C', inplace=True)
    #print(content_arr)
    #print(csvdf_mirror)
    for i in range(len(csvdf)):
        #print(csvdf.iloc[content_arr.iloc[[i]].index[0]][1])
        csvdf_mirror.iloc[i] = csvdf.iloc[content_arr.iloc[[i]].index[0]][0], csvdf.iloc[content_arr.iloc[[i]].index[0]][1], csvdf.iloc[content_arr.iloc[[i]].index[0]][2], csvdf.iloc[content_arr.iloc[[i]].index[0]][3], csvdf.iloc[content_arr.iloc[[i]].index[0]][4], csvdf.iloc[content_arr.iloc[[i]].index[0]][5], csvdf.iloc[content_arr.iloc[[i]].index[0]][6], csvdf.iloc[content_arr.iloc[[i]].index[0]][7], csvdf.iloc[content_arr.iloc[[i]].index[0]][8], csvdf.iloc[content_arr.iloc[[i]].index[0]][9], csvdf.iloc[content_arr.iloc[[i]].index[0]][10], csvdf.iloc[content_arr.iloc[[i]].index[0]][11], csvdf.iloc[content_arr.iloc[[i]].index[0]][12], csvdf.iloc[content_arr.iloc[[i]].index[0]][13], csvdf.iloc[content_arr.iloc[[i]].index[0]][14], csvdf.iloc[content_arr.iloc[[i]].index[0]][15], csvdf.iloc[content_arr.iloc[[i]].index[0]][16], csvdf.iloc[content_arr.iloc[[i]].index[0]][17], csvdf.iloc[content_arr.iloc[[i]].index[0]][18], csvdf.iloc[content_arr.iloc[[i]].index[0]][19], csvdf.iloc[content_arr.iloc[[i]].index[0]][20], csvdf.iloc[content_arr.iloc[[i]].index[0]][21], csvdf.iloc[content_arr.iloc[[i]].index[0]][22]
    #print(csvdf_mirror)
    csvdf_mirror.to_csv(outpath)


def save_mean_temps_light_spot(n_x, n_y, n_z, temperature, path):
    target = open(path, 'a')
    lightspot = np.zeros((11, 11), dtype=np.float64)
    for j in range(n_y//2-5, n_y//2+6):
        for k in range(n_x//2-5, n_x//2+6):
            for i in range(1, n_z-1):
                if temperature[i][j][k] > 0:
                    lightspot[j - n_y//2-5][k - n_x//2-5] = temperature[i][j][k]
                    break
    temp_string = str(np.max(lightspot)) + ',' + str(np.mean(lightspot)) + ',' + str(np.median(lightspot)) + '\n'
    target.write(temp_string)
    target.close()