import numpy as np
from numba import njit, prange
import csv
import pandas as pd
from os import path
import matplotlib.pyplot as plt


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

def prescribe_temp_profile_from_data(n_x, n_y, n_z, temperature, surface_temp, bottom_temp, file, timestamp, height_1, height_2, height_3, height_4, height_5, min_dx, min_dy, r_sh):
    sensor_1, sensor_2, sensor_3, sensor_4, sensor_5 = 0, 0, 0, 0, 0
    with open(file) as csvdatei:
        dat = csv.reader(csvdatei)
        for each in dat:
            if each[0] == timestamp:
                sensor_1 = float(each[1])
                sensor_2 = float(each[2])
                sensor_3 = float(each[3])
                sensor_4 = float(each[4])
                sensor_5 = float(each[5])
    profile = np.zeros(np.shape(temperature)[0], dtype=np.float64)
    if np.isnan(sensor_1):
        sensor_1 = (surface_temp + sensor_2) / 2
    for i in range(0, height_1-1):
        profile[i+1] = surface_temp + (sensor_1 - surface_temp) * i / (height_1-1)
    for i in range(height_1+1, height_2):
        profile[i] = sensor_1 + (sensor_2 - sensor_1) * (i - height_1)/(height_2 - height_1)
    for i in range(height_2+1, height_3):
        profile[i] = sensor_2 + (sensor_3 - sensor_2) * (i - height_2)/(height_3 - height_2)
    for i in range(height_3+1, height_4):
        profile[i] = sensor_3 + (sensor_4 - sensor_3) * (i - height_3)/(height_4 - height_3)
    for i in range(height_4+1, height_5):
        profile[i] = sensor_4 + (sensor_5 - sensor_4) * (i - height_4) / (height_5 - height_4)
    for i in range(height_5+1, n_z-1):
        profile[i] = sensor_5 + (bottom_temp - sensor_5) * (i - height_5) / (n_z - 1 - height_5)
    profile[height_1] = sensor_1
    profile[height_2] = sensor_2
    profile[height_3] = sensor_3
    profile[height_4] = sensor_4
    profile[height_5] = sensor_5
    sand_base_temp = np.full(n_z, bottom_temp, dtype=np.float64)
    for j in range(1, n_y-1):
        for k in range(1, n_x-1):
            temperature[1:n_z-1, j, k] = sand_base_temp[1:n_z-1] + (profile[1:n_z-1] - sand_base_temp[1:n_z-1]) * np.exp(- (((j - n_y//2) * min_dy)**2 + ((k - n_x//2) * min_dx)**2) / (1/2*r_sh)**2)
            #temperature[0:n_z, j, k] = profile
            if np.isnan(temperature[1, j, k]):
                print(j, k)
    '''print(temperature[11][n_y//2-8][n_x//2-8] * 3/4 + temperature[11][n_y//2-7][n_x//2-7] * 1/4)
    print(temperature[21][n_y//2+3-8][n_x//2-8] * 3/4 + temperature[21][n_y//2+3-7][n_x//2-7] * 1/4)
    print((temperature[22][n_y//2-3-8][n_x//2-8] * 3/4 + temperature[22][n_y//2-3-7][n_x//2-7] * 1/4) * 4/5 + (temperature[23][n_y//2-3-8][n_x//2-8] * 3/4 + temperature[23][n_y//2-3-7][n_x//2-7] * 1/4) * 1/5)
    print(temperature[24][n_y//2-8][n_x//2-8] * 3/4 + temperature[24][n_y//2-7][n_x//2-7] * 1/4)
    print(temperature[27][n_y//2+3-8][n_x//2-8] * 3/4 + temperature[27][n_y//2+3-8][n_x//2-8] * 1/4)'''
    #plt.plot(np.arange(1, n_z), profile[1:n_z])
    #plt.show()
    return temperature


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


def sort_csv_big(path, sort_avrg, outpath):
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


def ambient_temperature_from_data(path):
    times = []
    outer_mean = []
    csv_file = open(path, 'r')
    dat = csv.reader(csv_file)
    for count, each in enumerate(dat):
        if count == 0:
            pass
        else:
            if count == 1:
                start_time = np.datetime64(each[0])
            times.append((np.datetime64(each[0]) - start_time).astype(int))
            outer_mean.append(float(each[4]))
    return times, outer_mean