import numpy as np
from numba import njit, prange


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