import json
import numpy as np
import matplotlib.pyplot as plt
import constants as const
from data_input import read_temperature_data, getPath

'''with open('test.json') as json_file:
    data_vis = json.load(json_file)'''


dx_arr = np.full((const.n_z, const.n_y, const.n_x), const.min_dx, dtype=np.float64)
dy_arr = np.full((const.n_z, const.n_y, const.n_x), const.min_dy, dtype=np.float64)
dz_arr = np.full((const.n_z, const.n_y, const.n_x), const.min_dz, dtype=np.float64)

def temperature(x, t):
    return np.sin(np.pi * x / np.sum(dy_arr[const.n_z//2][1:const.n_y-1][const.n_x//2])) * np.exp(- const.lambda_constant/(const.density_water_ice * const.heat_capacity_water_ice) * np.pi**2/(np.sum(dy_arr[const.n_z//2][1:const.n_y-1][const.n_x//2]))**2 * t)

'''temp_begin = []
temp_end = []
for i in range(1, const.n_z-1):
    temp_begin.append(data_vis['Temperature'][0][const.n_z//2][i][const.n_x//2])
    temp_end.append(data_vis['Temperature'][20][const.n_z//2][i][const.n_x//2])
z = [i * const.min_dy for i in range(1, const.n_y-1)]

temp_end_analytical = []
for each in z:
    temp_end_analytical.append(temperature(each, 2000*const.dt))

plt.plot(z, temp_begin)
plt.plot(z, temp_end)
plt.scatter(z, temp_end_analytical)
plt.show()'''

surface_temp = np.genfromtxt('D:/Masterarbeit_data/surface_temp.csv', delimiter=',')
sample_holder_temp = np.genfromtxt('D:/Masterarbeit_data/sample_holder_temp.csv', delimiter=',')

time_deltas_data_surface, surface_temp_data = read_temperature_data(getPath(), '2023-02-15 16:45:00', '2023-02-15 17:45:02', [1], [])
time_deltas_data_interior, sample_holder_temp_data = read_temperature_data(getPath(), '2023-02-15 16:45:00', '2023-02-15 17:45:01', [5], [])

time_surface = [np.sum(time_deltas_data_surface[0:i]) for i in range(len(time_deltas_data_surface))]
time_interior = [np.sum(time_deltas_data_interior[0:i]) for i in range(len(time_deltas_data_interior))]

#plt.plot([const.dt * i for i in range(0, const.k)], surface_temp)
#plt.scatter(time_surface, surface_temp_data, color='red', s=4)
plt.plot([const.dt * i for i in range(0, const.k)], sample_holder_temp)
plt.scatter(time_interior, sample_holder_temp_data, color='green')
plt.xlim(-50, 1000)
plt.show()
