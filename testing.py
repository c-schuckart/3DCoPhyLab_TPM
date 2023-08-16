import json
import numpy as np
import matplotlib.pyplot as plt
import constants as const
import csv
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

timestamps = []
sen_1 = []
sen_2 = []
sen_3 = []
sen_4 = []
sen_5 = []
sen_6 = []

with open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/temps_sandy_randy.txt') as csvdatei:
    dat = csv.reader(csvdatei)
    b = True
    for each in dat:
        if b:
            start_time = np.datetime64(each[0])
            b = False
        timestamps.append(np.datetime64(each[0]) - start_time)
        sen_1.append(float(each[1]))
        sen_2.append(float(each[2]))
        sen_3.append(float(each[3]))
        sen_4.append(float(each[4]))
        sen_5.append(float(each[5]))
        sen_6.append(float(each[6]))

plt.plot(timestamps, sen_1, label='1. mid sensor')
plt.plot(timestamps, sen_2, label='2. mid sensor')
plt.plot(timestamps, sen_3, label='3. mid sensor')
plt.plot(timestamps, sen_4, label='4. mid sensor')
plt.plot(timestamps, sen_5, label='5. mid sensor')
plt.plot(timestamps, sen_6, label='6. mid sensor')
plt.ylim(290, 420)
plt.legend()
plt.show()
