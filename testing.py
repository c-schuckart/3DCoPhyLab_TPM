import json
import numpy as np
import matplotlib.pyplot as plt
import constants as const
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
from IPython.display import Video
import csv
from data_input import read_temperature_data, getPath
from utility_functions import sort_csv
from os import listdir

rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'
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
print(np.datetime64('2023-07-17 10:53:05') + np.timedelta64(270000, 's'))
'''
A_arr = []
D_arr = []
L_arr = []

with open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/results_csv_sorted.csv') as csvdatei:
    dat = csv.reader(csvdatei)
    for each in dat:
        path = 'C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/' + str(each[1]) + '.json'
        A = ''
        L = ''
        D = ''
        nr = 0
        last_letter_num = False
        for letters in each[1]:
            if letters.isnumeric() or letters == '.':
                if nr == 0:
                    A = A + letters
                if nr == 1:
                    D = D + letters
                if nr == 2:
                    L = L + letters
                last_letter_num = True
            elif last_letter_num:
                nr += 1
                last_letter_num = False
        print(A, D, L)
        if each[0].isnumeric():
            A_arr.append(A)
            D_arr.append(D)
            L_arr.append(L)
        if each[0] == '15':
            break

print(A_arr)

for i in range(0, 16):
    fig, ax = plt.subplots(1, 1)
    A = A_arr[i]
    D = D_arr[i]
    L = L_arr[i]
'''
fig, ax = plt.subplots(1, 1)
A, D, L = '0.95', '0.001', '0.003'

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
    start = False
    for each in dat:
        #if each[0] == '2023-07-17 10:53:05' or start:
        if each[0] == '2023-07-20 13:53:07' or start:
            if b:
                start_time = np.datetime64(each[0])
                b = False
                start = True
            timestamps.append(np.datetime64(each[0]) - start_time)
            sen_1.append(float(each[1]))
            sen_2.append(float(each[2]))
            sen_3.append(float(each[3]))
            sen_4.append(float(each[4]))
            sen_5.append(float(each[5]))
            #sen_1.append(float(each[7]))
            #sen_2.append(float(each[8]))
            #sen_3.append(float(each[9]))
            #sen_4.append(float(each[10]))
            #sen_5.append(float(each[11]))
            #sen_6.append(float(each[6]))
            if timestamps[len(timestamps) - 1] > 120000:
            #if timestamps[len(timestamps) - 1] > 330000:
                break

surface_measured = []
timestamps_surface = []
with open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/sand_surface_temperatures.csv') as csvdatei2:
    dat = csv.reader(csvdatei2)
    b = True
    start = False
    for each in dat:
        #if each[0] == '2023_07_17_10h_52m_52s.png' or start:
        if each[0] == '2023_07_20_13h_53m_00s.png' or start:
            if b:
                start_time = np.datetime64(each[0][0:4] + '-' + each[0][5:7] + '-' + each[0][8:10] + ' ' + each[0][11:13] + ':' + each[0][15:17] + ':' + each[0][19:21])
                b = False
                start = True
            timestamps_surface.append(np.datetime64(each[0][0:4] + '-' + each[0][5:7] + '-' + each[0][8:10] + ' ' + each[0][11:13] + ':' + each[0][15:17] + ':' + each[0][19:21]) - start_time)
            surface_measured.append(float(each[1]))
            if timestamps_surface[len(timestamps_surface) - 1] > 120000:
            #if timestamps[len(timestamps) - 1] > 330000:
                break

#ax.plot(timestamps, sen_1, label='1. mitt. sensor', color='tab:blue')
ax.plot(timestamps, sen_2, label='2. mitt. sensor', color='tab:orange')
ax.plot(timestamps, sen_3, label='3. mitt. sensor', color='tab:green')
ax.plot(timestamps, sen_4, label='4. mitt. sensor', color='tab:red')
ax.plot(timestamps, sen_5, label='5. mitt. sensor', color='tab:purple')
#plt.plot(timestamps, sen_6, label='6. mid sensor')

#ax.plot(timestamps_surface, surface_measured, label='surface hottest temperature', color='tab:blue')

time = [i * const.dt for i in range(0, const.k)]
sen_1_sim = []
sen_2_sim = []
sen_3_sim = []
sen_4_sim = []
sen_5_sim = []
#sen_6 = []

with open('D:/TPM_Data/Big_sand/Thesis_run/Periodic_sand_sim_thesis_day_night_third_best_fit_300K.json') as json_file:
#with open('D:/TPM_Data/Big_sand/Thesis_run/Periodic_sand_sim_thesis_' + A + '_Absdepth_' + D + '_Lambda_' + L +'.json') as json_file:
    jdata = json.load(json_file)

for i in range(0, const.k):
    sen_1_sim.append(jdata['Temperature'][i][0])
    sen_2_sim.append(jdata['Temperature'][i][1])
    sen_3_sim.append(jdata['Temperature'][i][2])
    sen_4_sim.append(jdata['Temperature'][i][3])
    sen_5_sim.append(jdata['Temperature'][i][4])

ax.plot(time, sen_1_sim, label='1. mitt. sensor SIM', color='#000000')
ax.plot(time, sen_2_sim, label='2. mitt. sensor SIM', color='#272727', ls='--')
ax.plot(time, sen_3_sim, label='3. mitt. sensor SIM', color='#474747', ls=':')
ax.plot(time, sen_4_sim, label='4. mitt. sensor SIM', color='#636363', ls='-.')
ax.plot(time, sen_5_sim, label='5. mitt. sensor SIM', color='#858585')

#with open('D:/TPM_data/Big_sand/Sand_sim_thesis_' + A + '_Absdepth_' + D + '_Lambda_' + L + '.json') as json_file:
#with open('D:/TPM_data/Big_sand/TSand_sim_thesis_heat_cap_840.json') as json_file:
    #jdata_2 = json.load(json_file)

#surface_temp = []
#for each in  jdata_2['Temperature']:
    #surface_temp.append(each[1])


#ax.plot(time, surface_temp, color='#000000', ls='-.')

ax.set_ylim(290, 420)
ax.set_xlim(-5000, 120000)
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
ax.grid(True, lw=0.5)
plt.legend(fontsize='xx-small')
#plt.title('Albedo: ' + A + '; Abs. depth: ' + D + 'm; Lambda: ' + L + 'W/(mK)')
#plt.title('Albedo: ' + A + '; abs. Tiefe: ' + D + 'm; Lambda: ' + L + 'W/(mK)')
plt.title('Innere Sensoren - Kammer Temp. 300K')
ax.set_xlabel('Zeit (s)')
ax.set_ylabel('Temperatur (K)')
#plt.savefig('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Plots/P_thesis_DAY_NIGHT_heat_cap_840_ambient_300K_A_' + A + '_Absdepth_' + D + '_Lambda_' + L + '.png', dpi=600)
plt.savefig('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Plots/P_thesis_day_night_illum_third_best_fit_mid_ambient_300K_de.png', dpi=600)
plt.show()
#plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Plots/CORROuter_sensors_sand_L_chamber_A_0.95_Absdepth_0.001_Lambda_0.003.png', dpi=600)
ax.clear()
fig.clear()

'''with open('D:/TPM_data/Big_sand/sand_L_chamber_test_quick.json') as json_file:
    data_q = json.load(json_file)

with open('D:/TPM_data/Big_sand/sand_L_chamber_test.json') as json_file:
    data = json.load(json_file)

fig, ax = plt.subplots(1, 1)
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
depth_q = []
depth = [i * const.min_dz for i in range(0, 320)]
time = [i * const.dt for i in range(0, 500)]
for i in range(0, 50):
    if i < 21:
        depth_q.append(i * const.min_dz)
    else:
        depth_q.append((i-20) * const.min_dz * 10 + 20 * const.min_dz)
def update(t):
    ax.clear()
    ax.plot(depth, data['Temperature'][int(t//const.dt)][1:321], label='high resolution everywhere')
    ax.plot(depth_q, data_q['Temperature'][int(t//const.dt)][1:51], label='high resolution surface')
    #ax.set_xlim(-15.5, 15.5)
    #ax.set_ylim(290, 420)
    ax.set_title('homogenous mesh v heterogenous mesh')
    ax.set_xlabel('depth (m)')
    ax.set_ylabel('Temperature (K)')
    ax.text(0.9, 430, str(time))
    #if t >= 4240800.0:
        #ax[0].text(0, 15.5, 'EQUILIBRATED')
    #ax[1].set_title('With sublimation')
    #ax[1].set_xlabel('width (cm)')
    #ax[1].set_ylabel('height (cm)')
    fig.canvas.draw()
    fig.canvas.flush_events()
    #plt.show()

anim = animation.FuncAnimation(fig, update, frames=time, interval=200)

#Writer = animation.writers['ffmpeg']
Writer = animation.FFMpegWriter(fps=24, codec='mpeg4', bitrate=8000)
#writer = Writer(fps=5, bitrate=1800)
writer = Writer
anim.save('D:/TPM_Data/Big_sand/test.mp4', writer=writer, dpi=600)
Video('D:/TPM_Data/Big_sand/test.mp4')'''

'''timestamps = []
sen_1 = []
sen_2 = []
sen_3 = []
sen_4 = []
sen_5 = []
sen_6 = []

with open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/temps_sandy_randy.txt') as csvdatei:
    dat = csv.reader(csvdatei)
    b = True
    start = False
    for each in dat:
        if each[0] == '2023-07-17 10:53:05' or start:
            if b:
                start_time = np.datetime64(each[0])
                b = False
                start = True
            timestamps.append(np.datetime64(each[0]) - start_time)
            sen_1.append(float(each[1]))
            sen_2.append(float(each[2]))
            sen_3.append(float(each[3]))
            sen_4.append(float(each[4]))
            sen_5.append(float(each[5]))
            #sen_6.append(float(each[6]))
            if timestamps[len(timestamps) - 1] > 120000:
                break

sen_1s = []
sen_2s = []
sen_3s = []
sen_4s = []
sen_5s = []
counter = 0
for i in range(len(timestamps)-1):
    if timestamps[i]//50 == counter:
        sen_1s.append(sen_1[i])
        sen_2s.append(sen_2[i])
        sen_3s.append(sen_3[i])
        sen_4s.append(sen_4[i])
        sen_5s.append(sen_5[i])
        counter += 1

files = listdir('D:/TPM_Data/Big_sand/Thesis_run/')
target = open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Day_night_thesis_csv.csv', 'w')
for each in files:
    if each[0:8] == 'Periodic':
        with open('D:/TPM_Data/Big_sand/Thesis_run/' + each) as json_file:
            jdata = json.load(json_file)
        sen_1_sim = []
        sen_2_sim = []
        sen_3_sim = []
        sen_4_sim = []
        sen_5_sim = []
        for i in range(0, const.k-600):
            sen_1_sim.append(jdata['Temperature'][i][0])
            sen_2_sim.append(jdata['Temperature'][i][1])
            sen_3_sim.append(jdata['Temperature'][i][2])
            sen_4_sim.append(jdata['Temperature'][i][3])
            sen_5_sim.append(jdata['Temperature'][i][4])
        json_file.close()
        deltas_1 = np.abs(np.array(sen_1s, dtype=np.float64) - np.array(sen_1_sim, dtype=np.float64))
        deltas_2 = np.abs(np.array(sen_2s, dtype=np.float64) - np.array(sen_2_sim, dtype=np.float64))
        deltas_3 = np.abs(np.array(sen_3s, dtype=np.float64) - np.array(sen_3_sim, dtype=np.float64))
        deltas_4 = np.abs(np.array(sen_4s, dtype=np.float64) - np.array(sen_4_sim, dtype=np.float64))
        deltas_5 = np.abs(np.array(sen_5s, dtype=np.float64) - np.array(sen_5_sim, dtype=np.float64))
        #target.write(str(each[:-4]) + '\t' + str(np.average(deltas_1)) + ',' + str(np.max(deltas_1)) + '\t' + str(np.average(deltas_2)) + ',' + str(np.max(deltas_2)) + '\t' + str(np.average(deltas_3)) + ',' + str(np.max(deltas_3)) + '\t' + str(np.average(deltas_4)) + ',' + str(np.max(deltas_4)) + '\t' + str(np.average(deltas_5)) + ',' + str(np.max(deltas_5)) +'\n')
        #target.write(str(each[:-5]) + ',' + str(np.average(deltas_1)) + ',' + str(np.max(deltas_1)) + ',' + str(np.average(deltas_2)) + ',' + str(np.max(deltas_2)) + ',' + str(np.average(deltas_3)) + ',' + str(np.max(deltas_3)) + ',' + str(np.average(deltas_4)) + ',' + str(np.max(deltas_4)) + ',' + str(np.average(deltas_5)) + ',' + str(np.max(deltas_5)) +'\n')
        target.write(str(each[:-5]) + ',' + str(0.0) + ',' + str(0.0) + ',' + str(np.average(deltas_2)) + ',' + str(np.max(deltas_2)) + ',' + str(np.average(deltas_3)) + ',' + str(np.max(deltas_3)) + ',' + str(np.average(deltas_4)) + ',' + str(np.max(deltas_4)) + ',' + str(np.average(deltas_5)) + ',' + str(np.max(deltas_5)) +'\n')

target.close()'''

#sort_csv('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Day_night_thesis_csv.csv', True, 'C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Day_night_thesis_csv_sorted.csv')

'''with open('D:/TPM_data/Big_sand/sand_L_chamber_A_0.95_Absdepth_0.001_Lambda_0.004.json') as json_file:
    jdata1 = json.load(json_file)

with open('D:/TPM_data/Big_sand/sand_L_chamber_A_0.9_Absdepth_0.0005_Lambda_0.004.json') as json_file:
    jdata2 = json.load(json_file)

depth = []
for i in range(0, const.n_z-2):
    if i <= 20:
        depth.append(i * const.min_dz)
    else:
        depth.append(0.01 + (i-20) * const.min_dz * 10)
print(depth)
#print(len(depth), len(jdata1['Temperature'][len(jdata1['Temperature'])-1]))
plt.plot(depth, jdata1['Temperature'][len(jdata1['Temperature'])-1][1:const.n_z-1], label='Best fit')
#plt.plot(depth, jdata2['Temperature'][len(jdata1['Temperature'])-1][1:const.n_z-1], label='Second best fit', ls='--')

plt.ylim(290, 420)
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
plt.grid(True, lw=0.5)
plt.legend(fontsize='x-small')
plt.show()
#plt.savefig('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Temp_profile.png', dpi=600)'''