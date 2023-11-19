import json
import numpy as np
import matplotlib.pyplot as plt
import constants as const
import matplotlib.pyplot as plt
import matplotlib.lines as line
from matplotlib import rcParams
from matplotlib.ticker import LogLocator, AutoMinorLocator
import matplotlib.animation as animation
from IPython.display import Video
import csv
from data_input import read_temperature_data, getPath
from utility_functions import sort_csv
from os import listdir
from scipy.interpolate import interp1d
from scipy.optimize import brentq

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
'''fig, ax = plt.subplots(1, 1)
A, D, L = '0.95', '0.003', '0.05'

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
        if each[0] == '2023-07-17 10:53:05' or start:
        #if each[0] == '2023-07-20 13:53:07' or start:
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
            if timestamps[len(timestamps) - 1] > 150000:
            #if timestamps[len(timestamps) - 1] > 330000:
                break

ax.plot(timestamps, sen_1, label='1. mid sensor')
ax.plot(timestamps, sen_2, label='2. mid sensor')
ax.plot(timestamps, sen_3, label='3. mid sensor')
ax.plot(timestamps, sen_4, label='4. mid sensor')
ax.plot(timestamps, sen_5, label='5. mid sensor')
#plt.plot(timestamps, sen_6, label='6. mid sensor')

time = [i * const.dt for i in range(0, const.k)]
sen_1_sim = []
sen_2_sim = []
sen_3_sim = []
sen_4_sim = []
sen_5_sim = []
#sen_6 = []

#with open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/sand_L_chamber_A_0.95_Absdepth_0.001_Lambda_0.003.json') as json_file:
with open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/sand_L_chamber_A_' + A + '_Absdepth_' + D + '_Lambda_' + L +'.json') as json_file:
    jdata = json.load(json_file)

for i in range(0, const.k):
    sen_1_sim.append(jdata['Temperature'][i][0])
    sen_2_sim.append(jdata['Temperature'][i][1])
    sen_3_sim.append(jdata['Temperature'][i][2])
    sen_4_sim.append(jdata['Temperature'][i][3])
    sen_5_sim.append(jdata['Temperature'][i][4])

ax.plot(time, sen_1_sim, label='1. mid sensor SIM', color='#000000')
ax.plot(time, sen_2_sim, label='2. mid sensor SIM', color='#272727', ls='--')
ax.plot(time, sen_3_sim, label='3. mid sensor SIM', color='#474747', ls=':')
ax.plot(time, sen_4_sim, label='4. mid sensor SIM', color='#636363', ls='-.')
ax.plot(time, sen_5_sim, label='5. mid sensor SIM', color='#858585')

with open('D:/TPM_data/Big_sand/sand_L_chamber_A_' + A + '_Absdepth_' + D + '_Lambda_' + L + '.json') as json_file:
    jdata_2 = json.load(json_file)

surface_temp = []
for each in  jdata_2['Temperature']:
    surface_temp.append(each[1])


ax.plot(time, surface_temp, color='#000000', ls='-.')

ax.set_ylim(290, 420)
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
ax.grid(True, lw=0.5)
plt.legend(fontsize='x-small')
plt.title('Albedo: ' + A + '; Abs. depth: ' + D + 'm; Lambda: ' + L + 'W/(mK)')
#plt.title('Best fit (inner sensors): Outer sensors')
plt.show()
#plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Plots/CORRPresentation_sand_L_chamber_A_' + A + '_Absdepth_' + D + '_Lambda_' + L + '.png', dpi=600)
#plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Plots/CORROuter_sensors_sand_L_chamber_A_0.95_Absdepth_0.001_Lambda_0.003.png', dpi=600)
ax.clear()
fig.clear()'''

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

files = listdir('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/')
target = open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/CORRresults_csv.csv', 'w')
for each in files:
    if each[0:10] == 'CORRsand_L':
        with open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/' + each) as json_file:
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
        target.write(str(each[:-5]) + ',' + str(np.average(deltas_1)) + ',' + str(np.max(deltas_1)) + ',' + str(np.average(deltas_2)) + ',' + str(np.max(deltas_2)) + ',' + str(np.average(deltas_3)) + ',' + str(np.max(deltas_3)) + ',' + str(np.average(deltas_4)) + ',' + str(np.max(deltas_4)) + ',' + str(np.average(deltas_5)) + ',' + str(np.max(deltas_5)) +'\n')

target.close()'''

#sort_csv('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/CORRresults_csv.csv', True, 'C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/CORRresults_csv_sorted.csv')

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

'''with open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/Ice/Light_absorption_curve_real.csv') as csvdatei:
    dat = csv.reader(csvdatei, delimiter=';')
    depth = []
    energy = []
    for each in dat:
        depth.append(float(each[0]))
        energy.append(float(each[1]))

depth = np.sort(np.array(depth))
energy = np.sort(np.array(energy))

curve = interp1d(depth, energy)
z = np.zeros(const.n_z+1, dtype=np.float64)
mi = True
for i in range(0, const.n_z+1):
    z[i] = i*const.min_dz
    if z[i] > 1E-3 and mi:
        max_interp = i
        mi = False
intensity = curve(z[0:max_interp])
int_per_layer = np.zeros(const.n_z, dtype=np.float64)
for i in range(0, const.n_z):
    if z[i+1] > 1E-3:
        int_per_layer[i] = 0
    else:
        int_per_layer[i] = (intensity[i+1] - intensity[i])/energy[len(energy)-1]

data_dict = {'factors': int_per_layer.tolist()}

print(np.sum(int_per_layer))
print(max_interp)

with open('lamp_layer_absorption_factors_periodic300.json', 'w') as outfile:
    json.dump(data_dict, outfile)
'''

'''chi = [0.326, 0.1, 0.05, 0.01, 0.005, 0.001]
T_10 = [85.1647, 91.3946, 96.8681, 116.779, 128.6403, 158.08926]
T_60 = [97.6746, 114.0831, 128.6047, 165.8284, 170.3352, 174.6266]

fig, ax = plt.subplots(1, 1)
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
ax.plot(chi, T_10, label='Highest Temp. after 10s', marker='x')
ax.plot(chi, T_60, label='Highest Temp. after 60s', ls='--', marker='x')
ax.set_xlabel(r'$\chi$')
ax.set_ylabel('Temperature (K)')
ax.set_title(r'Highest temperature with changing $\chi$')
ax.set_xscale('log')
plt.legend()
plt.savefig('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/Ice/chi_dependency.png', dpi=600)'''

'''with open('D:/TPM_Data/Noah/Outgassing_tests/middle_injection_dt_5e-8.json') as json_file:
    jdata1 = json.load(json_file)
with open('D:/TPM_Data/Noah/Outgassing_tests/middle_injection_dt_1e-7.json') as json_file:
    jdata2 = json.load(json_file)
with open('D:/TPM_Data/Noah/Outgassing_tests/middle_injection_dt_5e-7.json') as json_file:
    jdata3 = json.load(json_file)
with open('D:/TPM_Data/Noah/Outgassing_tests/middle_injection_dt_1e-6.json') as json_file:
    jdata4 = json.load(json_file)

time_5e8 = [i * 5E-8 for i in range(0, 2000)]
time_1e7 = [i * 1E-7 for i in range(0, 1000)]
time_5e7 = [i * 5E-7 for i in range(0, 200)]
time_1e6 = [i * 1E-6 for i in range(0, 100)]
fig, ax = plt.subplots(1, 1)
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
#ax.fill_between(time_5e8, jdata1['min'], jdata1['max'], label='dt = 5*10^-8', alpha=0.4)
#ax.fill_between(time_1e7, jdata2['min'], jdata2['max'], ls='--', label='dt = 1*10^-7', alpha=0.4)
#ax.fill_between(time_5e7, jdata3['min'], jdata3['max'], ls=':', label='dt = 5*10^-7', alpha=0.4)
#ax.fill_between(time_1e6, jdata4['min'], jdata4['max'], ls='-.', label='dt = 1*10^-6', alpha=0.4)
ax.plot(time_5e8, jdata1['max'], label='dt = 5*10^-8')
ax.plot(time_1e7, jdata2['max'], ls='--', label='dt = 1*10^-7')
ax.plot(time_5e7, jdata3['max'], ls=':', label='dt = 5*10^-7')
ax.plot(time_1e6, jdata4['max'],  ls='-.', label='dt = 1*10^-6')
#ax.plot()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Gas density (kg/m^3)')
ax.set_title('Max gas densities')
ax.set_ylim(1E-21, 1E-14)
ax.set_yscale('log')
plt.legend()
plt.savefig('D:/TPM_Data/Noah/Outgassing_tests/Max_gas_densities_zoom.png', dpi=600)
plt.show()'''

'''with open('D:/TPM_Data/Noah/Outgassing_tests/dynamic_production.json') as json_file:
    data = json.load(json_file)

fig, ax = plt.subplots(1, 1)
time = [i for i in range(0, 300)]'''
def update(t):
    ax.clear()
    #fig.clear()
    a = ax.imshow(np.array(data['field'][t], dtype=np.float64)[0:const.n_z, 1:const.n_y-1])
    l1 = line.Line2D([-0.5, 24], [1.5, 1.5], color='black', lw=4)
    l2 = line.Line2D([-0.5, 24], [22.5, 22.5], color='black', lw=4)
    l3 = line.Line2D([1.5, 1.5], [1.5, 22.5], color='black', lw=4)
    l4 = line.Line2D([20.5, 20.5], [1.5, 22.5], color='black', lw=4)
    l5 = line.Line2D([15.5, 15.5], [1.5, 22.5], color='black', lw=4)
    '''static_lines = [l1, l2, l3, l4]
    for each in static_lines:
        ax.add_artist(each)
    if t < 20:
        ax.add_artist(l5)'''

    #plt.colorbar(a)


'''anim = animation.FuncAnimation(fig, update, frames=time, interval=200)

#Writer = animation.writers['ffmpeg']
Writer = animation.FFMpegWriter(fps=10, codec='mpeg4', bitrate=6000)
#writer = Writer(fps=5, bitrate=1800)
writer = Writer

anim.save('D:/TPM_Data/Noah/Outgassing_tests/Vdynamic_production.mp4', writer=writer, dpi=600)
Video('D:/TPM_Data/Noah/Outgassing_tests/Vdynamic_production.mp4')'''

'''temps = []
times = []
for i in range(0, 100):
    if i == 0:
        time = 0
    else:
        time = i * 0.1
    arr = np.load('D:/TPM_Data/Noah/diffusion_sh/temperatures_' + str(round(time, 2)) + '.npy')
    #print(arr[0:const.n_z, 12, 12], time)
    temps.append(arr[2, 12, 12])
    times.append(time)

fig, ax = plt.subplots(1, 1)
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
ax.plot(times, temps, label=r'$\chi$ = 0.05')
plt.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (K)')
ax.set_title('First layer temperature I=3150 W/m^2 and chi=0.05')
#plt.savefig('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/Ice/Top_temp_chi_0.05_diffusion_uniform.png', dpi=600)
#plt.show()

print(np.load('D:/TPM_Data/Noah/diffusion_sh/r_n_' + str(round(5.0, 2)) + '.npy')[0:const.n_z, 12, 12])
print(np.load('D:/TPM_Data/Noah/diffusion_sh/r_mono_' + str(round(5.0, 2)) + '.npy')[0:const.n_z, 12, 12])
print('--------')
print(np.load('D:/TPM_Data/Noah/diffusion_sh - Kopie/r_n_' + str(round(5.0, 2)) + '.npy')[0:const.n_z, 12, 12])
print(np.load('D:/TPM_Data/Noah/diffusion_sh - Kopie/r_mono_' + str(round(5.0, 2)) + '.npy')[0:const.n_z, 12, 12])'''
'''temps = []
pressures = []
times = []
for i in range(0, 56):
    if i == 0:
        time = 0
    else:
        time = i * 0.1
    arr = np.load('D:/TPM_Data/Noah/diffusion_sh/temperatures_' + str(round(time, 2)) + '.npy')
    arr_p = np.load('D:/TPM_Data/Noah/diffusion_sh/pressure_' + str(round(time, 2)) + '.npy')
    #print(arr[0:const.n_z, 12, 12])
    temps.append(arr[2, 12, 12])
    pressures.append(arr_p[2, 12, 12])
    times.append(time)

temps = np.array(temps, dtype=np.float64)
sub_pressures = 10 ** (const.lh_a_1[0] + const.lh_b_1[0] / temps + const.lh_c_1[0] * np.log10(temps) + const.lh_d_1[0] * temps)

fig, ax = plt.subplots(1, 1)
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
ax.plot(times, pressures, label=r'Top layer pressure')
ax.plot(times, sub_pressures, label=r'Tl sublimation pressures', ls='--', c='black')
plt.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pressure (Pa)')
ax.set_yscale('log')
ax.set_title('First layer pressure and subl. pressure')
plt.savefig('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/Ice/Top_pressure_v_sublpress.png', dpi=600)
#plt.show()'''

'''path = 'C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Ice/dt_dx_test_L2.csv'
segments = ['dx dy = 10E-5 dz = 5E-6 and 1000000* vacuum outgassing:', 'dx dy = 10E-5 dz = 1E-6 and 1000000* vacuum outgassing:', 'dx dy = 10E-5 dz = 5E-7 and 1000000* vacuum outgassing:', 'dx dy = 10E-5 dz = 5E-6 bigger empty top layer (5E-4) and 1000000* vacuum outgassing:', 'dx dy = 10E-5 dz = 5E-6 bigger empty top layer (5E-4):']
dt = [[], [], [], [], []]
ORdx = [1.5224120029217286e-11, 9.371385099535165e-12, 5.0069003492523505e-12]
ORdx_2 = [1.6274905964107774e-11, 1.6466408223601428e-11, 1.6563859370241558e-11, 1.661301765484791e-11]
dx = [5E-6, 1E-5, 2E-5]
dx_2 = [2.5E-6, 5E-6, 1E-5, 2E-5]
OR = [[], [], [], [], []]

for i in range(len(segments)):
    with open(path) as csvdatei:
        dat = csv.reader(csvdatei)
        b = -1
        for each in dat:
            if b < 0 and each[0] == segments[i]:
                b = 0
            elif b >= 0:
                dt[i].append(float(each[0]))
                OR[i].append(float(each[2]))
                b += 1
            if b == 5:
                break


print(dt, OR)

fig, ax = plt.subplots(1, 1)
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, bottom=True, left=True, labeltop=False, labelright=False, labelbottom=True, labelleft=True)
ax.plot(dt[0], OR[0], marker='x', label=r'dz = $5 * 10^{-6}$')
ax.plot(dt[1], OR[1], marker='o', label=r'dz = $1 * 10^{-6}$')
ax.plot(dt[2], OR[2], marker='d', label=r'dz = $5 * 10^{-7}$')
#ax.plot(dt[3], OR[3], marker='p', label=r'vacuum diff rate = $* 1000000$')
#ax.plot(dt[4], OR[4], marker='s', label=r'empty layer = $5 * 10^{-4}$')
plt.legend(loc=5, fontsize='x-small')
ax.set_xlabel('Time step (s)')
ax.set_ylabel('Outgassing rate (kg/s)')
ax.set_xscale('log')
ax.set_yscale('log')
#plt.ylim(1.5223E-11, 1.5225E-11)
fig.set_tight_layout(True)
ax.set_title(r'High diffrate with changing dz')
plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Ice/changing_dz_same_empty_as_dz.png', dpi=600)
plt.show()'''

path = 'C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Ice/'
files = ['5e-6_first_layer_particle_radius.npy', '1e-6_first_layer_particle_radius.npy', '5e-7_first_layer_particle_radius.npy', '1e-5_first_layer_particle_radius.npy']

arr_normal = np.load(path+files[0])
arr_smaller = np.load(path+files[1])
arr_smallest = np.load(path+files[2])
arr_bigger = np.load(path+files[3])

dx_normal = []
dx_smaller = []
dx_smallest = []
dx_bigger = []


for i in range(0, 43):
    if i > 2:
        dx_normal.append(i * 5E-6 + 5E-6)
    elif i == 0:
        dx_normal.append(0)
        dx_smaller.append(0)
        dx_smallest.append(0)
        dx_bigger.append(0)
    elif i == 1:
        dx_normal.append(2.5E-6)
        dx_smaller.append(2.5E-6)
        dx_smallest.append(2.5E-6)
        dx_bigger.append(2.5E-6)
    elif i == 2:
        dx_normal.append(5E-6)
        dx_smaller.append(5E-6)
        dx_smallest.append(5E-6)
        dx_bigger.append(5E-6)

for i in range(3, 203):
    dx_smaller.append(i * 1E-6 + 5E-6)
for i in range(3, 403):
    dx_smallest.append(i * 5E-7 + 5E-6)
for i in range(3, 23):
    dx_bigger.append(i * 1E-5 + 5E-6)

fig, ax = plt.subplots(1, 1)
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, bottom=True, left=True, labeltop=False, labelright=False, labelbottom=True, labelleft=True)
ax.plot(dx_bigger, arr_bigger, marker='p', label=r'dz = $1 * 10^{-5}$, OR = $2.001 * 10^{-12}')
ax.plot(dx_normal, arr_normal, marker='x', label=r'dz = $5 * 10^{-6}$, OR = $2.013 * 10^{-12}')
ax.plot(dx_smaller, arr_smaller, marker='o', label=r'dz = $1 * 10^{-6}$, OR = $2.059 * 10^{-12}')
ax.plot(dx_smallest, arr_smallest, marker='d', label=r'dz = $5 * 10^{-7}$, OR = $2.083 * 10^{-12}')
#ax.plot(dt[4], OR[4], marker='s', label=r'empty layer = $5 * 10^{-4}$')
plt.legend(loc=5, fontsize='x-small')
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Gas density (kg/m^3)')
#ax.set_xscale('log')
#ax.set_yscale('log')
#plt.ylim(1.5223E-11, 1.5225E-11)
fig.set_tight_layout(True)
ax.set_title(r'Bigger empty layer')
plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Ice/Intervoxel_outgassing_first_layer_particle_radius.png', dpi=600)
plt.show()


'''lh_a_1 = np.array([4.07023,49.21,53.2167])
lh_b_1 = np.array([-2484.986,-2008.01,-795.104])
lh_c_1 = np.array([3.56654,-16.4542,-22.3452])
lh_d_1 = np.array([-0.00320981,0.0194151,0.0529476])


def p_sub(T, target, idx):
    print(10 ** (lh_a_1[idx] + lh_b_1[idx] / T + lh_c_1[idx] * np.log10(T) + lh_d_1[idx] * T))
    return 10 ** (lh_a_1[idx] + lh_b_1[idx] / T + lh_c_1[idx] * np.log10(T) + lh_d_1[idx] * T) - target

target = 3
index = 0       # 0 = H2O, 1 = CO2, 2 = CO
#val = brentq(p_sub, 200, 250, args=(target))
#print(val)

T = np.linspace(77, 250, 500)
factors = np.sqrt(const.m_H2O/(2 * np.pi * const.k_boltzmann * T))
P = p_sub(T, 0, index)

fig, ax = plt.subplots(1, 1)
ax.plot(T, P * factors)
#plt.scatter(val, 3, marker='x', color='black', label='p(' + str(round(val, 2)) + ' K) = 3 Pa')
#ax.set_xlim(170, 220)
#ax.set_ylim(1E-7, 1E-1)
ax.set_yscale('log')
ax.grid(which='both')
locmaj = LogLocator(base=10,numticks=40)
ax.yaxis.set_major_locator(locmaj)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('mass loss rate (kg/(s*m^2))')
ax.set_title('Mass loss rate of H2O')
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, bottom=True, left=True, labeltop=False, labelright=False, labelbottom=True, labelleft=True)
plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/mass_loss_H2O.png', dpi=600)
plt.show()'''
'''l1 = line.Line2D([val, val], [1E-3, 3], color='black', lw=4)
l2 = line.Line2D([145, val], [3, 3], color='black', lw=4)
for each in [l1, l2]:
    ax.add_artist(each)
    plt.show()'''


'''lh_a_1 = np.array([4.07023,49.21,53.2167])
lh_b_1 = np.array([-2484.986,-2008.01,-795.104])
lh_c_1 = np.array([3.56654,-16.4542,-22.3452])
lh_d_1 = np.array([-0.00320981,0.0194151,0.0529476])


def p_sub(T, target):
    return 10 ** (lh_a_1[0] + lh_b_1[0] / T + lh_c_1[0] * np.log10(T) + lh_d_1[0] * T) - target

target = 3
val = brentq(p_sub, 150, 230, args=(target))
print(val)

T = np.linspace(150, 230, 100)
P = p_sub(T, 0)

fig, ax = plt.subplots(1, 1)
ax.plot(T, P)
plt.scatter(val, 3, marker='x', color='black', label='p(' + str(round(val, 2)) + ' K) = 3 Pa')
ax.set_xlim(150, 230)
ax.set_ylim(1E-3, 20)
ax.set_yscale('log')
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, bottom=True, left=True, labeltop=False, labelright=False, labelbottom=True, labelleft=True)
l1 = line.Line2D([val, val], [1E-3, 3], color='black', lw=0.5)
l2 = line.Line2D([145, val], [3, 3], color='black', lw=0.5)
for each in [l1, l2]:
    ax.add_artist(each)
plt.show()'''

