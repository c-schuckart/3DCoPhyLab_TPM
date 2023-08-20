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

timestamps = []
sen_1 = []
sen_2 = []
sen_3 = []
sen_4 = []
sen_5 = []
sen_6 = []

with open('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/temps_sandy_randy.txt') as csvdatei:
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