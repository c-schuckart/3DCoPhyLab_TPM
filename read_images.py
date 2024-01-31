import numpy as np
import csv
import PIL.Image
import matplotlib.pyplot as plt
import constants as const
from numba import njit
from tqdm import tqdm
from pims import ImageSequence
from os import listdir
import json


def calibration_low(OS):
    B = 537.42945148
    R = 122.25819449
    F = 7.56861749
    OFF = 1.95976821
    return B/(np.log((R/(OS-OFF))+F))


def calibration_high(OS):
    B = 1415.3
    R = 13402.6
    F = 2.67783
    OFF = 15.5951
    return B/(np.log((R/(OS-OFF))+F))


@njit
def GCD(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    else:
        while b != 0:
            h = np.mod(a, b)
            a = b
            b = h
        return a

@njit
def convolve(Surface_temperatures, length, size, width, n_x, n_y):
    convolved = np.zeros((n_y, n_x), dtype=np.float64)
    im2_scaled = np.zeros((length * size, length * size), dtype=np.float64)
    factor = length*size//width
    for j in range(0, length * size):
        for k in range(0, length * size):
            im2_scaled[j][k] = Surface_temperatures[j // factor][k // factor]
    for j in range(0, size-2):
        for k in range(0, size-2):
            convolved[j + 1][k + 1] = np.average(im2_scaled[length * j:length * (j + 1), length * k:length * (k + 1)])
    return convolved, im2_scaled


@njit
def replace_values(array, repl_val, goal_val):
    a, b = np.shape(array)
    for i in range(0, a):
        for j in range(0, b):
            if array[i][j] == repl_val:
                array[i][j] = goal_val
    return array


@njit
def calculate_temperature_deltas(k, n_x, n_y, current_surface_temp_scaled, next_surface_temp_scaled, dt, t_deltas):
    surface_temperature_section = np.zeros((k, n_y, n_x), dtype=np.float64)
    for t in range(0, k):
        for j in range(0, n_y):
            for k in range(0, n_x):
                surface_temperature_section[t][j][k] = current_surface_temp_scaled[j][k] + (next_surface_temp_scaled[j][k] - current_surface_temp_scaled[j][k]) * t * dt/t_deltas
    return surface_temperature_section

#@njit
def get_surface_temperatures_csv(n_x, n_y, directory, file_list, current_file, time_cur, current_surface_temp_scaled, dt, next_segment_time, start_up):
    if start_up:
        time_cur = np.datetime64(file_list[current_file][0:4] + '-' + file_list[current_file][5:7] + '-' + file_list[current_file][8:10] + ' ' + file_list[current_file][11:13] + ':' + file_list[current_file][15:17] + ':' + file_list[current_file][19:21])
        current_surface_temp = np.genfromtxt(directory + file_list[current_file] , dtype=np.float64, delimiter=',')
        current_surface_temp = replace_values(current_surface_temp[1:200, 1:200], 200.0, 0.0)
        width, height = np.shape(current_surface_temp)
        ggT = GCD(n_x, width)
        length = width // ggT
        current_surface_temp_scaled = convolve(current_surface_temp, length, n_x-2, width, n_x, n_y)[0]
        print(file_list[current_file])
    current_file += 1
    time_next = np.datetime64(file_list[current_file][0:4] + '-' + file_list[current_file][5:7] + '-' + file_list[current_file][8:10] + ' ' + file_list[current_file][11:13] + ':' + file_list[current_file][15:17] + ':' + file_list[current_file][19:21])
    next_surface_temp = np.genfromtxt(directory + file_list[current_file] , dtype=np.float64, delimiter=',')
    next_surface_temp = replace_values(next_surface_temp[1:200, 1:200], 200.0, 0.0)
    width, height = np.shape(next_surface_temp)
    ggT = GCD(n_x, width)
    length = width // ggT
    next_surface_temp_scaled = convolve(next_surface_temp, length, n_x-2, width, n_x, n_y)[0]
    k = int((time_next.astype(int) - time_cur.astype(int))/dt)
    surface_temperature_section = calculate_temperature_deltas(k, n_x, n_y, current_surface_temp_scaled, next_surface_temp_scaled, dt, (time_next.astype(int) - time_cur.astype(int)))
    next_segment_time += (time_next.astype(int) - time_cur.astype(int))
    return surface_temperature_section, current_file, time_next, next_surface_temp_scaled, next_segment_time

def get_lamp_spot_pixel(image, boundary_value):
    y_min = np.shape(image)[1] + 1
    y_max = -1
    x_min = np.shape(image)[0] + 1
    x_max = -1
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if image[i][j] >= boundary_value:
                if j < y_min:
                    y_min = j
                if j > y_max:
                    y_max = j
                if i < x_min:
                    x_min = i
                if i > x_max:
                    x_max = i
    return np.array([x_min, x_max], np.int32), np.array([y_min, y_max], np.int32)
def image_to_scale(path):
    im = np.array(PIL.Image.open(path).convert('L'))
    x_coords, y_coords = get_lamp_spot_pixel(im, np.max(im)/5)
    center_x = x_coords[0] + (x_coords[1] - x_coords[0])//2
    center_y = y_coords[0] + (y_coords[1] - y_coords[0])//2
    max = np.mean(im[center_x-15:center_x+15, center_y-15:center_y+15])
    #print(np.max(im))
    #max = np.max(im)
    im_scaled = im/max
    max_allowed = np.ones(np.shape(im_scaled))
    #print(np.max(im_scaled))
    im_scaled = np.where(im_scaled < max_allowed, im_scaled, max_allowed)
    #print(np.max(im_scaled))
    x_coords, y_coords = get_lamp_spot_pixel(im_scaled, 0.3)
    if x_coords[1] - x_coords[0] != y_coords[1] - y_coords[0]:
        if (x_coords[1] - x_coords[0]) & 2 != 0:
            x_coords[1] += 1
        if (y_coords[1] - y_coords[0]) & 2 != 0:
            y_coords[1] += 1
        delta = (x_coords[1] - x_coords[0]) - (y_coords[1] - y_coords[0])
        if delta > 0:
            y_coords[0] -= delta//2
            y_coords[1] += delta//2
        elif delta < 0:
            x_coords[0] += delta//2
            x_coords[1] -= delta//2
    lamp_spot = im_scaled[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]
    lamp_to_scale = np.zeros((int(np.ceil((x_coords[1] - x_coords[0]) * 25/7)), int(np.ceil((y_coords[1] - y_coords[0]) * 25/7))), dtype=np.float64)
    x_start = int(np.ceil((x_coords[1] - x_coords[0]) * 9/7))
    y_start = int(np.ceil((y_coords[1] - y_coords[0]) * 9/7))
    lamp_to_scale[x_start:x_start+(x_coords[1] - x_coords[0]), y_start:y_start+(y_coords[1] - y_coords[0])] = lamp_spot
    #print(np.shape(lamp_to_scale))
    #print(x_coords[1]-x_coords[0], y_coords[1]-y_coords[0])
    #plot = plt.imshow(lamp_to_scale)
    #plt.colorbar(plot)
    #plt.show()
    return lamp_to_scale





#path = 'C:/Users/Christian Schuckart/OneDrive/Work/Paper - sand/2023-12-28_11_42_11_875.bmp'
#image_to_scale(path)
'''#im = np.array(PIL.Image.open('D:/Laboratoy_data/IR/screenshots/2023_04_02_00h_55m_33s.png').convert('L'))
im = np.array(PIL.Image.open('D:/Masterarbeit_data/IR/ice_block/2023_02_22_11h_35m_12s.png').convert('L'))
images = ImageSequence('D:/Masterarbeit_data/IR/ice_block/2023_02_*.png')
filenames = listdir('D:/Masterarbeit_data/IR/ice_block')
x=626
y=653
width=183
height=183
ggT = GCD(const.n_x-2, width)
length = width//ggT
Surface_temperatures_cam = np.zeros((const.k, const.n_x, const.n_y), dtype=np.float64)
for i in range(len(images)-1):
    if i == 0:
        im_cur = images[i][int(y-height/2):int(y+height/2), int(x-width/2):int(x+width/2)]
        OS_cur = (im_cur / 255) * 50
        Surface_temperatures_cur = calibration(OS_cur)
        convolved_cur = convolve(Surface_temperatures_cur, length, const.n_x - 2)[0]
        time_cur = np.datetime64(filenames[i][0:4] + '-' + filenames[i][5:7] + '-' + filenames[i][8:10] + ' ' + filenames[i][11:13] + ':' + filenames[i][15:17] + ':' + filenames[i][19:21])
    else:
        im_cur = im_next
        OS_cur = OS_next
        Surface_temperatures_cur = Surface_temperatures_next
        convolved_cur = convolved_next
        time_cur = time_next
    OS_next = (im_next/255)*50
    im_next = images[i + 1][int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    Surface_temperatures_next = calibration(OS_next)
    convolved_next = convolve(Surface_temperatures_next, length, const.n_x - 2)[0]
    time_next = np.datetime64(filenames[i+1][0:4] + '-' + filenames[i+1][5:7] + '-' + filenames[i+1][8:10] + ' ' + filenames[i+1][11:13] + ':' + filenames[i+1][15:17] + ':' + filenames[i+1][19:21])
    for time in range()
        for j in range(0, const.n_y):
            for k in range(0, const.n_x):'''




'''print(len(images))
print(images[0].metadata)
print(len(filenames))
print(filenames[0])
constructed_date = np.datetime64(filenames[0][0:4] + '-' + filenames[0][5:7] + '-' + filenames[0][8:10] + ' ' + filenames[0][11:13] + ':' + filenames[0][15:17] + ':' + filenames[0][19:21])
images[0].metadata['Date'] = constructed_date
print(images[0].metadata)'''
'''convolution_width = 160/(const.n_y-2)
adaptive_convolution = np.array([[1, 1, convolution_width-2, 0], [1, 1, convolution_width-2, 0], [convolution_width-2, convolution_width-2, (convolution_width-2)**2, 0], [0, 0, 0, 0]], dtype=np.float64)
start_next = 1
for j in range(0, 611):
    for k in range(0, 61):
        next_grid = np.zeros((4, 4), dtype=np.float64)
        for a in range(0, 4):
            for b in range(0, 4):
                if convolution_width - (b+1) >= 1 and convolution_width - (a+1) >= 1:
                    adaptive_convolution[a][b] = 1
                if convolution_width - (b+1) < 1 and convolution_width - (b+1) >= 1:
                    adaptive_convolution[a][b] = convolution_width - (b+1)
                    start_next = 1 - (convolution_width - (b+1))
                if convolution_width - (a + 1) < 1 and convolution_width - (a + 1) >= 1:
                    adaptive_convolution[a][b] = convolution_width - (a + 1)
                    start_next = 1 - (convolution_width - (a + 1))
                if (convolution_width - (b+1) < 1 and convolution_width - (b+1) >= 1) and (convolution_width - (a + 1) < 1 and convolution_width - (a + 1) >= 1):
                    
                else:
                    adaptive_convolution[a][b] = 0'''

'''current_surface_temp = np.genfromtxt('D:/Laboratory_data/Sand_without_tubes/temp_profile/temp_profile/2023_03_05_18h_53m_55s.csv', dtype=np.float64, delimiter=',')
current_surface_temp = replace_values(current_surface_temp[1:200, 1:200], 200.0, 0.0)
width, height = np.shape(current_surface_temp)
ggT = GCD(const.n_x-2, width)
length = width//ggT
convolved, im2_scaled = convolve(current_surface_temp, length, const.n_x-2, width, const.n_x, const.n_y)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#ax3.imshow(convolved[1:const.n_x-2-1, 1:const.n_x-2-1])
ax3.imshow(convolved)
ax1.imshow(current_surface_temp)
ax2.imshow(im2_scaled)
plt.show()'''

'''with open('D:/TPM_Data/Big_sand/YNEGXPOSsand_L_chamber_A_0.95_Absdepth_0.001_Lambda_0.003.json') as json_file:
    jdata = json.load(json_file)
im = np.array(PIL.Image.open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/2023_08_31_09h_38m_21s.png').convert('L'))
temps = np.array(jdata['Temperature Surface'])
for j in range(0, const.n_y-2):
    for k in range(0, const.n_x-2):
        if temps[j][k] == 0:
            temps[j][k] = np.nan
#temps[(const.n_y-1)//2][(const.n_x-3)] = 400
#im = np.array(PIL.Image.open('D:/Masterarbeit_data/IR/ice_block/2023_02_22_11h_35m_12s.png').convert('L'))
#images = ImageSequence('D:/Masterarbeit_data/IR/ice_block/2023_02_*.png')
#filenames = listdir('D:/Masterarbeit_data/IR/ice_block')
#x=626
#y=653
#width=183
#height=183
width=(929-655)*2+100
height=(929-655)*2+100
x=578
y=655
ggT = GCD(const.n_x, width)
length = width//ggT
Surface_temperatures_cam = np.zeros((const.k, const.n_x, const.n_y), dtype=np.float64)
im_cur = im[int(y-height/2):int(y+height/2), int(x-width/2):int(x+width/2)]
print(np.max(im_cur))
#OS_cur = (im_cur / 255) * 255 + 145
OS_cur = (im_cur / 255) * 50
for i in range(np.shape(OS_cur)[0]):
    for j in range(np.shape(OS_cur)[1]):
        if OS_cur[i][j] >= 50:
            OS_cur[i][j] = np.nan
print(np.max(OS_cur))
#Surface_temperatures_cur = np.rot90(calibration_high(OS_cur), 3)
Surface_temperatures_cur = calibration_low(OS_cur)
for i in range(np.shape(Surface_temperatures_cur)[0]):
    for j in range(np.shape(Surface_temperatures_cur)[1]):
        if Surface_temperatures_cur[i][j] <= 309:
            Surface_temperatures_cur[i][j] = np.nan
convolved_cur = convolve(Surface_temperatures_cur, length, const.n_x, len(Surface_temperatures_cur[0]), const.n_x, const.n_y)[0]
Sur_shifted = np.full(np.shape(Surface_temperatures_cur), np.nan)
Sur_shifted[0:height-48, 0:width] = Surface_temperatures_cur[48:height, 0:width]
Con_shifted = np.full(np.shape(Surface_temperatures_cur), np.nan)
Con_shifted[0:const.n_y-5-1, 0:const.n_x] = convolved_cur[5:const.n_y-1, 0:const.n_x]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_figheight(10)
fig.set_figwidth(15)
im_one = ax1.imshow(Sur_shifted, cmap='viridis')
im_two = ax2.imshow(Con_shifted[1:const.n_y-1, 1:const.n_x-1], cmap='viridis')
im_three = ax3.imshow(temps, cmap='viridis')
plt.colorbar(im_one, ax=ax1, shrink=0.3)
plt.colorbar(im_two, ax=ax2, shrink=0.3)
plt.colorbar(im_three, ax=ax3, shrink=0.3)
ax1.set_title('IR cam')
ax2.set_title('IR cam scaled')
ax3.set_title('Simulation')
for each in [ax1, ax2, ax3]:
    each.set_xticks([])
    each.set_yticks([])
fig, ax = plt.subplots(1, 1)
im_one = ax.imshow(Surface_temperatures_cur, cmap='viridis')
ax.set_title('2023_08_31_09h_38m_21s')
plt.colorbar(im_one, ax=ax, shrink=1.0)
#plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Plots/Comparison_surface_309Kco.png', dpi=600)
plt.show()'''


'''path = 'D:/Laboratory_data/Big_sand/sand_daten2/screenshots/'
filenames = listdir(path)

for each in filenames:
    im = np.array(PIL.Image.open(path + each).convert('L'))
    width = (929 - 655) * 2 + 100
    height = (929 - 655) * 2 + 100
    x = 578
    y = 655
    ggT = GCD(const.n_x, width)
    length = width // ggT
    #Surface_temperatures_cam = np.zeros((const.k, const.n_x, const.n_y), dtype=np.float64)
    im_cur = im[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    OS_cur = (im_cur / 255) * 255 + 145
    #OS_cur = (im_cur / 255) * 50
    # Surface_temperatures_cur = np.rot90(calibration_high(OS_cur), 3)
    Surface_temperatures_cur = calibration_high(OS_cur)
    convolved_cur = convolve(Surface_temperatures_cur, length, const.n_x, len(Surface_temperatures_cur[0]), const.n_x, const.n_y)[0]
    Con_shifted = np.full(np.shape(Surface_temperatures_cur), np.nan)
    Con_shifted[0:const.n_y - 5 - 1, 0:const.n_x - 1 - 1] = convolved_cur[5:const.n_y - 1, 1:const.n_x - 1]
    target = open('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/sand_surface_temperatures_mid.csv', 'a')
    target.write(each + ',' + str(Con_shifted[const.n_y//2][const.n_x//2]) + ',' + str(Con_shifted[const.n_y//2-8][const.n_x//2-8] * 3/4 + Con_shifted[const.n_y//2-7][const.n_x//2-7] * 1/4) + ',' + str(np.mean(Surface_temperatures_cur)) + ',' + str(np.median(Surface_temperatures_cur)) + '\n')
    #target.write(each + ',' + str(np.max(Surface_temperatures_cur)) + ',' + str(np.mean(Surface_temperatures_cur)) + ',' + str(np.median(Surface_temperatures_cur)) + '\n')
    target.close()'''

'''with open('D:/TPM_data/Big_sand/TSand_sim_thesis_heat_cap_840_ambient_300K.json') as json_file:
    jdata = json.load(json_file)
im = np.array(PIL.Image.open('D:/Laboratory_data/Big_sand/sand_daten1/screenshots/2023_07_18_20h_12m_55s.png').convert('L'))
temps = np.array(jdata['Temperature Surface'])
for j in range(0, const.n_y-2):
    for k in range(0, const.n_x-2):
        if temps[j][k] == 0:
            temps[j][k] = np.nan
#temps[(const.n_y-1)//2][(const.n_x-3)] = 400
#im = np.array(PIL.Image.open('D:/Masterarbeit_data/IR/ice_block/2023_02_22_11h_35m_12s.png').convert('L'))
#images = ImageSequence('D:/Masterarbeit_data/IR/ice_block/2023_02_*.png')
#filenames = listdir('D:/Masterarbeit_data/IR/ice_block')
#x=626
#y=653
#width=183
#height=183
width=(929-655)*2+100
height=(929-655)*2+100
x=578
y=655
ggT = GCD(const.n_x, width)
length = width//ggT
Surface_temperatures_cam = np.zeros((const.k, const.n_x, const.n_y), dtype=np.float64)
im_cur = im[int(y-height/2):int(y+height/2), int(x-width/2):int(x+width/2)]
im_cur =np.rot90(im_cur, 3)
print(np.max(im_cur))
OS_cur = (im_cur / 255) * 255 + 145
#OS_cur = (im_cur / 255) * 50
#for i in range(np.shape(OS_cur)[0]):
    #for j in range(np.shape(OS_cur)[1]):
        #if OS_cur[i][j] >= 50:
            #OS_cur[i][j] = np.nan
print(np.max(OS_cur), np.min(OS_cur))
#Surface_temperatures_cur = np.rot90(calibration_high(OS_cur), 3)
Surface_temperatures_cur = calibration_high(OS_cur)
#for i in range(np.shape(Surface_temperatures_cur)[0]):
    #for j in range(np.shape(Surface_temperatures_cur)[1]):
        #if Surface_temperatures_cur[i][j] <= 309:
            #Surface_temperatures_cur[i][j] = np.nan
convolved_cur = convolve(Surface_temperatures_cur, length, const.n_x, len(Surface_temperatures_cur[0]), const.n_x, const.n_y)[0]
Sur_shifted = np.full(np.shape(Surface_temperatures_cur), np.nan)
Sur_shifted[0:height-48, 0:width] = Surface_temperatures_cur[48:height, 0:width]
Con_shifted = np.full(np.shape(Surface_temperatures_cur), np.nan)
Con_shifted[0:const.n_y-5-1, 0:const.n_x-1-1] = convolved_cur[5:const.n_y-1, 1:const.n_x-1]
dx = np.linspace(-13, 13, 52)

fig, ax = plt.subplots(1, 1)
plt.tick_params(axis='both', which='both', direction='in', top=True, labeltop=False, right=True, labelright=False, bottom=True, labelbottom=True, left=True, labelleft=True)
#ax.plot(dx, np.array(jdata['Temperature Surface'], dtype=np.float64)[0:52, const.n_x//2], label='Surface temperature SIM')
#ax.plot(dx, Con_shifted[1:const.n_y-1, const.n_x//2], ls='--', label='Oberflächentemperatur gemessen')
ax.plot(dx, np.array(jdata['Temperature Surface'], dtype=np.float64)[const.n_y//2, 0:52], label='Surface temperature measured')
ax.plot(dx, Con_shifted[const.n_y//2, 1:const.n_x-1], ls='--', label='Surface temperature measured')
ax.set_ylim(290, 390)
ax.set_xlabel('Position (cm)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Surface temperature slice x-axis - ambient temp. 300K')
plt.legend(fontsize='x-small')
ax.grid(True, lw=0.5)
plt.savefig('C:/Users/Christian Schuckart/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Plots/Slice_surface_temp_x_300K_eng.png', dpi=600)
plt.show()'''

'''Con_shifted[const.n_y//2, 0:const.n_x] = np.nan
T = np.array(jdata['Temperature Surface'], dtype=np.float64)
T[const.n_y//2, 0:52] = np.nan
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_figheight(10)
fig.set_figwidth(15)
im_one = ax1.imshow(Sur_shifted, cmap='viridis')
im_two = ax2.imshow(Con_shifted[1:const.n_y-1, 1:const.n_x-1], cmap='viridis')
im_three = ax3.imshow(T, cmap='viridis')
plt.colorbar(im_one, ax=ax1, shrink=0.3)
plt.colorbar(im_two, ax=ax2, shrink=0.3)
plt.colorbar(im_three, ax=ax3, shrink=0.3)
ax1.set_title('IR cam')
ax2.set_title('IR cam scaled')
ax3.set_title('Simulation')
for each in [ax1, ax2, ax3]:
    each.set_xticks([])
    each.set_yticks([])
#fig, ax = plt.subplots(1, 1)
#im_one = ax.imshow(Surface_temperatures_cur, cmap='viridis')
#ax.set_title('2023_08_31_09h_38m_21s')
#plt.colorbar(im_one, ax=ax, shrink=1.0)
#plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/BIG_sand/Plots/Comparison_surface_309Kco.png', dpi=600)
plt.show()'''