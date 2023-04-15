import numpy as np
import csv
import PIL.Image
import matplotlib.pyplot as plt
import constants as const
from numba import njit
from tqdm import tqdm
from pims import ImageSequence
from os import listdir


def calibration(OS):
    B = 537.42945148
    R = 122.25819449
    F = 7.56861749
    OFF = 1.95976821
    return B/(np.log((R/(OS-OFF))+F))


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

#@njit
def convolve(Surface_temperatures, length, size, width):
    convolved = np.zeros((const.n_y, const.n_x), dtype=np.float64)
    im2_scaled = np.zeros((length * size, length * size), dtype=np.float64)
    factor = length*size//width
    for j in range(0, length * size):
        for k in range(0, length * size):
            im2_scaled[j][k] = Surface_temperatures[j // factor][k // factor]
    for j in range(0, size):
        for k in range(0, size):
            convolved[j + 1][k + 1] = np.average(im2_scaled[length * j:length * (j + 1), length * k:length * (k + 1)])
    return convolved, im2_scaled


#im = np.array(PIL.Image.open('D:/Laboratoy_data/IR/screenshots/2023_04_02_00h_55m_33s.png').convert('L'))
'''im = np.array(PIL.Image.open('D:/Masterarbeit_data/IR/ice_block/2023_02_22_11h_35m_12s.png').convert('L'))
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
#convolved, im2_scaled = convolve(Surface_temperatures, length, const.n_x-2)
'''fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax3.imshow(convolved[1:const.n_x-2-1, 1:const.n_x-2-1])
ax1.imshow(Surface_temperatures)
ax2.imshow(im2_scaled)
plt.show()'''

