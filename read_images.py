import numpy as np
import csv
import PIL.Image
import matplotlib.pyplot as plt
import constants as const


def calibration(OS):
    B = 537.42945148
    R = 122.25819449
    F = 7.56861749
    OFF = 1.95976821
    return B/(np.log((R/(OS-OFF))+F))


def euclid(a, b):
    if a == 0:
        return b
    else:
        while b != 0:
            if a > b:
                a = a - b
            else:
                b = b - a
        return a


im = np.array(PIL.Image.open('D:/Laboratoy_data/IR/screenshots/2023_04_02_00h_55m_33s.png').convert('L'))
x=626
y=653
width=160
height=160
im2 = im[int(y-height/2):int(y+height/2), int(x-width/2):int(x+width/2)]

OS = (im2/255)*50
Surface_temperatures = calibration(OS)

temp = np.array()

print(euclid(const.n_x, 160))



plt.imshow(Surface_temperatures)
plt.show()

