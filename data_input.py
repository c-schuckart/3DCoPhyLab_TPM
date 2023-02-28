import numpy as np
import csv
from tkinter import *
from tkinter import filedialog

def getPath():
    root = Tk()
    root.withdraw()
    InputPath = filedialog.askopenfilename(title="Select File To Upload")
    root.destroy()
    return InputPath

def read_temperature_data(path, timestamp_start, timestamp_end, positions, *args):
    time_deltas = []
    with open(path) as csvdatei:
        dat = csv.reader(csvdatei)
        bool = False
        for each in dat:
            if bool:
                time_deltas.append((np.datetime64(each[0]) - current_date))
                for i in range(len(args)):
                    args[i].append(float(each[positions[i]]))
                current_date = np.datetime64(each[0])
            if each[0] == timestamp_start:
                bool = True
                current_date = np.datetime64(each[0])
                time_deltas.append(0)
                for i in range(len(args)):
                    args[i].append(float(each[positions[i]]))
            if each[0] == timestamp_end:
                break
    return time_deltas, *args


def transform_temperature_data(k, dt, time_deltas, *args):
    div = (len(args)) // 2
    max_k = k
    counter = 0
    for i in range(0, k):
        for j in range(0, div):
            args[j].append((args[j+div][counter + 1] - args[j+div][counter]) * (i * dt - np.sum(time_deltas[0:counter].astype(int))) / time_deltas[counter + 1].astype(int) + args[j+div][counter])
        if ((i+1) * dt - np.sum(time_deltas[0:counter].astype(int))) / time_deltas[counter + 1].astype(int) > 1:
            counter += 1
        if counter + 1 == len(time_deltas):
            max_k = i
            break
    return max_k, *args[0:div]

