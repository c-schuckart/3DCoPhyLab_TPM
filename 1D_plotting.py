import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoLocator, AutoMinorLocator, MultipleLocator, LogLocator, LogFormatterMathtext)
import matplotlib.lines as mlines
import constants as const
from scipy import interpolate
from data_input import read_temperature_data,getPath

'''file_name = 'Check_Knudsen_regime'
#path = 'C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/' + file_name + '.png'
path = 'C:/Users/Christian/Documents/Masterarbeit/Plots/' + file_name + '.png'

def Knudsen_number(T):
    p = 10 ** (const.lh_a_1[0] + const.lh_b_1[0] / T + const.lh_c_1[0] * np.log10(T) + const.lh_d_1[0] * T)
    return (3 * np.sqrt(np.pi * const.R * T / (2 * const.m_mol[0])) * (1 - (1 - const.VFF_pack_const)) / (1 - const.VFF_pack_const) * 1 / (p * const.r_mono * 2) * 0.499 / (np.sqrt(2) * np.pi * (2.8E-10) ** 2) * np.sqrt(8 * const.m_H2O * const.k_boltzmann * T / np.pi)), p


temperature = np.linspace(115, 275, 300)
Kn, P = Knudsen_number(temperature)

f = interpolate.interp1d(temperature, P)
g = interpolate.interp1d(P, temperature)
print(P)
print(Knudsen_number(240)[1], np.interp(240, temperature, P))
print(Knudsen_number(115)[1], np.interp(115, temperature, P))
x_ticks = [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]
x_labels = [r'$10^{-8}$', r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', r'$10^{-0}$', r'$10^{2}$']

def forward(x):
    return np.interp(x, temperature, P)


def inverse(x):
    return np.interp(x, P, temperature)


fig, ax = plt.subplots()
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Kn (1)')
ax.tick_params(axis='x', which='both', direction='in', top=False, labeltop=False)
ax.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
ax.plot(temperature, Kn)
#ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.set_yscale('log')
secax = ax.secondary_xaxis('top', functions=(forward, inverse))
#secax.xaxis.set_minor_locator(AutoMinorLocator())
#secax.xaxis.set_major_locator(LogLocator())
#secax.xaxis.set_major_formatter(LogFormatterMathtext())
secax.set_xlabel('Subl. pressure (Pa)')
#secax.set_xscale('log')
secax.set_xticks(x_ticks, x_labels)
secax.tick_params(axis='x', which='both', direction='in')
ax.add_artist(mlines.Line2D([100, 290], [10, 10], lw=0.8, ls='--', alpha=1, color='grey'))
temp_l = 250
temp_r = 265
temp = (temp_r + temp_l) / 2
while np.abs(10 - np.interp(temp, temperature,Kn)) > 0.001:
    current = np.interp(temp, temperature, Kn)
    if 10 - current > 0:
        temp_r = temp
        temp = (temp + temp_l)/2
    if 10 - current < 0:
        temp_l = temp
        temp = (temp + temp_r)/2
print(temp, Knudsen_number(temp)[1], forward(temp))
ax.add_artist(mlines.Line2D([temp, temp], [1E-1, 10E15], lw=0.8, ls='--', alpha=1, color='grey'))
ax.set_xlim(115, 275)
ax.set_ylim(5E-1, 1E14)

plt.title('Knudsen number')
#plt.legend()
#plt.show()
#plt.savefig('Constant_lambda_test_sand_' + str(const.lambda_sand) + '.png', dpi=600)
plt.savefig(path, dpi=600)'''

'''file_name = 'Diffusion_velocity'
path = 'C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/' + file_name + '.png'

def Diffsuion_coefficient(T):
    return const.R * T / (1 - const.VFF_pack_const) * 1 / np.sqrt(2 * np.pi * const.m_mol[0] * const.R * T) * (1 - const.VFF_pack_const) ** 2 * 2 * const.r_mono / (3 * (1 - (1 - const.VFF_pack_const))) * 4 / (2.18 * const.tortuosity)

L = 2/3 * (1 -const.VFF_pack_const) / (1 - (1 - const.VFF_pack_const)) * 2 * const.r_mono

temperature = np.linspace(120, 270, 300)
D = Diffsuion_coefficient(temperature)
print(D * (1 - const.VFF_pack_const) / (const.R * temperature))
v = D/L

fig, ax = plt.subplots()
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Diffusion velocity (m/s)')
ax.tick_params(axis='x', which='both', direction='in', top=False, labeltop=False)
ax.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
ax.plot(temperature, v)

plt.title('Diffusion velocity')
#plt.legend()
plt.show()
#plt.savefig('Constant_lambda_test_sand_' + str(const.lambda_sand) + '.png', dpi=600)
#plt.savefig(path, dpi=600)
'''

data = np.genfromtxt("D:/Masterarbeit_data/sensor_temp_sand_bigger_dot.csv", delimiter=",")
time_sim = [const.dt * i for i in range(0, 35940)]

time_deltas_data_interior, temp_10mm, temp_20mm, temp_35mm, temp_55mm, temp_90mm = read_temperature_data(getPath(), '2023-02-15 16:45:00', '2023-02-15 17:45:01', [1, 2, 3, 4, 5], [], [], [], [], [])
time_data = [np.sum(time_deltas_data_interior[0:i+1]).astype(int) for i in range(len(time_deltas_data_interior)-1)]

print(len(time_data), len(temp_10mm))
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
plt.scatter(time_data, temp_10mm[:-1], label='10mm sensor', s=1, color='black')
plt.scatter(time_data, temp_20mm[:-1], label='20mm sensor', s=1, color='black', marker='x')
plt.scatter(time_data, temp_35mm[:-1], label='35mm sensor', s=1, color='black', marker='d')
plt.scatter(time_data, temp_55mm[:-1], label='55mm sensor', s=1, color='black', marker='+')
plt.scatter(time_data, temp_90mm[:-1], label='90mm sensor', s=1, color='black', marker='1')
plt.plot(time_sim, data[0], label='10mm simulation')
plt.plot(time_sim, data[1], label='20mm simulation')
plt.plot(time_sim, data[2], label='35mm simulation')
plt.plot(time_sim, data[3], label='55mm simulation')
plt.plot(time_sim, data[4], label='90mm simulation')
#plt.xlim(6000, 6200)
#plt.title('Lambda linear ' + str(const.lambda_a) + '*T + ' + str(const.lambda_b))
#plt.title('Lambda constant = ' + str(const.lambda_constant) + r', $b_{\eta}$ = ' + str(const.b)
plt.title('Lambda constant = ' + str(const.lambda_sand))
plt.legend()
#plt.show()
#plt.savefig('Constant_lambda_test_sand_' + str(const.lambda_sand) + '.png', dpi=600)
plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Plots/Sand_surface_bigger_dot_r_20mm (1).png', dpi=600)
#plt.savefig('linear_lambda_test_sand_' + str(const.lambda_a) + 'T + ' + str(const.lambda_b) + '.png', dpi=600)