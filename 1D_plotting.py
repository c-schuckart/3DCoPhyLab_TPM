import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoLocator, AutoMinorLocator, MultipleLocator, LogLocator, LogFormatterMathtext)
import matplotlib.lines as mlines
import json
import constants as const
from scipy import interpolate
from data_input import read_temperature_data, getPath

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
albedos = [567, 325.5, 162.75]
for Lambda in albedos:
    with open('D:/Masterarbeit_data/L_chamber_albedo/Albedo_0.80_Lambda_wi_' + str(Lambda) + '_lamp_test.json') as json_file:
        data_outgassing = json.load(json_file)

    outgassed_mass = np.sum(data_outgassing['Outgassing rate'])*const.dt/1.565*1000
    json_file.close()

    data = np.genfromtxt('D:/Masterarbeit_data/L_chamber_albedo/sensor_temp_lamp_test_albedo_0.80_Lambda_wi' + str(Lambda) +  '.csv', delimiter=",")
    time_sim = [const.dt * i for i in range(0, 1124800)]

    time_deltas_data_interior, temp_10mm, temp_20mm, temp_35mm, temp_55mm, temp_90mm = read_temperature_data('D:/Laboratoy_data/temps_ice.txt', '2023-02-22 11:27:02', '2023-02-22 13:00:56', [1, 2, 3, 4, 5], [], [], [], [], [])
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
    plt.ylim(65, 220)
    #plt.title('Lambda linear ' + str(const.lambda_a) + '*T + ' + str(const.lambda_b))
    #plt.title('Lambda constant = ' + str(const.lambda_constant) + r', $b_{\eta}$ = ' + str(const.b)
    plt.title('Albedo: 0.8 - Lambda: ' + str(Lambda) + '/T - Outgassed mass: ' + str(round(outgassed_mass, 3)) + ' g/h')
    plt.legend()
    #plt.show()
    plt.savefig('C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/Albedo_0.80_Lambda_wi_' + str(Lambda) + '_ice_block.png', dpi=600)
    plt.clf()
    #plt.savefig('C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/Sand_surface_1D_TPM.png', dpi=600)
    #plt.savefig('linear_lambda_test_sand_' + str(const.lambda_a) + 'T + ' + str(const.lambda_b) + '.png', dpi=600)


'''path_list_1 = ['D:/Masterarbeit_data/Albedo_0.9_lambda_scale_1_VFF_0.62.json', 'D:/Masterarbeit_data/Albedo_0.9_lambda_scale_2_VFF_0.62.json']
path_list_2 = ['D:/Masterarbeit_data/Albedo_0.85_lambda_scale_1_VFF_0.62.json', 'D:/Masterarbeit_data/Albedo_0.85_lambda_scale_2_VFF_0.62.json']
path_list_3 = ['D:/Masterarbeit_data/Albedo_0.8_lambda_scale_1_VFF_0.62.json', 'D:/Masterarbeit_data/Albedo_0.8_lambda_scale_2_VFF_0.62.json']
with open(getPath()) as json_file:
#with open('test.json') as json_file:
    data_vis = json.load(json_file)
temperature_data = np.zeros((2, const.n_y, const.n_x), dtype=np.float64)
for a in range(len(path_list_1)):
    with open(path_list_1[a]) as json_file:
        data = json.load(json_file)
    surface_temp = np.zeros((const.n_y, const.n_x), dtype=np.float64)
    temp = np.array(data['Temperature'][len(data['Temperature'])-1])
    for i in range(0, const.n_z):
        for j in range(0, const.n_y):
            for k in range(0, const.n_x):
                if temp[i][j][k] > 0 and surface_temp[j][k] == 0:
                    surface_temp[j][k] = temp[i][j][k]
    temperature_data[a] = surface_temp
    print(np.sum([data['Outgassing rate'][b] * const.dt for b in range(len(data['Outgassing rate']))]))
    json_file.close()
reduced_surface_temp = np.zeros(2, dtype=np.float64)
for a in range(len(reduced_surface_temp)):
    ts = []
    for j in range(0, const.n_y):
        for k in range(0, const.n_x):
            if surface_temp[j][k] != 0 and surface_temp[j][k] != 110:
                ts.append(temperature_data[a][j][k])
    reduced_surface_temp[a] = np.average(ts)
max_temps = np.zeros(2, dtype=np.float64)
avrg_temp = np.zeros(2, dtype=np.float64)
for i in range(0, 2):
    max_temps[i] = np.max(temperature_data[i])
    avrg_temp[i] = np.average(reduced_surface_temp[i])
with open('lamp_input_S_chamber.json') as json_file:
    data_lamp = json.load(json_file)
reduced_temp = np.zeros(2, dtype=np.float64)
for a in range(len(reduced_temp)):
    t = []
    for j in range(0, const.n_y):
        for k in range(0, const.n_x):
            if data_lamp['Lamp Power'][1][j][k] > 0.00159:
                t.append(temperature_data[a][j][k])
    reduced_temp[a] = np.average(t)
    print(len(t))
lambda_factor = [1, 2]
plt.xlabel('Lambda factor')
plt.ylabel('Temperature (K)')
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
plt.scatter(lambda_factor, max_temps, label='Maximum surf. temp', s=10, color='red')
plt.scatter(lambda_factor, avrg_temp, label='Avrg. surf. temp.', s=10, color='blue', marker='x')
plt.scatter(lambda_factor, reduced_temp, label='Avrg. surf. temp. within lamp circle', s=10, color='black', marker='d')
#plt.xlim(6000, 6200)
#plt.title('Lambda linear ' + str(const.lambda_a) + '*T + ' + str(const.lambda_b))
#plt.title('Lambda constant = ' + str(const.lambda_constant) + r', $b_{\eta}$ = ' + str(const.b)
plt.title('Surface temps. after 1h - Albedo 0.9 - Temp. dep. lambda')
plt.legend()
#plt.show()
#plt.savefig('Constant_lambda_test_sand_' + str(const.lambda_sand) + '.png', dpi=600)
plt.savefig('C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/Surface_temperature_albedo_0.9.png', dpi=600)
#plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Plots/Surface_temperature_albedo_0.8.png', dpi=600)'''

'''path_list_1 = ['D:/Masterarbeit_data/Albedo_0.90_surface_corr_factor_0.10.json', 'D:/Masterarbeit_data/Albedo_0.89_surface_corr_factor_0.10.json', 'D:/Masterarbeit_data/Albedo_0.88_surface_corr_factor_0.10.json', 'D:/Masterarbeit_data/Albedo_0.87_surface_corr_factor_0.10.json', 'D:/Masterarbeit_data/Albedo_0.86_surface_corr_factor_0.10.json', 'D:/Masterarbeit_data/Albedo_0.85_surface_corr_factor_0.10.json', 'D:/Masterarbeit_data/Albedo_0.84_surface_corr_factor_0.10.json']
#path_list_1 = ['D:/Masterarbeit_data/Albedo_0.90_surface_corr_factor_0.10.json', 'D:/Masterarbeit_data/Albedo_0.89_surface_corr_factor_0.10.json', 'D:/Masterarbeit_data/Albedo_0.88_surface_corr_factor_0.10.json']
#path_list_1 = ['D:/Masterarbeit_data/Albedo_0.9_lambda_scale_1_VFF_0.62.json', 'D:/Masterarbeit_data/Albedo_0.85_lambda_scale_1_VFF_0.62.json', 'D:/Masterarbeit_data/Albedo_0.8_lambda_scale_1_VFF_0.62.json', 'D:/Masterarbeit_data/Albedo_0.75_lambda_scale_1_VFF_0.62.json']
outgassed_masses = []
temperature_data = []
for a in range(len(path_list_1)):
    with open(path_list_1[a]) as json_file:
        data = json.load(json_file)
    outgassed_masses.append(np.sum([data['Outgassing rate'][b] * const.dt for b in range(len(data['Outgassing rate']))]) * 1000)
    surface_temp = np.zeros((const.n_y, const.n_x), dtype=np.float64)
    temp = np.array(data['Temperature'][len(data['Temperature']) - 1])
    for i in range(0, const.n_z):
        for j in range(0, const.n_y):
            for k in range(0, const.n_x):
                if temp[i][j][k] > 0 and surface_temp[j][k] == 0:
                    surface_temp[j][k] = temp[i][j][k]
    temperature_data.append(surface_temp)
    json_file.close()
max_temps = np.zeros(len(path_list_1), dtype=np.float64)
for i in range(0, len(path_list_1)):
    max_temps[i] = np.max(temperature_data[i])

albedo_values = [0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84]
plt.xlabel('Albedo')
plt.ylabel('Outgassed mass (g)')
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
plt.scatter(albedo_values, outgassed_masses, s=10, label='Outgassed mass')
#plt.xlim(6000, 6200)
#plt.title('Lambda linear ' + str(const.lambda_a) + '*T + ' + str(const.lambda_b))
#plt.title('Lambda constant = ' + str(const.lambda_constant) + r', $b_{\eta}$ = ' + str(const.b)
plt.title('Activity fraction: 10% - Outgassed mass')
plt.legend()
#plt.show()
#plt.savefig('Constant_lambda_test_sand_' + str(const.lambda_sand) + '.png', dpi=600)
plt.savefig('C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/activity_fraction_0.10_outgassed_mass.png', dpi=600)
#plt.savefig('C:/Users/Christian/OneDrive/Uni/Master/3 - Masterarbeit/Plots/Surface_temperature_albedo_0.8.png', dpi=600)

plt.show()

plt.xlabel('Albedo')
plt.ylabel('Max. surface temperature (K)')
plt.tick_params(axis='x', which='both', direction='in', top=True, labeltop=False)
plt.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
plt.scatter(albedo_values, max_temps, label='Maximum surf. temp', s=10)
#plt.xlim(6000, 6200)
#plt.title('Lambda linear ' + str(const.lambda_a) + '*T + ' + str(const.lambda_b))
#plt.title('Lambda constant = ' + str(const.lambda_constant) + r', $b_{\eta}$ = ' + str(const.b)
plt.title('Activity fraction: 10% - Max. surface temperatures')
plt.legend()
#plt.show()
#plt.savefig('Constant_lambda_test_sand_' + str(const.lambda_sand) + '.png', dpi=600)
plt.savefig('C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/activity_fraction_0.10_max_surface_temp.png', dpi=600)'''