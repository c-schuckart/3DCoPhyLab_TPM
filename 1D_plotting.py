import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoLocator, AutoMinorLocator)
import matplotlib.lines as mlines
import constants as const

'''file_name = 'Check_Knudsen_regime'
path = 'C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/' + file_name + '.png'

def Knudsen_number(T):
    p = 10 ** (const.lh_a_1[0] + const.lh_b_1[0] / T + const.lh_c_1[0] * np.log10(T) + const.lh_d_1[0] * T)
    return (3 * np.sqrt(np.pi * const.R * T / (2 * const.m_mol[0])) * (1 - (1 - const.VFF_pack_const)) / (1 - const.VFF_pack_const) * 1 / (p * const.r_mono * 2) * 0.499 / (np.sqrt(2) * np.pi * (2.8E-10) ** 2) * np.sqrt(8 * const.m_H2O * const.k_boltzmann * T / np.pi)), p


temperature = np.linspace(120, 270, 300)
Kn, P = Knudsen_number(temperature)


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
ax.set_yscale('log')
secax = ax.secondary_xaxis('top', functions=(forward, inverse))
secax.xaxis.set_minor_locator(AutoMinorLocator())
secax.set_xlabel('Subl. pressure (Pa)')
secax.set_xscale('log')
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
ax.add_artist(mlines.Line2D([temp, temp], [1E-1, 10E15], lw=0.8, ls='--', alpha=1, color='grey'))
ax.set_xlim(115, 275)
ax.set_ylim(5E-1, 1E14)

plt.title('Knudsen number')
#plt.legend()
#plt.show()
#plt.savefig('Constant_lambda_test_sand_' + str(const.lambda_sand) + '.png', dpi=600)
plt.savefig(path, dpi=600)'''

file_name = 'Diffusion_velocity'
path = 'C:/Users/Christian Schuckart/Documents/Masterarbeit/Plots/' + file_name + '.png'

def Diffsuion_coefficient(T):
    return const.R * T / (1 - const.VFF_pack_const) * 1 / np.sqrt(2 * np.pi * const.m_mol[0] * const.R * T) * (1 - const.VFF_pack_const) ** 2 * 2 * const.r_mono / (3 * (1 - (1 - const.VFF_pack_const))) * 4 / (2.18 * const.tortuosity)

L = 2/3 * (1 -const.VFF_pack_const) / (1 - (1 - const.VFF_pack_const)) * 2 * const.r_mono

temperature = np.linspace(120, 270, 300)
D = Diffsuion_coefficient(temperature)
v = D/L

fig, ax = plt.subplots()
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Diffusion velocity (m/s)')
ax.tick_params(axis='x', which='both', direction='in', top=False, labeltop=False)
ax.tick_params(axis='y', which='both', direction='in', right=True, labelright=False)
ax.plot(temperature, v)

plt.title('Diffusion velocity')
#plt.legend()
#plt.show()
#plt.savefig('Constant_lambda_test_sand_' + str(const.lambda_sand) + '.png', dpi=600)
plt.savefig(path, dpi=600)

