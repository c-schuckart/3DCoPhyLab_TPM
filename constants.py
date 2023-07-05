import numpy as np

#Definition of Simulation Parameters
min_dx = 5E-6                      # Layer Thickness of smallest layer in x direction, Numerical Resolution  [m]
min_dy = 5E-6                      # Layer Thickness of smallest layer in y direction, Numerical Resolution  [m]
min_dz = 5E-6                      # Layer Thickness of smallest layer in z direction, Numerical Resolution  [m]
dt = 0.5E-5                           # Timestep                               [s]
n_x = 25                            # Number of Layers                       [-]
n_y = 25
n_z = 25
'''min_dx = 2E-3                      # Layer Thickness of the smallest layer in x direction, Numerical Resolution  [m]
min_dy = 2E-3                      # Layer Thickness of the smallest layer in y direction, Numerical Resolution  [m]
min_dz = 2E-3                      # Layer Thickness of the smallest layer in z direction, Numerical Resolution  [m]
dt = 0.1                           # Timestep                               [s]
n_x = 39                           # Number of Layers                       [-]
n_y = 39
n_z = 53'''
n = np.array([n_x, n_y, n_z])
k = 36000 * 1                      # Number of Timesteps                    [-]
#k = 130320//4                     # Number of Timesteps                    [-]
#Definition of Physical Parameters
#Material Properties
lambda_constant = 1E-2            # Thermal Conductivity                   [W/(K m)]
#lambda_constant = 100000            # Thermal Conductivity                   [W/(K m)]
lambda_scaling_factor = 1
r_mono = 2.5E-6                     # Radius of Monomeres                    [m]
e_1 = 1.34                        # Parameter for mean free path           [-]
#VFF_pack_const = 1              # Volume Filling Factor of Packing       [-]
VFF_pack_const = 0.62              # Volume Filling Factor of Packing       [-]
poisson_ratio_par = 0.31          # Poisson`s ratio of Particle            [-]
young_modulus_par = 10.5E9        # Young`s modulus of Particle            [Pa]
tortuosity = 1.10                 # Tortuosity factor for monodisperse spherical particles for VFF 0.2
#surface_energy_par = 0.014          # specific surface energy of Particle    [J/m^2]
surface_energy_par = 0.2         # specific surface energy of Particle    [J/m^2]
molecular_volume_H2O = 2E-5
packing_geometry_factor = np.pi/2       #pi/2 for simple cubic, pi/3 for fcc packing
f_1 = 5.18E-2                     # empirical constant for Packing Structure Factor 1 [-]
f_2 = 5.26                        # empirical constant for Packing Structure Factor 2 [-]
m_H2O = 2.99E-26                  # mass of a water molecule               [kg]
m_CO2 = 7.31E-26
b = 1 * (2 * r_mono)                          # Permeability Coefficient               [-]
co2_h2o_ratio_global = 0      # Percentage of CO2 ice content of total ice content [-]
density_water_ice = 930           # Density if water ice at around 90K     [kg/m^3]
density_co2_ice = 1600
density_sand = 1500
density_sample_holder = 2698.9  # Density of aluminium, the sample holder material  [kg/m^3]
density_copper = 8960                                                  # [kg/m^3]
molar_mass_water = 18.015E-3    # Molar mass of water                    [kg/mol]
molar_mass_co2 = 44.010E-3
heat_capacity_water_ice = 1610      # Heat Capacity of water ice             [J/(kg * K)]
heat_capacity_co2_ice = 850       # Heat Capacity of CO2 ice               [J/(kg * K)]
heat_capacity_sand = 830
heat_capacity_sample_holder = 900   #Heat capacity of aluminium, the sample holder material [J/(kg*K)]
heat_capacity_copper = 386      # Heat capacity of copper              [J/(kg * K)]
latent_heat_water = 2.86E6      # Latent heat of water ice               [J/kg]
latent_heat_co2 = 0.57E6          # Latent heat of CO2 ice                 [J/kg]
lh_a_1 = np.array([4.07023,49.21,53.2167])
lh_b_1 = np.array([-2484.986,-2008.01,-795.104])
lh_c_1 = np.array([3.56654,-16.4542,-22.3452])
lh_d_1 = np.array([-0.00320981,0.0194151,0.0529476])
m_mol = np.array([1.8E-2,4.4E-2,2.8E-2])        #[kg/mol]
depth_dependent_strength = 3      # Parameter used to calculate the tensile strength [Pa]
const_tensile_strength = 0.0045       # Tensile strength of the comet material [Pa]
x_0 = 5E-2                        # Length scaling factor used to calculate the tensile strength [m]
gravitational_pressure = 0        #Placeholder! [Pa]
surface_reduction_factor = 1
#Thermal Properties
temperature_ini = 77             # Start Temperature                      [K]
sample_holder_starting_temp = 110  # Starting temperature of the sample holder [K]
Input_Intensity = 6500             # Intensity of the Light Source (Sun)    [W/m^2]
epsilon = 1                       # Emissivity                             [-]
albedo = 0.80                  # Albedo                                 [-]
lambda_water_ice = 651            # thermal conductivity of water ice      [W/(m * T)], depending on T!
#lambda_water_ice = 567
lambda_co2_ice = 0.02             # thermal conductivity of water ice      [W/(m * K)]
lambda_sample_holder = 236        # thermal conductivity of aluminium, the sample holder material   [W/(m*K)]
lambda_copper = 401               # thermal conductivity of copper      [W/(m * K)]
lambda_sand = 0.0074 * 0.2
a_H2O = 3.23E12                   # Sublimation Pressure Coefficient of water [Pa]
b_H2O = 6134.6                     # Sublimation Pressure Coefficient of water [K]
a_CO2 = 1.32E12                   # Sublimation Pressure Coefficient of water [Pa]
b_CO2 = 3167.8                    # Sublimation Pressure Coefficient of water [K]
#Illumination Condition/Celestial Mechanics
r_H = 1                           # Lamp Distance                  		   [m]
wire_cross_section = 0.08096E-6   # Cross-section of the AWG 28 wire      [m^2]
wire_length = 0.30                # Length of the sensor wires from the plug to the sensor  [m]
temperature_plug = 295            # Temperature of the plug


#Constants
sigma = 5.67E-8                   # Stefan-Boltzmann Constant              [W/(m^2 K^4)]
solar_constant = 1367             # Solar Constant                         [W/m^2]
#solar_constant = 1600            # Solar Constant                         [W/m^2]
k_boltzmann = 1.38E-23            # Boltzmann's Constant                   [m^2 kg / (s^2 K)]
avogadro_constant = 6.022E23      # Avogadro constant                      [1/mol]
R = 8.31447                         # Gas constant                            [J/(mol * K)]
Phi_Asaeda = 2.18                        # Geometrical constant factor from Asaeda et al. 1973
Phi_Guettler = 13/6
var_lamp_profile = 9/4 * 1E-3            # Variance of the gaussian distribution of the lamp profile   [m]
