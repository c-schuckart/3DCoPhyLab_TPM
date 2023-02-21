import numpy as np

#Definition of Simulation Parameters
min_dx = 10E-3                      # Layer Thickness of smallest layer in x direction, Numerical Resolution  [m]
min_dy = 10E-3                      # Layer Thickness of smallest layer in y direction, Numerical Resolution  [m]
min_dz = 10E-3                      # Layer Thickness of smallest layer in z direction, Numerical Resolution  [m]
dt = 0.085                           # Timestep                               [s]
n_x = 51                            # Number of Layers                       [-]
n_y = 51
n_z = 51
n = np.array([n_x, n_y, n_z])
k = 10000                      # Number of Timesteps                    [-]
total_depth = 0.2                # Total simulated depth                  [m]

#Definition of Physical Parameters
#Material Properties
lambda_constant = 1E-2            # Thermal Conductivity                   [W/(K m)]
#lambda_constant = 100000            # Thermal Conductivity                   [W/(K m)]
r_mono = 2.5E-6                     # Radius of Monomeres                    [m]
e_1 = 1.34                        # Parameter for mean free path           [-]
VFF_pack_const = 0.2              # Volume Filling Factor of Packing       [-]
poisson_ratio_par = 0.17          # Poisson`s ratio of Particle            [-]
young_modulus_par = 5.5E10        # Young`s modulus of Particle            [Pa]
#surface_energy_par = 0.014          # specific surface energy of Particle    [J/m^2]
surface_energy_par = 0.2         # specific surface energy of Particle    [J/m^2]
f_1 = 5.18E-2                     # empirical constant for Packing Structure Factor 1 [-]
f_2 = 5.26                        # empirical constant for Packing Structure Factor 2 [-]
m_H2O = 2.99E-26                  # mass of a water molecule               [kg]
m_CO2 = 7.31E-26
b = 1 * (2 * r_mono)                          # Permeability Coefficient               [-]
co2_h2o_ratio_global = 0      # Percentage of CO2 ice content of total ice content [-]
density_water_ice = 810           # Density if water ice at around 90K     [kg/m^3]
density_co2_ice = 1600
density_sample_holder = 2698.9  # Density of aluminium, the sample holder material  [kg/m^3]
molar_mass_water = 18.015E-3    # Molar mass of water                    [kg/mol]
molar_mass_co2 = 44.010E-3
heat_capacity_water_ice = 1610      # Heat Capacity of water ice             [J/(kg * K)]
heat_capacity_co2_ice = 850       # Heat Capacity of CO2 ice               [J/(kg * K)]
heat_capacity_sample_holder = 900   #Heat capacity of aluminium, the sample holder material [J/(kg*K)]
latent_heat_water = 2.86E6      # Latent heat of water ice               [J/kg]
latent_heat_co2 = 0.57E6          # Latent heat of CO2 ice                 [J/kg]
lh_a_1 = np.asarray([4.07023,49.21,53.2167])
lh_b_1 = np.asarray([-2484.986,-2008.01,-795.104])
lh_c_1 = np.asarray([3.56654,-16.4542,-22.3452])
lh_d_1 = np.asarray([-0.00320981,0.0194151,0.0529476])
m_mol = np.asarray([1.8E-2,4.4E-2,2.8E-2])        #[kg/mol]
depth_dependent_strength = 3      # Parameter used to calculate the tensile strength [Pa]
const_tensile_strength = 0.0045       # Tensile strength of the comet material [Pa]
x_0 = 5E-2                        # Length scaling factor used to calculate the tensile strength [m]
gravitational_pressure = 0        #Placeholder! [Pa]
#Thermal Properties
temperature_ini = 140             # Start Temperature                      [K]
Input_Intensity = 6500             # Intensity of the Light Source (Sun)    [W/m^2]
epsilon = 1                       # Emissivity                             [-]
albedo = 0.05                  # Albedo                                 [-]
lambda_water_ice = 651            # thermal conductivity of water ice      [W/(m * T)], depending on T!
lambda_co2_ice = 0.02             # thermal conductivity of water ice      [W/(m * K)], depending on T!
lambda_sample_holder = 210        # thermal conductivity of aluminium, the sample holder material   [W/(m*K)]
a_H2O = 3.23E12                   # Sublimation Pressure Coefficient of water [Pa]
b_H2O = 6134.6                     # Sublimation Pressure Coefficient of water [K]
a_CO2 = 1.32E12                   # Sublimation Pressure Coefficient of water [Pa]
b_CO2 = 3167.8                    # Sublimation Pressure Coefficient of water [K]
#Illumination Condition/Celestial Mechanics
r_H = 1                           # Lamp Distance                  		   [m]

#Constants
sigma = 5.67E-8                   # Stefan-Boltzmann Constant              [W/(m^2 K^4)]
solar_constant = 6500             # Solar Constant                         [W/m^2]
#solar_constant = 1600            # Solar Constant                         [W/m^2]
k_boltzmann = 1.38E-23            # Boltzmann's Constant                   [m^2 kg / (s^2 K)]
avogadro_constant = 6.022E23      # Avogadro constant                      [1/mol]
R = 8.314                         # Gas constant                            [J/(mol * K)]
