import numpy as np
import constants as const
import settings as sett

surface = np.zeros((const.n_z, const.n_y, const.n_x, 6), dtype=np.int32)
heat_capacity = np.full((const.n_z, const.n_y, const.n_x), const.heat_capacity_water_ice, dtype=np.float64)
delta_T = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
density = np.full((const.n_z, const.n_y, const.n_x), const.density_water_ice, dtype=np.float64)
water_content_per_layer = np.full((const.n_z, const.n_y, const.n_x), const.min_dx * const.min_dy * const.min_dz * const.density_water_ice * (1/(const.co2_h2o_ratio_global + 1)), dtype=np.float64)
co2_content_per_layer = np.full((const.n_z, const.n_y, const.n_x), const.min_dx * const.min_dy * const.min_dz * const.density_co2_ice * (const.co2_h2o_ratio_global/(const.co2_h2o_ratio_global + 1)), dtype=np.float64)
h2o_mass_fraction_per_layer = np.full((const.n_z, const.n_y, const.n_x), (1/(const.co2_h2o_ratio_global + 1)), dtype=np.float64)
co2_mass_fraction_per_layer = np.full((const.n_z, const.n_y, const.n_x), (const.co2_h2o_ratio_global/(const.co2_h2o_ratio_global + 1)), dtype=np.float64)
E_conservation = np.zeros(const.k, dtype=np.float64)
Energy_Increase_Total_per_time_Step_arr = np.zeros(const.k, dtype=np.float64)
E_Rad_arr = np.zeros(const.k, dtype=np.float64)
Latent_Heat_per_time_step_arr = np.zeros(const.k, dtype=np.float64)
E_In_arr = np.zeros(const.k, dtype=np.float64)
Fourier_number = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
VFF_pack = np.full((const.n_z, const.n_y, const.n_x), const.VFF_pack_const, dtype=np.float64)
n_z_lr = np.array([-1, 1, 0, 0, 0, 0], dtype=np.int32)
n_y_lr = np.array([0, 0, 1, -1, 0, 0], dtype=np.int32)
n_x_lr = np.array([0, 0, 0, 0, 1, -1], dtype=np.int32)
