import numpy as np
import json
import variables_and_arrays as var


def save_current_arrays(temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, pressure, pressure_co2, ejection_times, time_passed):
	data = {'Temperature': temperature.tolist(), 'Water content': water_content_per_layer.tolist(), 'CO2 content' :co2_content_per_layer.tolist(),
			'Dust ice ratio': dust_ice_ratio_per_layer.tolist(), 'CO2 H2O ratio': co2_h2o_ratio_per_layer.tolist(), 'Heat capacity': heat_capacity.tolist(),
			'Pressure': pressure.tolist(), 'Pressure CO2': pressure_co2.tolist(), 'Ejections': ejection_times.tolist(),
			'Time passed': time_passed}
	with open('TPM_save_data.json', 'w') as outfile:
		json.dump(data, outfile)
        
def load_from_save():
    with open('TPM_save_data.json') as json_file:
        data = json.load(json_file)
    temperature = np.array(data['Temperature'])
    water_content_per_layer = np.array(data['Water content'])
    co2_content_per_layer = np.array(data['CO2 content'])
    dust_ice_ratio_per_layer = np.array(data['Dust ice ratio'])
    co2_h2o_ratio_per_layer = np.array(data['CO2 H2O ratio'])
    heat_capacity = np.array(data['Heat capacity'])
    pressure = np.array(data['Pressure'])
    pressure_co2 = np.array(data['Pressure CO2'])
    time_passed = data['Time passed']
    return temperature, water_content_per_layer, co2_content_per_layer, dust_ice_ratio_per_layer, co2_h2o_ratio_per_layer, heat_capacity, pressure, pressure_co2, time_passed


def data_store(j, temperature, water_content_per_layer, co2_content_per_layer, outgassing_rate, outgassing_rate_co2, temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save, data_reduction):
    temperature_save[j//data_reduction] = temperature
    #water_content_save[j//data_reduction] = water_content_per_layer
    #co2_content_save[j//data_reduction] = co2_content_per_layer
    #outgassing_save[j] = outgassing_rate
    #outgassing_co2_save[j] = outgassing_rate_co2
    return temperature_save, water_content_save, co2_content_save, outgassing_save, outgassing_co2_save


def data_store_sensors(j, n_x, n_y, n_z, temperature, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, data_reduction):
    sensor_10mm = (temperature[4][n_y//2][n_x//2] + temperature[5][n_y//2][n_x//2])/2
    sensor_20mm = (temperature[9][n_y//2][n_x//2] + temperature[10][n_y//2][n_x//2])/2
    sensor_35mm = temperature[17][n_y//2][n_x//2]
    sensor_55mm = temperature[27][n_y//2][n_x//2]
    sensor_90mm = (temperature[44][n_y//2][n_x//2] + temperature[45][n_y//2][n_x//2])/2
    return sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm


def data_save(temperature_save, water_content_save, outgassing_save, sublimated_mass_save, filename):
    dict = {'Temperature': temperature_save.tolist(), 'Water content': water_content_save.tolist(), 'Outgassing rate': outgassing_save.tolist(), 'Voxel subl. rate': sublimated_mass_save.tolist()}
    with open(filename + '.json', 'w') as outfile:
        json.dump(dict, outfile)


'''def data_save_sensors(temperature_save, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, path, path_2):
    dict = {'Temperature': temperature_save.tolist()}
    with open(path + '.json', 'w') as outfile:
        json.dump(dict, outfile)
    np.savetxt(path_2 + '.csv', np.array([sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm]), delimiter=",")'''

def data_save_sensors(time_stamp, sensor_10mm, sensor_20mm, sensor_35mm, sensor_55mm, sensor_90mm, file):
    file.write(str(time_stamp) + ',' + str(sensor_10mm) + ',' + str(sensor_20mm) + ',' + str(sensor_35mm) + ',' + str(sensor_55mm) + ',' + str(sensor_90mm) + '\n')