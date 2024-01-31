import numpy as np
from numba import njit, prange

@njit
def generate_topography(surface, reduced_surface, dx, dy, dz):
    polygon_number = 0
    for each in reduced_surface:
        polygon_number += np.sum(surface[each[2]][each[1]][each[0]])
    polygon_list = np.zeros((polygon_number, 2, 3), dtype=np.float64)
    counter = 0
    for each in reduced_surface:
        if surface[each[2]][each[1]][each[0]][0] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([0, 0, dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / (5E-3)**2], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][1] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([0, 0, - dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / (5E-3)**2], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][2] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([0, dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / (5E-3)**2, 0], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][3] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([0, - dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / (5E-3)**2, 0], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][4] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / (5E-3)**2, 0, 0], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][5] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([- dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / (5E-3)**2, 0, 0], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
    return polygon_list


@njit
def get_temperature_vector(n_x, n_y, n_z, temperature, surface, surface_reduced, length, view_factor_matrix, sigma, epsilon, albedo, lamp_power, dx, dy, dz):
    surface_temperature_vector = np.zeros(length, dtype=np.float64)
    lamp_power_vector = np.zeros(length, dtype=np.float64)
    thermal_heating_energy = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    counter_polygons = 0
    areas = np.zeros(length, dtype=np.float64)
    for each in surface_reduced:
        for i in range(len(surface[each[2]][each[1]][each[0]])):
            if (i == 0 or i == 1) and surface[each[2]][each[1]][each[0]][i] != 0:
                areas[counter_polygons] = dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]]
                counter_polygons += 1
            elif (i == 2 or i == 3) and surface[each[2]][each[1]][each[0]][i] != 0:
                areas[counter_polygons] = dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]]
                counter_polygons += 1
            elif (i == 4 or i == 5) and surface[each[2]][each[1]][each[0]][i] != 0:
                areas[counter_polygons] = dz[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]]
                counter_polygons += 1
    counter_polygons = 0
    for each in surface_reduced:
        for i in range(0, np.sum(surface[each[2]][each[1]][each[0]])):
            surface_temperature_vector[counter_polygons] = temperature[each[2]][each[1]][each[0]]**4
            if np.sum(temperature[0:each[2], each[1], each[0]]) == 0:
                lamp_power_vector[counter_polygons] = lamp_power[each[2]][each[1]][each[0]]     # Constructing a vector where only the actual surface elements that see the light source directly are listed with their incoming energy
            counter_polygons += 1
    counter_polygons = 0
    for each in surface_reduced:
        for i in range(0, np.sum(surface[each[2]][each[1]][each[0]])):
            thermal_heating_energy[each[2]][each[1]][each[0]] += (1 - albedo) * (sigma * epsilon * areas[counter_polygons] * (np.sum(view_factor_matrix[counter_polygons] * surface_temperature_vector) - view_factor_matrix[counter_polygons][counter_polygons] * surface_temperature_vector[counter_polygons]) + (albedo) * (np.sum(view_factor_matrix[counter_polygons] * lamp_power_vector) - view_factor_matrix[counter_polygons][counter_polygons] * lamp_power_vector[counter_polygons]))
            counter_polygons += 1
            '''if np.isnan(thermal_heating_energy[each[2]][each[1]][each[0]]):
                print(np.sum(view_factor_matrix[counter_polygons-1]))
                print(lamp_power_vector, surface_temperature_vector, view_factor_matrix[counter_polygons-1])
                break'''
    return thermal_heating_energy, surface_temperature_vector


@njit
def intersect_plane(ray_origin, ray_direction, plane_point, plane_normal):
    if np.dot(ray_direction, plane_normal) == 0:
        return np.infty
    else:
        d = np.dot((plane_point - ray_origin), plane_normal) / np.dot(ray_direction, plane_normal) #d is only the distance if ray_direction is normalised
    if d < 0:
        return np.infty
    p = ray_origin + ray_direction * d #Point of intersection between ray and infinite plane
    L = np.linalg.norm(p - plane_point, ord=np.inf)
    if L > 0.5:
        return np.infty
    else:
        return d


@njit
def trace_rays(polygon_list):
    view_factor_matrix = np.zeros((len(polygon_list), len(polygon_list)), dtype=np.float64)
    for i in range(0, len(polygon_list)):
        ray_origin = polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])
        for j in range(i+1, len(polygon_list)):
            distances = np.full((len(polygon_list)), np.infty, dtype=np.float64)
            ray_direction = ((polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1])) - (polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1]))) / np.linalg.norm((polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])) - (polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1]))) #Normalized ray direction
            if np.dot(ray_direction, polygon_list[i][1]) < 0:
                pass
            else:
                for a in range(0, len(polygon_list)):
                    plane_point = polygon_list[a][0] + 1/2 * np.sign(polygon_list[a][1])
                    plane_normal = polygon_list[a][1]
                    d = intersect_plane(ray_origin, ray_direction, plane_point, plane_normal)
                    if d == 0:
                        pass
                    else:
                        distances[a] = d
            if round(np.min(distances), 8) == round(np.linalg.norm((polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])) - (polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1]))), 8):
                view_factor_matrix[i][j] = np.dot(polygon_list[i][1], ray_direction)/(np.linalg.norm(polygon_list[i][1]) * np.linalg.norm(ray_direction)) * np.abs(np.dot(polygon_list[j][1], ray_direction)/(np.linalg.norm(polygon_list[j][1]) * np.linalg.norm(ray_direction))) / ((np.pi * np.linalg.norm((polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])) - (polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1])))**2)) * np.linalg.norm(polygon_list[j][1])
                #print(np.dot(polygon_list[i][1], ray_direction)/(np.linalg.norm(polygon_list[i][1]) * np.linalg.norm(ray_direction)), np.abs(np.dot(polygon_list[j][1], ray_direction)/(np.linalg.norm(polygon_list[j][1]) * np.linalg.norm(ray_direction))), np.linalg.norm((polygon_list[i][0] + 1/2 * polygon_list[i][1]) - (polygon_list[j][0] + 1/2 * polygon_list[j][1])), np.linalg.norm(polygon_list[j][1]))
                view_factor_matrix[j][i] = view_factor_matrix[i][j] * np.linalg.norm(polygon_list[i][1]) / np.linalg.norm(polygon_list[j][1])
            else:
                view_factor_matrix[i][j] = 0
                view_factor_matrix[j][i] = view_factor_matrix[i][j]
    return view_factor_matrix


@njit
def trace_rays_MC(polygon_list, iterations, omit_surface, surface):
    view_factor_matrix = np.zeros((len(polygon_list), len(polygon_list)), dtype=np.float64)
    for i in range(0, len(polygon_list)):
        #print(polygon_list[i])
        if omit_surface and int(polygon_list[i][0][2]) == 1 and np.sum(surface[int(polygon_list[i][0][2])][int(polygon_list[i][0][1])][int(polygon_list[i][0][0])]) == 1:
            pass
        else:
            ray_origin = polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])
            for j in range(i+1, len(polygon_list)):
                distances = np.full((len(polygon_list)), np.infty, dtype=np.float64)
                ray_direction = ((polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1])) - (polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1]))) / np.linalg.norm((polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])) - (polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1]))) #Normalized ray direction
                if np.dot(ray_direction, polygon_list[i][1]) < 0:
                    pass
                else:
                    for a in range(0, len(polygon_list)):
                        plane_point = polygon_list[a][0] + 1/2 * np.sign(polygon_list[a][1])
                        plane_normal = polygon_list[a][1]
                        d = intersect_plane(ray_origin, ray_direction, plane_point, plane_normal)
                        if d == 0:
                            pass
                        else:
                            distances[a] = d
                #print(ray_origin, (polygon_list[j][0] + 1/2 * polygon_list[j][1]))
                if round(np.min(distances), 8) == round(np.linalg.norm((polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])) - (polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1]))), 8):
                    delta = np.infty
                    view_factor_points = np.zeros(iterations, dtype=np.float64)
                    for a in range(0, iterations):
                        ran_o1, ran_o2, ran_o3, ran_t1, ran_t2, ran_t3 = np.random.rand(6) - 0.5
                        ray_origin_hit = polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1]) + np.array([((1 - np.abs(np.sign(polygon_list[i][1][0]))) * ran_o1), ((1 - np.abs(np.sign(polygon_list[i][1][1]))) * ran_o2), ((1 - np.abs(np.sign(polygon_list[i][1][2]))) * ran_o3)], dtype=np.float64)
                        point = polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1]) + np.array([((1 - np.abs(np.sign(polygon_list[j][1][0]))) * ran_t1), ((1 - np.abs(np.sign(polygon_list[j][1][1]))) * ran_t2), ((1 - np.abs(np.sign(polygon_list[j][1][2]))) * ran_t3)], dtype=np.float64)
                        ray_direction_hit = (point - ray_origin_hit) / (np.linalg.norm(point - ray_origin_hit))
                        #print(ray_origin_hit, ray_direction_hit, point, polygon_list[j][1])
                        d = intersect_plane(ray_origin_hit, ray_direction_hit, point, polygon_list[j][1])
                        #print(d)
                        view_factor_points[a] = np.dot(polygon_list[i][1], ray_direction_hit)/(np.linalg.norm(polygon_list[i][1]) * np.linalg.norm(ray_direction_hit)) * np.abs(np.dot(polygon_list[j][1], ray_direction_hit)/(np.linalg.norm(polygon_list[j][1]) * np.linalg.norm(ray_direction_hit))) / (np.pi * d**2) * np.linalg.norm(polygon_list[j][1])
                    #print(i, j, view_factor_points)
                    view_factor_matrix[i][j] = np.sum(view_factor_points)/iterations
                    view_factor_matrix[j][i] = view_factor_matrix[i][j] * np.linalg.norm(polygon_list[i][1]) / np.linalg.norm(polygon_list[j][1])
                else:
                    view_factor_matrix[i][j] = 0
                    view_factor_matrix[j][i] = view_factor_matrix[i][j]
    return view_factor_matrix


@njit
def trace_rays_MC_partial_shadowing(polygon_list, iterations):
    view_factor_matrix = np.zeros((len(polygon_list), len(polygon_list)), dtype=np.float64)
    for i in range(0, len(polygon_list)):
        print(i)
        for j in range(i+1, len(polygon_list)):
            #print(ray_origin, (polygon_list[j][0] + 1/2 * polygon_list[j][1]))
                delta = np.infty
                view_factor_points = np.zeros(iterations, dtype=np.float64)
                for a in range(0, iterations):
                    ran_o1, ran_o2, ran_o3, ran_t1, ran_t2, ran_t3 = np.random.rand(6) - 0.5
                    ray_origin_hit = polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1]) + np.array([((1 - np.abs(np.sign(polygon_list[i][1][0]))) * ran_o1), ((1 - np.abs(np.sign(polygon_list[i][1][1]))) * ran_o2), ((1 - np.abs(np.sign(polygon_list[i][1][2]))) * ran_o3)], dtype=np.float64)
                    point = polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1]) + np.array([((1 - np.abs(np.sign(polygon_list[j][1][0]))) * ran_t1), ((1 - np.abs(np.sign(polygon_list[j][1][1]))) * ran_t2), ((1 - np.abs(np.sign(polygon_list[j][1][2]))) * ran_t3)], dtype=np.float64)
                    ray_direction_hit = (point - ray_origin_hit) / (np.linalg.norm(point - ray_origin_hit))
                    #print(ray_origin_hit, ray_direction_hit, point, polygon_list[j][1])
                    d_direct = intersect_plane(ray_origin_hit, ray_direction_hit, point, polygon_list[j][1])
                    distances = np.full((len(polygon_list)), np.infty, dtype=np.float64)
                    if np.dot(ray_direction_hit, polygon_list[i][1]) < 0:
                        pass
                    else:
                        for b in range(0, len(polygon_list)):
                            plane_point = polygon_list[b][0] + 1/2 * np.sign(polygon_list[b][1])
                            plane_normal = polygon_list[b][1]
                            d = intersect_plane(ray_origin_hit, ray_direction_hit, plane_point, plane_normal)
                            if d == 0:
                                pass
                            else:
                                distances[b] = d
                    #print(d)
                    if round(np.min(distances), 8) == round(d_direct, 8):
                        view_factor_points[a] = np.dot(polygon_list[i][1], ray_direction_hit)/(np.linalg.norm(polygon_list[i][1]) * np.linalg.norm(ray_direction_hit)) * np.abs(np.dot(polygon_list[j][1], ray_direction_hit)/(np.linalg.norm(polygon_list[j][1]) * np.linalg.norm(ray_direction_hit))) / (np.pi * d_direct**2) * np.linalg.norm(polygon_list[j][1])
                    else:
                        view_factor_points[a] = 0
                #print(i, j, view_factor_points)
                view_factor_matrix[i][j] = np.sum(view_factor_points)/iterations
                view_factor_matrix[j][i] = view_factor_matrix[i][j] * np.linalg.norm(polygon_list[i][1]) / np.linalg.norm(polygon_list[j][1])
    return view_factor_matrix


@njit
def analytical_viewfactors_rect_to_rect_90deg(x, y, eta, xi):
    A_1 = (x[1] - x[0]) * (y[1] - y[0])
    F = 0
    for l in range(0, 2):
        for k in range(0, 2):
            for j in range(0, 2):
                for i in range(0, 2):
                    C = np.sqrt(x[i]**2 + xi[l]**2)
                    D = (y[j] - eta[k]) / C
                    F += 1/(2 * np.pi * A_1) * (-1) ** ((i+1) + (j+1) + (k+1) + (l+1)) * ((y[j] - eta[k]) * C * np.arctan(D) - C**2/4 * (1 - D**2) * np.log(C**2 * (1 + D**2)))
                    #print((-1) ** ((i+1) + (j+1) + (k+1) + (l+1)) * ((y[j] - eta[k]) * C * np.arctan(D) - C**2/4 * (1 - D**2) * np.log(C**2 * (1 + D**2))))
    return F


@njit
def analytical_viewfactors_rect_to_rect_180deg(x, y, eta, xi, z):
    A_1 = (x[1] - x[0]) * (y[1] - y[0])
    F = 0
    for l in range(0, 2):
        for k in range(0, 2):
            for j in range(0, 2):
                for i in range(0, 2):
                    u = x[i] - xi[l]
                    v = y[j] - eta[k]
                    p =  np.sqrt(u**2 + z**2)
                    q = np.sqrt(v**2 + z**2)
                    F += 1/(2 * np.pi * A_1) * (-1) ** ((i+1) + (j+1) + (k+1) + (l+1)) * (v * p * np.arctan(v / p) + u * q * np.arctan(u / q) - z**2/2 * np.log(u**2 + v**2 + z**2))
                    #print((-1) ** ((i+1) + (j+1) + (k+1) + (l+1)) * (v * p * np.arctan(v / p) + u * q * np.arctan(u / q) - z**2/2 * np.log(u**2 + v**2 + z**2)))
    return F


@njit
def trace_rays_with_analytical_solution(polygon_list):
    view_factor_matrix = np.zeros((len(polygon_list), len(polygon_list)), dtype=np.float64)
    for i in range(0, len(polygon_list)):
        ray_origin = polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])
        for j in range(i+1, len(polygon_list)):
            distances = np.full((len(polygon_list)), np.infty, dtype=np.float64)
            ray_direction = ((polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1])) - (polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1]))) / np.linalg.norm((polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])) - (polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1]))) #Normalized ray direction
            if np.dot(ray_direction, polygon_list[i][1]) < 0:
                pass
            else:
                for a in range(0, len(polygon_list)):
                    plane_point = polygon_list[a][0] + 1/2 * np.sign(polygon_list[a][1])
                    plane_normal = polygon_list[a][1]
                    d = intersect_plane(ray_origin, ray_direction, plane_point, plane_normal)
                    if d == 0:
                        pass
                    else:
                        distances[a] = d
            if round(np.min(distances), 8) == round(np.linalg.norm((polygon_list[i][0] + 1/2 * np.sign(polygon_list[i][1])) - (polygon_list[j][0] + 1/2 * np.sign(polygon_list[j][1]))), 8):
                local_origin = np.zeros(3, dtype=np.float64)
                deg180 = False
                for val in range(0, 3):
                    if polygon_list[j][1][val] != 0:
                        #local_origin[0] = np.minimum(polygon_list[j][1][val] + 1/2 * np.sign(polygon_list[i][1][val]) - polygon_list[i][0][val] - 1 / 2, polygon_list[j][1][val] + 1/2 * np.sign(polygon_list[i][1][val]) - polygon_list[i][0][val] + 1 / 2)
                        #x_1 = np.minimum(polygon_list[i][0][val] - 1 / 2 + np.sign(local_origin[0]) * local_origin[0])
                        #x_2 = polygon_list[i][0][val] + 1 / 2 + np.sign(local_origin[0]) * local_origin[0]
                        print(np.abs(polygon_list[j][1][val] + 1/2 * np.sign(polygon_list[j][1][val])), np.abs(polygon_list[i][0][val] - 1 / 2), np.abs(polygon_list[j][1][val] + 1/2 * np.sign(polygon_list[j][1][val])), np.abs(polygon_list[i][0][val] + 1 / 2))
                        x_1 = np.minimum(np.abs(polygon_list[j][1][val] + 1/2 * np.sign(polygon_list[j][1][val])) - np.abs(polygon_list[i][0][val] - 1 / 2), np.abs(polygon_list[j][1][val] + 1/2 * np.sign(polygon_list[j][1][val])) - np.abs(polygon_list[i][0][val] + 1 / 2))
                        x_2 = x_1 + 1
                    if polygon_list[i][1][val] != 0:
                        #local_origin[2] = np.minimum(polygon_list[i][0][val] - 1 / 2, polygon_list[j][0][val] - 1 / 2)
                        #xi_1 = (polygon_list[j][0][val] - 1 / 2 + np.sign(local_origin[2]) * local_origin[2])
                        #xi_2 = polygon_list[j][0][val] + 1 / 2 + np.sign(local_origin[2]) * local_origin[2]
                        xi_1 = np.minimum(np.abs(polygon_list[i][1][val] + 1/2 * np.sign(polygon_list[i][1][val])) - np.abs(polygon_list[j][0][val] - 1 / 2), np.abs(polygon_list[i][1][val] + 1/2 * np.sign(polygon_list[i][1][val])) - np.abs(polygon_list[j][0][val] + 1 / 2))
                        xi_2 = xi_1 + 1
                    if polygon_list[i][1][val] == 0 and polygon_list[j][1][val] == 0:
                        local_origin[1] = np.minimum(polygon_list[i][0][val] - 1 / 2, polygon_list[j][0][val] - 1 / 2)
                        y_1 = polygon_list[i][0][val] - 1 / 2 + np.sign(local_origin[1]) * local_origin[1]
                        y_2 = polygon_list[i][0][val] + 1 / 2 + np.sign(local_origin[1]) * local_origin[1]
                        eta_1 = polygon_list[j][0][val] - 1 / 2 + np.sign(local_origin[1]) * local_origin[1]
                        eta_2 = polygon_list[j][0][val] + 1 / 2 + np.sign(local_origin[1]) * local_origin[1]
                        #y_1 = np.min(np.array([np.abs(polygon_list[i][1][val] + 1/2) - np.abs(polygon_list[j][0][val] - 1 / 2), np.abs(polygon_list[i][1][val] - 1/2) - np.abs(polygon_list[j][0][val] - 1 / 2), np.abs(polygon_list[i][1][val] + 1/2 * np.sign(polygon_list[i][1][val])) - np.abs(polygon_list[j][0][val] + 1 / 2)]))
                        #y_2 = polygon_list[i][0][val] + 1 / 2 + np.sign(local_origin[1]) * local_origin[1]
                        #eta_1 = polygon_list[j][0][val] - 1 / 2 + np.sign(local_origin[1]) * local_origin[1]
                        #eta_2 = polygon_list[j][0][val] + 1 / 2 + np.sign(local_origin[1]) * local_origin[1]
                    if polygon_list[i][1][val] != 0 and polygon_list[j][1][val] != 0:
                        deg180 = True
                        z = (polygon_list[i][0][val] + 1/2 * np.sign(polygon_list[i][1][val])) - (polygon_list[j][0][val] + 1/2 * np.sign(polygon_list[j][1][val]))
                        local_origin[0] = np.minimum(polygon_list[i][0][(val + 1) % 3] - 1 / 2, polygon_list[j][0][(val + 1) % 3] - 1 / 2)
                        local_origin[1] = np.minimum(polygon_list[i][0][(val + 2) % 3] - 1 / 2, polygon_list[j][0][(val + 2) % 3] - 1 / 2)
                        x_1 = polygon_list[i][0][(val + 1) % 3] - 1 / 2
                        x_2 = polygon_list[i][0][(val + 1) % 3] + 1 / 2
                        xi_1 = polygon_list[j][0][(val + 1) % 3] - 1 / 2
                        xi_2 = polygon_list[j][0][(val + 1) % 3] + 1 / 2
                        y_1 = polygon_list[i][0][(val + 2) % 3] - 1 / 2
                        y_2 = polygon_list[i][0][(val + 2) % 3] + 1 / 2
                        eta_1 = polygon_list[j][0][(val + 2) % 3] - 1 / 2
                        eta_2 = polygon_list[j][0][(val + 2) % 3] + 1 / 2
                print(local_origin)
                print(polygon_list[i][0], polygon_list[j][0])
                print(np.array([np.maximum(x_1, 1E-6), x_2], dtype=np.float64), np.array([y_1, y_2], dtype=np.float64), np.array([eta_1, eta_2], dtype=np.float64), np.array([np.maximum(xi_1, 1E-6), xi_2], dtype=np.float64))
                if deg180:
                    view_factor_matrix[i][j] = analytical_viewfactors_rect_to_rect_180deg(np.array([x_1, x_2], dtype=np.float64), np.array([y_1, y_2], dtype=np.float64), np.array([eta_1, eta_2], dtype=np.float64), np.array([xi_1, xi_2], dtype=np.float64), z)
                else:
                    view_factor_matrix[i][j] = analytical_viewfactors_rect_to_rect_90deg(np.array([np.maximum(x_1, 1E-6), x_2], dtype=np.float64), np.array([y_1, y_2], dtype=np.float64), np.array([eta_1, eta_2], dtype=np.float64), np.array([np.maximum(xi_1, 1E-6), xi_2], dtype=np.float64))
                #print(np.dot(polygon_list[i][1], ray_direction)/(np.linalg.norm(polygon_list[i][1]) * np.linalg.norm(ray_direction)), np.abs(np.dot(polygon_list[j][1], ray_direction)/(np.linalg.norm(polygon_list[j][1]) * np.linalg.norm(ray_direction))), np.linalg.norm((polygon_list[i][0] + 1/2 * polygon_list[i][1]) - (polygon_list[j][0] + 1/2 * polygon_list[j][1])), np.linalg.norm(polygon_list[j][1]))
                view_factor_matrix[j][i] = view_factor_matrix[i][j]
            else:
                view_factor_matrix[i][j] = 0
                view_factor_matrix[j][i] = view_factor_matrix[i][j]
    return view_factor_matrix


print(analytical_viewfactors_rect_to_rect_90deg(np.array([1E-6, 1], dtype=np.float64), np.array([0, 1], dtype=np.float64), np.array([0, 1], dtype=np.float64), np.array([1E-6, 1], dtype=np.float64)))

print(analytical_viewfactors_rect_to_rect_90deg(np.array([1, 2], dtype=np.float64), np.array([0, 1], dtype=np.float64), np.array([0, 1], dtype=np.float64), np.array([1E-6, 1], dtype=np.float64)))

#print(analytical_viewfactors_rect_to_rect(np.array([1E-6, 1], dtype=np.float64), np.array([10.5, 11.5], dtype=np.float64), np.array([10.5, 11.5], dtype=np.float64), np.array([1E-6, 1], dtype=np.float64)))

#print(analytical_viewfactors_rect_to_rect(np.array([-1.5, -0.5], dtype=np.float64), np.array([0, 1], dtype=np.float64), np.array([0, 1], dtype=np.float64), np.array([1E-6, 1], dtype=np.float64)))

#print(analytical_viewfactors_rect_to_rect(np.array([-0.5, 0.5], dtype=np.float64), np.array([0, 1], dtype=np.float64), np.array([0, 1], dtype=np.float64), np.array([1E-6, 1], dtype=np.float64)))