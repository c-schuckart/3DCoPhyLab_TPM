import numpy as np
from numba import njit, prange

@njit
def generate_topology(surface, reduced_surface, dx, dy, dz):
    polygon_number = 0
    for each in reduced_surface:
        polygon_number += np.sum(surface[each[2]][each[1]][each[0]])
    polygon_list = np.zeros((polygon_number, 2, 3), dtype=np.float64)
    counter = 0
    for each in reduced_surface:
        if surface[each[2]][each[1]][each[0]][0] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([0, 0, dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]]], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][1] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([0, 0, - dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]]], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][2] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([0, dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]], 0], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][3] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([0, - dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]], 0], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][4] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]], 0, 0], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
        if surface[each[2]][each[1]][each[0]][5] == 1:
            polygon_list[counter][0], polygon_list[counter][1] = np.array([each[0], each[1], each[2]], dtype=np.float64), np.array([- dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]], 0, 0], dtype=np.float64) #middle point of plane and normal vector
            counter += 1
    return polygon_list

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
    if L >= 0.5:
        return np.infty
    else:
        return d


@njit
def trace_rays(polygon_list):
    view_factor_matrix = np.zeros((len(polygon_list), len(polygon_list)), dtype=np.float64)
    for i in range(0, len(polygon_list)):
        ray_origin = polygon_list[i][0] + 1/2 * polygon_list[i][1]
        for j in range(i+1, len(polygon_list)):
            distances = np.full((len(polygon_list)), np.infty, dtype=np.float64)
            ray_direction = ((polygon_list[j][0] + + 1/2 * polygon_list[j][1]) - (polygon_list[i][0] + 1/2 * polygon_list[i][1])) / np.linalg.norm((polygon_list[i][0] + 1/2 * polygon_list[i][1]) - (polygon_list[j][0] + 1/2 * polygon_list[j][1])) #Normalized ray direction
            for a in range(0, len(polygon_list)):
                plane_point = polygon_list[a][0] + 1/2 * polygon_list[a][1]
                plane_normal = polygon_list[a][1]
                d = intersect_plane(ray_origin, ray_direction, plane_point, plane_normal)
                if d == 0:
                    pass
                else:
                    distances[a] = d
            if round(np.min(distances), 8) == round(np.linalg.norm((polygon_list[i][0] + 1/2 * polygon_list[i][1]) - (polygon_list[j][0] + 1/2 * polygon_list[j][1])), 8):
                view_factor_matrix[i][j] = np.dot(polygon_list[i][1], ray_direction)/(np.linalg.norm(polygon_list[i][1]) * np.linalg.norm(ray_direction)) * np.abs(np.dot(polygon_list[j][1], ray_direction)/(np.linalg.norm(polygon_list[j][1]) * np.linalg.norm(ray_direction))) / (np.pi * np.linalg.norm(polygon_list[i][0] +1/2 * polygon_list[i][1] - polygon_list[j][0] + 1/2 * polygon_list[j][1]) * np.linalg.norm(polygon_list[j][1]))
                view_factor_matrix[j][i] = view_factor_matrix[i][j]
            else:
                view_factor_matrix[i][j] = 0
                view_factor_matrix[j][i] = view_factor_matrix[i][j]
    return view_factor_matrix


