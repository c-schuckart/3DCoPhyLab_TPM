import numpy as np
from numba import jit, njit, prange
import scipy
from tri_diag_solve import tridiagonal_matrix_solver

@njit(parallel=True)
def set_inner_matrices(n_x, n_y, n_z, dx, dy, dz, Dr, Lambda, dt, density, heat_capacity, Q_const, Q_lin, temperature, surface_reduced, sample_holder, input_energy, albedo, sigma, epsilon, alpha):
    A_bottom = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_top = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_south = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_north = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_west = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_east = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    A_point = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    q = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    a = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    b = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    c = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    d = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    e = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    f = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_1 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_2 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_3 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_4 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_5 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_6 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_7 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_8 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_9 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_10 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_11 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    phi_12 = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    g = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    h = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    p = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    r = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    s = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    u = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    v = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] != 160 and sample_holder[i][j][k] != 1:
                    A_bottom[i][j][k] = - Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
                    A_top[i][j][k] = - Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
                    A_south[i][j][k] = - Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
                    A_north[i][j][k] = - Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
                    A_west[i][j][k] = - Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
                    A_east[i][j][k] = - Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
                    A_point[i][j][k] = - A_bottom[i][j][k] - A_top[i][j][k] - A_south[i][j][k] - A_north[i][j][k] - A_west[i][j][k] - A_east[i][j][k] + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - Q_lin * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
                    q[i][j][k] = Q_const * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
    #Right now the volume at the surface is not a half volume. I'm not sure if this is effecting anything and it will have to be tested. It would then always require the adaptive mesh algorithm that slices the z-blocks.
    for each in surface_reduced:
        A_bottom[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0]
        A_top[each[2]][each[1]][each[0]] = 0
        A_south[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2]
        A_north[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3]
        A_west[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4]
        A_east[each[2]][each[1]][each[0]] = - Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[5]
        A_point[each[2]][each[1]][each[0]] = - A_bottom[each[2]][each[1]][each[0]] - A_top[each[2]][each[1]][each[0]] - A_south[each[2]][each[1]][each[0]] - A_north[each[2]][each[1]][each[0]] - A_west[each[2]][each[1]][each[0]] - \
                           A_east[each[2]][each[1]][each[0]] + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * \
                           dz[each[2]][each[1]][each[0]] / dt - Q_lin * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - (-4 * sigma * epsilon * temperature[each[2]][each[1]][each[0]]**3)
        q[each[2]][each[1]][each[0]] = Q_const * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * \
                     dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt * temperature[each[2]][each[1]][each[0]] + input_energy * (1 - albedo) + 3 * sigma * epsilon * temperature[each[2]][each[1]][each[0]]**4
    for i in prange(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] != 160:
                    a[i][j][k] = A_bottom[i][j][k] / (1 + alpha * (p[i-1][j][k] - h[i-1][j][k] * (h[i-1][j][k+1] + r[i-1][j][k+1]) - (r[i-1][j][k] - p[i-1][j][k+1] * h[i-1][j][k]) * (h[i-1][j+1][k] + p[i-1][j+1][k] + r[i-1][j+1][k])))
                    b[i][j][k] = - a[i][j][k] * h[i-1][j][k]
                    c[i][j][k] = - a[i][j][k] * r[i-1][j][k] - b[i][j][k] * p[i-1][j][k+1]
                    d[i][j][k] = (A_south[i][j][k] - a[i][j][k] * s[i-1][j][k] + alpha * ((h[i][j-1][k+1] + 2 * s[i][j-1][k+1] + v[i][j-1][k+1]) * b[i][j][k] * s[i-1][j][k+1] - s[i][j][k-1] * (A_west[i][j][k] - a[i][j][k] * u[i-1][j][k])))/(1 + alpha * (2 * s[i][j-1][k] + u[i][j-1][k] - s[i][j][k-1] * p[i][j-1][k] - h[i][j-1][k] * (h[i][j-1][k+1] + 2 * s[i][j-1][k+1] + v[i][j-1][k+1])))
                    e[i][j][k] = - b[i][j][k] * s[i-1][j][k+1] - d[i][j][k] * h[i][j-1][k]
                    f[i][j][k] = (A_west[i][j][k] - a[i][j][k] * u[i-1][j][k] - d[i][j][k] * p[i][j-1][k] - alpha * (a[i][j][k] * p[i-1][j][k] + c[i][j][k] * p[i-1][j+1][k] + d[i][j][k] * u[i][j-1][k]))/(1 + alpha * (2 * p[i][j][k-1] + s[i][j][k-1] + 2 * u[i][j][k-1]))
                    phi_1[i][j][k] = b[i][j][k] * h[i-1][j][k+1]
                    phi_2[i][j][k] = a[i][j][k] * p[i-1][j][k]
                    phi_3[i][j][k] = b[i][j][k] * r[i-1][j][k+1] + c[i][j][k] * h[i-1][j+1][k]
                    phi_4[i][j][k] = c[i][j][k] * p[i-1][j][k]
                    phi_5[i][j][k] = c[i][j][k] * r[i-1][j+1][k]
                    phi_6[i][j][k] = e[i][j][k] * h[i][j-1][k+1]
                    phi_7[i][j][k] = f[i][j][k] * p[i][j][k-1]
                    phi_8[i][j][k] = d[i][j][k] * s[i][j-1][k]
                    phi_9[i][j][k] = e[i][j][k] * s[i][j-1][k+1]
                    phi_10[i][j][k] = d[i][j][k] * u[i][j-1][k] + f[i][j][k] * s[i][j][k-1]
                    phi_11[i][j][k] = e[i][j][k] * v[i][j-1][k+1]
                    phi_12[i][j][k] = f[i][j][k] * u[i][j][k-1]
                    g[i][j][k] = A_point[i][j][k] - a[i][j][k] * v[i-1][j][k] - b[i][j][k] * u[i-1][j][k+1] - c[i][j][k] * s[i-1][j+1][k] - d[i][j][k] * r[i][j-1][k] - e[i][j][k] * p[i][j-1][k+1] - f[i][j][k] * h[i][j][k-1] + alpha * (2 * (phi_1[i][j][k] + phi_2[i][j][k] + phi_3[i][j][k]) + 3 * phi_4[i][j][k] + 2 * (phi_5[i][j][k] + phi_6[i][j][k] + phi_7[i][j][k] + phi_8[i][j][k]) + 3 * phi_9[i][j][k] + 2 * (phi_10[i][j][k] + phi_11[i][j][k] + phi_12[i][j][k]))
                    h[i][j][k] = (A_east[i][j][k] - b[i][j][k] * v[i-1][j][k+1] - e[i][j][k] * r[i][j-1][k+1] - alpha * (2 * phi_1[i][j][k] + phi_3[i][j][k] + 2 * phi_6[i][j][k] + phi_9[i][j][k] + phi_11[i][j][k]))/g[i][j][k]
                    p[i][j][k] = (- c[i][j][k] * u[i-1][j+1][k] - f[i][j][k] * r[i][j][k-1])/g[i][j][k]
                    r[i][j][k] = (A_north[i][j][k] - c[i][j][k] * v[i-1][j+1][k] - alpha * (phi_2[i][j][k] + phi_3[i][j][k] + 2 * phi_4[i][j][k] + 2 * phi_5[i][j][k] + phi_7[i][j][k]))/(g[i][j][k])
                    s[i][j][k] = (- d[i][j][k] * v[i][j-1][k] - e[i][j][k] * u[i][j-1][k+1])/g[i][j][k]
                    u[i][j][k] = (- f[i][j][k] * v[i][j][k-1])/g[i][j][k]
                    v[i][j][k] = (A_top[i][j][k] - alpha * (phi_8[i][j][k] + phi_9[i][j][k] + phi_10[i][j][k] + phi_11[i][j][k] + phi_12[i][j][k]))/g[i][j][k]
    A_bottom = A_bottom.flatten()
    A_top = A_top.flatten()
    A_south = A_south.flatten()
    A_north = A_north.flatten()
    A_west = A_west.flatten()
    A_east = A_east.flatten()
    A_point = A_point.flatten()
    q = q.flatten()
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()
    d = d.flatten()
    e = e.flatten()
    f = f.flatten()
    g = g.flatten()
    h = h.flatten()
    p = p.flatten()
    r = r.flatten()
    s = s.flatten()
    u = u.flatten()
    v = v.flatten()
    L = np.zeros((n_x*n_y*n_z, n_x*n_y*n_z), dtype=np.float64)
    U = np.zeros((n_x*n_y*n_z, n_x*n_y*n_z), dtype=np.float64)
    A = np.zeros((n_x*n_y*n_z, n_x*n_y*n_z), dtype=np.float64)
    for i in prange(0, n_x*n_y*n_z):
        for j in range(0, n_x*n_y*n_z):
            if i == j:
                L[i][j] = g[i]
                U[i][j] = 1
                A[i][j] = A_point[i]
            if i-1 == j:
                L[i][j] = f[j]
                A[i][j] = A_west[j]
            if i-n_x+1 == j:
                L[i][j] = e[j]
            if i-n_x == j:
                L[i][j] = d[j]
                A[i][j] = A_south[j]
            if i-n_x*(n_y-1) == j:
                L[i][j] = c[j]
            if i-n_x*n_y+1 == j:
                L[i][j] = b[j]
            if i-n_x*n_y == j:
                L[i][j] = a[j]
                A[i][j] = A_bottom[j]
            if i+1 == j:
                U[i][j] = h[i]
                A[i][j] = A_east[i]
            if i+n_x-1 == j:
                U[i][j] = p[i]
            if i+n_x == j:
                U[i][j] = r[i]
                A[i][j] = A_north[i]
            if i+n_x*(n_y-1) == j:
                U[i][j] = s[i]
            if i+n_x*n_y-1 == j:
                U[i][j] = u[i]
            if i+n_x*n_y == j:
                U[i][j] = v[i]
                A[i][j] = A_top[i]
    return L, U, q, A


#@njit(parallel=True)
def set_inner_matrices_constant(n_x, n_y, n_z, dx, dy, dz, Dr, Lambda, dt, density, heat_capacity, Q_const, Q_lin, temperature, surface_reduced, sample_holder, input_energy, albedo, sigma, epsilon, alpha):
    A_bottom = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    A_top = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    A_south = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    A_north = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    A_west = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    A_east = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    A_point = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    q = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    a = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    b = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    c = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    d = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    e = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    f = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_1 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_2 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_3 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_4 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_5 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_6 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_7 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_8 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_9 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_10 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_11 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    phi_12 = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    g = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    h = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    p = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    r = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    s = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    u = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    v = np.zeros((n_z+2, n_y+2, n_x+2), dtype=np.float64)
    for i in prange(0, n_z+2):
        for j in range(0, n_y+2):
            for k in range(0, n_x+2):
                #if i < n_z and j < n_y and k < n_x and temperature[i][j][k] == 250:
                    #print(i, j, k)
                if i < n_z and j < n_y and k < n_x and temperature[i][j][k] != 160 and temperature[i][j][k] != 0: #and temperature[i][j][k] != 250:
                #if 1 == 1:
                    A_top[i+1][j+1][k+1] = - Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
                    A_bottom[i+1][j+1][k+1] = - Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
                    A_north[i+1][j+1][k+1] = - Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
                    A_south[i+1][j+1][k+1] = - Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
                    A_east[i+1][j+1][k+1] = - Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
                    A_west[i+1][j+1][k+1] = - Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
                    A_point[i+1][j+1][k+1] = - A_bottom[i+1][j+1][k+1] - A_top[i+1][j+1][k+1] - A_south[i+1][j+1][k+1] - A_north[i+1][j+1][k+1] - A_west[i+1][j+1][k+1] - A_east[i+1][j+1][k+1] + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - Q_lin * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
                    q[i+1][j+1][k+1] = Q_const * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
                else: #sample_holder[i][j][k] == 1:
                    if i < n_z and j < n_y and k < n_x:
                        A_point[i+1][j+1][k+1] = 1
                        q[i+1][j+1][k+1] = temperature[i][j][k]
                    '''else:
                        A_point[i + 1][j + 1][k + 1] = 0
                        q[i + 1][j + 1][k + 1] = 0'''
    #Right now the volume at the surface is not a half volume. I'm not sure if this is effecting anything and it will have to be tested. It would then always require the adaptive mesh algorithm that slices the z-blocks.
    for each in surface_reduced:
        #print(each[2], each[1], each[0])
        A_bottom[each[2]+1][each[1]+1][each[0]+1] = 0
        A_top[each[2]+1][each[1]+1][each[0]+1] = 0
        A_south[each[2]+1][each[1]+1][each[0]+1] = 0
        A_north[each[2]+1][each[1]+1][each[0]+1] = 0
        A_west[each[2]+1][each[1]+1][each[0]+1] = 0
        A_east[each[2]+1][each[1]+1][each[0]+1] = 0
        A_point[each[2]+1][each[1]+1][each[0]+1] = 1
        q[each[2]+1][each[1]+1][each[0]+1] = temperature[each[2]][each[1]][each[0]]
        #q[each[2]][each[1]][each[0]] = 0
    for i in prange(1, n_z+1):
        for j in range(1, n_y+1):
            for k in range(1, n_x+1):
                #if temperature[i][j][k] != 160: #and temperature[i][j][k] != 250) and sample_holder[i][j][k] != 1:
                if 1 == 1:
                    a[i][j][k] = A_bottom[i][j][k] / (1 + alpha * (p[i-1][j][k] - h[i-1][j][k] * (h[i-1][j][k+1] + r[i-1][j][k+1]) - (r[i-1][j][k] - p[i-1][j][k+1] * h[i-1][j][k]) * (h[i-1][j+1][k] + p[i-1][j+1][k] + r[i-1][j+1][k])))
                    b[i][j][k] = - a[i][j][k] * h[i-1][j][k]
                    c[i][j][k] = - a[i][j][k] * r[i-1][j][k] - b[i][j][k] * p[i-1][j][k+1]
                    d[i][j][k] = (A_south[i][j][k] - a[i][j][k] * s[i-1][j][k] + alpha * ((h[i][j-1][k+1] + 2 * s[i][j-1][k+1] + v[i][j-1][k+1]) * b[i][j][k] * s[i-1][j][k+1] - s[i][j][k-1] * (A_west[i][j][k] - a[i][j][k] * u[i-1][j][k])))/(1 + alpha * (2 * s[i][j-1][k] + u[i][j-1][k] - s[i][j][k-1] * p[i][j-1][k] - h[i][j-1][k] * (h[i][j-1][k+1] + 2 * s[i][j-1][k+1] + v[i][j-1][k+1])))
                    e[i][j][k] = - b[i][j][k] * s[i-1][j][k+1] - d[i][j][k] * h[i][j-1][k]
                    f[i][j][k] = (A_west[i][j][k] - a[i][j][k] * u[i-1][j][k] - d[i][j][k] * p[i][j-1][k] - alpha * (a[i][j][k] * p[i-1][j][k] + c[i][j][k] * p[i-1][j+1][k] + d[i][j][k] * u[i][j-1][k]))/(1 + alpha * (2 * p[i][j][k-1] + s[i][j][k-1] + 2 * u[i][j][k-1]))
                    phi_1[i][j][k] = b[i][j][k] * h[i-1][j][k+1]
                    phi_2[i][j][k] = a[i][j][k] * p[i-1][j][k]
                    phi_3[i][j][k] = b[i][j][k] * r[i-1][j][k+1] + c[i][j][k] * h[i-1][j+1][k]
                    phi_4[i][j][k] = c[i][j][k] * p[i-1][j][k]
                    phi_5[i][j][k] = c[i][j][k] * r[i-1][j+1][k]
                    phi_6[i][j][k] = e[i][j][k] * h[i][j-1][k+1]
                    phi_7[i][j][k] = f[i][j][k] * p[i][j][k-1]
                    phi_8[i][j][k] = d[i][j][k] * s[i][j-1][k]
                    phi_9[i][j][k] = e[i][j][k] * s[i][j-1][k+1]
                    phi_10[i][j][k] = d[i][j][k] * u[i][j-1][k] + f[i][j][k] * s[i][j][k-1]
                    phi_11[i][j][k] = e[i][j][k] * v[i][j-1][k+1]
                    phi_12[i][j][k] = f[i][j][k] * u[i][j][k-1]
                    g[i][j][k] = A_point[i][j][k] - a[i][j][k] * v[i-1][j][k] - b[i][j][k] * u[i-1][j][k+1] - c[i][j][k] * s[i-1][j+1][k] - d[i][j][k] * r[i][j-1][k] - e[i][j][k] * p[i][j-1][k+1] - f[i][j][k] * h[i][j][k-1] + alpha * (2 * (phi_1[i][j][k] + phi_2[i][j][k] + phi_3[i][j][k]) + 3 * phi_4[i][j][k] + 2 * (phi_5[i][j][k] + phi_6[i][j][k] + phi_7[i][j][k] + phi_8[i][j][k]) + 3 * phi_9[i][j][k] + 2 * (phi_10[i][j][k] + phi_11[i][j][k] + phi_12[i][j][k]))
                    #print(g[i][j][k], A_point[i][j][k])
                    if g[i][j][k] == 0:
                        h[i][j][k], p[i][j][k], r[i][j][k], s[i][j][k], u[i][j][k], v[i][j][k] = 0, 0, 0, 0, 0, 0
                    else:
                        h[i][j][k] = (A_east[i][j][k] - b[i][j][k] * v[i-1][j][k+1] - e[i][j][k] * r[i][j-1][k+1] - alpha * (2 * phi_1[i][j][k] + phi_3[i][j][k] + 2 * phi_6[i][j][k] + phi_9[i][j][k] + phi_11[i][j][k]))/g[i][j][k]
                        p[i][j][k] = (- c[i][j][k] * u[i-1][j+1][k] - f[i][j][k] * r[i][j][k-1])/g[i][j][k]
                        r[i][j][k] = (A_north[i][j][k] - c[i][j][k] * v[i-1][j+1][k] - alpha * (phi_2[i][j][k] + phi_3[i][j][k] + 2 * phi_4[i][j][k] + 2 * phi_5[i][j][k] + phi_7[i][j][k]))/(g[i][j][k])
                        s[i][j][k] = (- d[i][j][k] * v[i][j-1][k] - e[i][j][k] * u[i][j-1][k+1])/g[i][j][k]
                        u[i][j][k] = (- f[i][j][k] * v[i][j][k-1])/g[i][j][k]
                        v[i][j][k] = (A_top[i][j][k] - alpha * (phi_8[i][j][k] + phi_9[i][j][k] + phi_10[i][j][k] + phi_11[i][j][k] + phi_12[i][j][k]))/g[i][j][k]
    A_bottom = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(A_bottom, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    A_top = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(A_top, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    A_south = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(A_south, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    A_north = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(A_north, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    A_west = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(A_west, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    A_east = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(A_east, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    A_point = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(A_point, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    q = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(q, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    #q = np.delete(np.delete(np.delete(np.delete(q, 0, 1), n_y-1-1, 1), 0, 2), n_x-1-1, 2).flatten()
    a = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(a, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    b = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(b, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    c = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(c, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    d = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(d, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    e = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(e, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    f = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(f, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    g = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(g, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    h = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(h, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    p = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(p, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    r = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(r, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    s = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(s, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    u = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(u, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    v = np.delete(np.delete(np.delete(np.delete(np.delete(np.delete(v, 0, 1), n_y, 1), 0, 2), n_x, 2), 0, 0), n_z, 0).flatten()
    L = np.zeros((n_x*n_y*n_z, n_x*n_y*n_z), dtype=np.float64)
    U = np.zeros((n_x*n_y*n_z, n_x*n_y*n_z), dtype=np.float64)
    A = np.zeros((n_x*n_y*n_z, n_x*n_y*n_z), dtype=np.float64)
    #print('a: ', np.max(a), np.min(a))
    print('A_point: ', np.max(A_point), np.min(A_point))
    print('g: ', np.max(g), np.min(g))
    for i in prange(0, n_x*n_y*n_z):
        for j in range(0, n_x*n_y*n_z):
            if i == j:
                if g[i] == 0:
                    L[i][j] = 1
                else:
                    L[i][j] = g[i]
                U[i][j] = 1
                A[i][j] = A_point[i]
            if i-1 == j:
                L[i][j] = f[j]
                A[i][j] = A_west[j]
            if i-n_x+1 == j:
                L[i][j] = e[j]
            if i-n_x == j:
                L[i][j] = d[j]
                A[i][j] = A_south[j]
            if i-n_x*(n_y-1) == j:
                L[i][j] = c[j]
            if i-n_x*n_y+1 == j:
                L[i][j] = b[j]
            if i-n_x*n_y == j:
                L[i][j] = a[j]
                A[i][j] = A_bottom[j]
            if i == j - 1:
                U[i][j] = h[i]
                A[i][j] = A_east[i]
            if i == j - n_x+1:
                U[i][j] = p[i]
            if i == j - n_x:
                U[i][j] = r[i]
                A[i][j] = A_north[i]
            if i == j - n_x*(n_y-1):
                U[i][j] = s[i]
            if i == j - n_x*n_y+1:
                U[i][j] = u[i]
            if i == j - n_x*n_y:
                U[i][j] = v[i]
                A[i][j] = A_top[i]
    return L, U, q, A


#@njit
def solve_iterative(iterations, temperature, q, A, L, U, n_x, n_y, n_z):
    T = temperature.flatten()
    R = q - np.dot(A, T)
    for i in range(iterations):
        V = scipy.linalg.solve_triangular(L, R, lower=True, unit_diagonal=False)
        delta = scipy.linalg.solve_triangular(U, V, lower=False, unit_diagonal=True)
        T = T + delta
        R = q - np.dot(A, T)
        print(np.max(delta), np.min(delta))
    return T.reshape((n_z, n_y, n_x))


'''@njit
def set_matrices_lhs(n_x, n_y, n_z, Lambda, dx, Dx, density, heat_capacity, dt, S_p, S_p_rad):
    sub_alpha = np.zeros(n_z+1, dtype=np.float64)
    diag = np.zeros(n_z+1, dtype=np.float64)
    sub_gamma = np.zeros(n_z+1, dtype=np.float64)
    for i in range(1, n_z):
        sub_alpha[i] = Lambda[i]/Dx[i]
        sub_gamma[i] = Lambda[i-1]/Dx[i-1]
        diag[i] = sub_alpha[i] + sub_gamma[i] + density[i] * heat_capacity[i] * dx[i] / dt - S_p[i] * dx[i]
    sub_alpha[0] = Lambda[0]/Dx[0]
    diag[0] = sub_alpha[0] + density[0] * heat_capacity[0] * dx[0] / dt - S_p_rad - S_p[0] * dx[0]
    sub_alpha[n_z] = 0
    diag[n_z] = density[n_z] * heat_capacity[n_z] * dx[n_z] / dt
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs(n_x, n_y, n_z, temperature, dx, density, heat_capacity, dt, S_c, S_c_rad, Q):
    rhs = np.zeros(n_z+1, dtype=np.float64)
    for i in range(1, n_z):
        rhs[i] = S_c[i] * dx[i] + density[i] * heat_capacity[i] * dx[i] / dt * temperature[i]
    rhs[0] = Q + S_c_rad + S_c[0] * dx[0] + density[0] * heat_capacity[0] * dx[0] / dt * temperature[0]
    rhs[n_z] = density[n_z] * heat_capacity[n_z] * dx[n_z] / dt * temperature[n_z]
    return rhs'''


'''@njit
def set_matrices_lhs(n_x, n_y, n_z, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, S_p_rad, temperature, sample_holder):
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            for k in range(1, n_x-1):
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] == 0:
                    a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][0]
                    a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][1]
                    a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][2]
                    a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][3]
                    a_e = Lambda[i][j][k][4] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][4]
                    a_w = Lambda[i][j][k][5] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][5]
                    sub_alpha[i][j][k] = - a_t
                    sub_gamma[i][j][k] = - a_b
                    diag[i][j][k] = a_t + a_b + a_n + a_s + a_e + a_w + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
                elif sample_holder[i][j][k] != 0:
                    diag[i][j][k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs(n_x, n_y, n_z, rhs, temperature, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c, S_c_rad, Q):
    for i in range(1, n_z - 1):
        for j in range(1, n_y - 1):
            for k in range(1, n_x - 1):
                if temperature[i][j][k] > 0 and sample_holder[i][j][k] == 0:
                    a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
                    a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
                    a_e = Lambda[i][j][k][4] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
                    a_w = Lambda[i][j][k][5] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
                    rhs[i][j][k] = a_n * temperature[i][j+1][k] + a_s * temperature[i][j-1][k] + a_e * temperature[i][j][k+1] + a_w * temperature[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i]
                elif sample_holder[i][j][k] != 0:
                    rhs[i][j][k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
    return rhs


@njit
def boundary_condition_implicit(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, rhs, S_c, S_p):
    for each in surface_reduced:
        S_c_rad = 3 * epsilon * sigma * temperature[each[2]][each[1]][each[0]]**4 * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * temperature[each[2]][each[1]][each[0]]**3 * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q =  input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[2]][each[1]][each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_t + a_n + a_s + a_e + a_w + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[2]][each[1]][each[0]] = -a_b
        rhs[each[2]][each[1]][each[0]] = a_n * temperature[each[2]][each[1] + 1][each[0]] + a_s * temperature[each[2]][each[1] - 1][each[0]] + a_e * temperature[each[2]][each[1]][each[0] + 1] + a_w * temperature[each[2]][each[1]][each[0] - 1] + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt * temperature[each[2]][each[1]][each[0]]
    return sub_alpha, diag, rhs'''


@njit
def set_matrices_lhs_z_sweep(n_z, j, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[i] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][1]
            a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][3]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][5]
            sub_alpha[i] = - a_t
            sub_gamma[i] = - a_b
            diag[i] = a_t + a_b + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[i] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[i] = 1
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_z_sweep(n_z, j, k, rhs, temperature, latest_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for i in range(0, n_z):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[i] == 0:
            a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
            rhs[i] = a_n * latest_temperature[i][j+1][k] + a_s * latest_temperature[i][j-1][k] + a_e * latest_temperature[i][j][k+1] + a_w * latest_temperature[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_n - a_s - a_e - a_w) * temperature[i][j][k]
        elif sample_holder[i][j][k] != 0:
            rhs[i] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[i] = 0
    return rhs


@njit
def boundary_condition_implicit_z_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p):
    for each in surface_reduced:
        S_c_rad = 3 * epsilon * sigma * temperature[each[2]][each[1]][each[0]]**4 * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * temperature[each[2]][each[1]][each[0]]**3 * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[2]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_t + a_b + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[2]] = -a_t
        sub_gamma[each[2]] = -a_b
        rhs[each[2]] = a_n * latest_temperature[each[2]][each[1] + 1][each[0]] + a_s * latest_temperature[each[2]][each[1] - 1][each[0]] + a_e * latest_temperature[each[2]][each[1]][each[0] + 1] + a_w * latest_temperature[each[2]][each[1]][each[0] - 1] + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_n - a_s - a_e - a_w) * temperature[each[2]][each[1]][each[0]]
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_lhs_y_sweep(n_y, i, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder):
    for j in range(0, n_y):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[j] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][1]
            a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][3]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][5]
            sub_alpha[j] = - a_n
            sub_gamma[j] = - a_s
            diag[j] = a_n + a_s + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[j] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[j] = 1
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_y_sweep(n_y, i, k, rhs, temperature, latest_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for j in range(0, n_y):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[j] == 0:
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][5]
            rhs[j] = a_t * latest_temperature[i+1][j][k] + a_b * latest_temperature[i-1][j][k] + a_e * latest_temperature[i][j][k+1] + a_w * latest_temperature[i][j][k-1] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_e - a_w) * temperature[i][j][k]
        elif sample_holder[i][j][k] != 0:
            rhs[j] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[j] = 0
    return rhs


@njit
def boundary_condition_implicit_y_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p):
    for each in surface_reduced:
        S_c_rad = 3 * epsilon * sigma * temperature[each[2]][each[1]][each[0]]**4 * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * temperature[each[2]][each[1]][each[0]]**3 * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[1]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad +  a_n + a_s + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[1]] = -a_n
        sub_gamma[each[1]] = -a_s
        rhs[each[1]] = a_t * latest_temperature[each[2] + 1][each[1]][each[0]] + a_b * latest_temperature[each[2] - 1][each[1]][each[0]] + a_e * latest_temperature[each[2]][each[1]][each[0] + 1] + a_w * latest_temperature[each[2]][each[1]][each[0] - 1] + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_e - a_w) * temperature[each[2]][each[1]][each[0]]
    return sub_alpha, diag, sub_gamma, rhs


@njit
def set_matrices_lhs_x_sweep(n_x, i, j, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder):
    for k in range(0, n_x):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and diag[k] == 0:   #The diag == 0 condition is needed for cases where there is an 'interior' top boundary condition
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k]/Dr[i][j][k][1]
            a_n = Lambda[i][j][k][2] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dx[i][j][k] * dz[i][j][k]/Dr[i][j][k][3]
            a_e = Lambda[i][j][k][4] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][4]
            a_w = Lambda[i][j][k][5] * dy[i][j][k] * dz[i][j][k]/Dr[i][j][k][5]
            sub_alpha[k] = - a_e
            sub_gamma[k] = - a_w
            diag[k] = a_e + a_w + density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - S_p[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k]
        elif sample_holder[i][j][k] != 0:
            diag[k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt
        elif temperature[i][j][k] == 0:
            diag[k] = 1
    return sub_alpha, diag, sub_gamma


@njit
def set_matrices_rhs_x_sweep(n_x, i, j, rhs, temperature, latest_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c):
    for k in range(0, n_x):
        if temperature[i][j][k] > 0 and np.sum(surface[i][j][k]) == 0 and rhs[k] == 0:
            a_t = Lambda[i][j][k][0] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][0]
            a_b = Lambda[i][j][k][1] * dx[i][j][k] * dy[i][j][k] / Dr[i][j][k][1]
            a_n = Lambda[i][j][k][2] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][2]
            a_s = Lambda[i][j][k][3] * dy[i][j][k] * dz[i][j][k] / Dr[i][j][k][3]
            rhs[k] = a_t * latest_temperature[i+1][j][k] + a_b * latest_temperature[i-1][j][k] + a_n * latest_temperature[i][j+1][k] + a_s * latest_temperature[i][j-1][k] + S_c[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] + (density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt - a_t - a_b - a_n - a_s) * temperature[i][j][k]
        elif sample_holder[i][j][k] != 0:
            rhs[k] = density[i][j][k] * heat_capacity[i][j][k] * dx[i][j][k] * dy[i][j][k] * dz[i][j][k] / dt * temperature[i][j][k]
        elif temperature[i][j][k] == 0:
            rhs[k] = 0
    return rhs


@njit
def boundary_condition_implicit_x_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_reduced, sub_alpha, diag, sub_gamma, rhs, S_c, S_p):
    for each in surface_reduced:
        S_c_rad = 3 * epsilon * sigma * temperature[each[2]][each[1]][each[0]]**4 * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        S_p_rad = -4 * epsilon * sigma * temperature[each[2]][each[1]][each[0]]**3 * (dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][0] + surface[each[2]][each[1]][each[0]][1]) + dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][2] + surface[each[2]][each[1]][each[0]][3]) + dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] * (surface[each[2]][each[1]][each[0]][4] + surface[each[2]][each[1]][each[0]][5]))
        Q = input_energy[each[2]][each[1]][each[0]] / r_H ** 2 * (1 - albedo) * surface[each[2]][each[1]][each[0]][1]
        a_t = Lambda[each[2]][each[1]][each[0]][0] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][0] * (1 - surface[each[2]][each[1]][each[0]][0])
        a_b = Lambda[each[2]][each[1]][each[0]][1] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][1] * (1 - surface[each[2]][each[1]][each[0]][1])
        a_n = Lambda[each[2]][each[1]][each[0]][2] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][2] * (1 - surface[each[2]][each[1]][each[0]][2])
        a_s = Lambda[each[2]][each[1]][each[0]][3] * dx[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][3] * (1 - surface[each[2]][each[1]][each[0]][3])
        a_e = Lambda[each[2]][each[1]][each[0]][4] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][4] * (1 - surface[each[2]][each[1]][each[0]][4])
        a_w = Lambda[each[2]][each[1]][each[0]][5] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / Dr[each[2]][each[1]][each[0]][5] * (1 - surface[each[2]][each[1]][each[0]][5])
        diag[each[0]] = -S_p[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] - S_p_rad + a_e + a_w + density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt
        sub_alpha[each[0]] = -a_e
        sub_gamma[each[0]] = -a_w
        rhs[each[0]] = a_t * latest_temperature[each[2] + 1][each[1]][each[0]] + a_b * latest_temperature[each[2] - 1][each[1]][each[0]] + a_n * latest_temperature[each[2]][each[1] + 1][each[0]] + a_s * latest_temperature[each[2]][each[1] - 1][each[0]] + Q + S_c[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] + S_c_rad + (density[each[2]][each[1]][each[0]] * heat_capacity[each[2]][each[1]][each[0]] * dx[each[2]][each[1]][each[0]] * dy[each[2]][each[1]][each[0]] * dz[each[2]][each[1]][each[0]] / dt - a_t - a_b - a_n - a_s) * temperature[each[2]][each[1]][each[0]]
    return sub_alpha, diag, sub_gamma, rhs


@njit
def hte_implicit(n_x, n_y, n_z, surface_reduced, r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder):
    dt = dt/3
    for j in range(1, n_y-1):
        for k in range(1, n_x-1):
            #if temperature[1][j][k] > 0 and sample_holder[1][j][k] == 0:
            sub_alpha = np.zeros(n_z, dtype=np.float64)
            diag = np.zeros(n_z, dtype=np.float64)
            sub_gamma = np.zeros(n_z, dtype=np.float64)
            rhs = np.zeros(n_z, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[1] == j and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_z_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_z_sweep(n_z, j, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_z_sweep(n_z, j, k, rhs, temperature, latest_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            latest_temperature[0:n_z, j, k] = tridiagonal_matrix_solver(n_z, diag, sub_gamma, sub_alpha, rhs)
    temperature[0:n_z, 0:n_y, 0:n_x] = latest_temperature[0:n_z, 0:n_y, 0:n_x]
    for i in range(1, n_z-1):
        for k in range(1, n_x-1):
            sub_alpha = np.zeros(n_y, dtype=np.float64)
            diag = np.zeros(n_y, dtype=np.float64)
            sub_gamma = np.zeros(n_y, dtype=np.float64)
            rhs = np.zeros(n_y, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_y_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_y_sweep(n_y, i, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_y_sweep(n_y, i, k, rhs, temperature, latest_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            latest_temperature[i, 0:n_y, k] = tridiagonal_matrix_solver(n_y, diag, sub_gamma, sub_alpha, rhs)
    temperature[0:n_z, 0:n_y, 0:n_x] = latest_temperature[0:n_z, 0:n_y, 0:n_x]
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            sub_alpha = np.zeros(n_x, dtype=np.float64)
            diag = np.zeros(n_x, dtype=np.float64)
            sub_gamma = np.zeros(n_x, dtype=np.float64)
            rhs = np.zeros(n_x, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[1] == j:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_x_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_x_sweep(n_x, i, j, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_x_sweep(n_x, i, j, rhs, temperature, latest_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            latest_temperature[i, j, 0:n_x] = tridiagonal_matrix_solver(n_x, diag, sub_gamma, sub_alpha, rhs)
    return latest_temperature


@njit
def hte_implicit_y(n_x, n_y, n_z, surface_reduced, r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder):
    for i in range(1, n_z-1):
        for k in range(1, n_x-1):
            sub_alpha = np.zeros(n_y, dtype=np.float64)
            diag = np.zeros(n_y, dtype=np.float64)
            sub_gamma = np.zeros(n_y, dtype=np.float64)
            rhs = np.zeros(n_y, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[0] == k:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_y_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_y_sweep(n_y, i, k, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_y_sweep(n_y, i, k, rhs, temperature, latest_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            latest_temperature[i, 0:n_y, k] = tridiagonal_matrix_solver(n_y, diag, sub_gamma, sub_alpha, rhs)
    return latest_temperature


@njit
def hte_implicit_x(n_x, n_y, n_z, surface_reduced, r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, S_c, S_p, sample_holder):
    for i in range(1, n_z-1):
        for j in range(1, n_y-1):
            sub_alpha = np.zeros(n_x, dtype=np.float64)
            diag = np.zeros(n_x, dtype=np.float64)
            sub_gamma = np.zeros(n_x, dtype=np.float64)
            rhs = np.zeros(n_x, dtype=np.float64)
            surface_elements_in_line = np.zeros(np.shape(surface_reduced), dtype=np.int32)
            counter = 0
            for each in surface_reduced:
                if each[2] == i and each[1] == j:
                    surface_elements_in_line[counter] = np.array([each[0], each[1], each[2]], dtype=np.float64)
                    counter += 1
            sub_alpha, diag, sub_gamma, rhs = boundary_condition_implicit_x_sweep(r_H, albedo, dt, input_energy, sigma, epsilon, temperature, latest_temperature, Lambda, Dr, sublimated_mass, resublimated_mass, latent_heat_water, heat_capacity, density, dx, dy, dz, surface, surface_elements_in_line[0:counter], sub_alpha, diag, sub_gamma, rhs, S_c, S_p)
            sub_alpha, diag, sub_gamma = set_matrices_lhs_x_sweep(n_x, i, j, sub_alpha, diag, sub_gamma, Lambda, dx, dy, dz, Dr, density, heat_capacity, dt, S_p, temperature, surface, sample_holder)
            rhs = set_matrices_rhs_x_sweep(n_x, i, j, rhs, temperature, latest_temperature, surface, sample_holder, dx, dy, dz, Dr, Lambda, density, heat_capacity, dt, S_c)
            latest_temperature[i, j, 0:n_x] = tridiagonal_matrix_solver(n_x, diag, sub_gamma, sub_alpha, rhs)
    return latest_temperature