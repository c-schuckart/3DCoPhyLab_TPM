import numpy as np
from numba import njit, prange

@njit
def tridiag_matrix_solver3D(sub_a, diag, sub_c, res_3D, n_x, n_y, n_z):
    """
    This function solves the tri-diagonal matrix for the Crank-Nicolson time steps.
    It employs the Thomas-algorithm to solve the system of equations in O(n) time.
    It is only stable for diagonally dominant matrices, which all the lhs-matrices fulfill.

    Parameters
    ----------
    sub_a : array_like
        subdiagonal matrix entries belonging to the T_{k-1} terms of the Crank-Nicolson scheme (m_max, nz).
    diag : array_like
        diagonal matrix entries belonging to the T_{k} terms of the Crank-Nicolson scheme (m_max, nz).
    sub_c : array_like
        subdiagonal matrix entries belonging to the T_{k+1} terms of the Crank-Nicolson scheme (m_max, nz).
    res_2D : array_like
        Field of the variable of which the results for the new time step need to be calculated - either spectral temperature or vorticity - dimension (m_max, nz).
    nz : int
        Number of vertical grid points.
    m_max : int
        maximal order of Fourier modes.
    f : array_like
        Field of dimension (m_max, nz) from which the matrices are set.

    Returns
    -------
    Field for the current time step of the variable passed to this function in res_2D of dimension (m_max, nz).
    """

    sol_3D = np.zeros((n_z, n_y, n_x), dtype=np.float64)

    for j in prange(1, n_y-1):
        for k in range(1, n_x-1):

            # initiate working arrays and set first values
            arr_1 = np.zeros(n_z, dtype=np.float64)
            arr_2 = np.zeros(n_z, dtype=np.float64)
            arr_1[0] = sub_c[0][j][k] / diag[0][j][k]
            arr_2[0] = res_3D[0][j][k] / diag[0][j][k]

            # calculate values for the working arrays
            for i in range(1, n_z):

                t = diag[i][j][k] - sub_a[i][j][k] * arr_1[i-1]

                arr_1[i] = sub_c[i][j][k] / t

                arr_2[i] = (res_3D[i][j][k] - sub_a[i][j][k] * arr_2[i-1]) / t

            sol = np.zeros(n_z, dtype=np.float64)

            sol[n_z-1] = arr_2[n_z-1]

            # calculate solution of the system via the working arrays
            for i in range(n_z-2, -1, -1):

                sol[i] = arr_2[i] - arr_1[i] * sol[i+1]

            sol_3D[:][j][k] = sol[:]

    return sol_3D


@njit
def tridiagonal_matrix_solver(n, diag, sub_a, sub_c, res):
    arr_1 = np.zeros(n)
    arr_2 = np.zeros(n)
    arr_1[0] = sub_c[0] / diag[0]
    arr_2[0] = res[0] / diag[0]
    for i in range(1, n):
        t = diag[i] - sub_a[i] * arr_1[i-1]
        arr_1[i] = sub_c[i] / t
        arr_2[i] = (res[i] - sub_a[i] * arr_2[i-1]) / t
    sol = np.zeros(n)
    sol[n-1] = arr_2[n-1]
    for i in range(n-2, -1, -1):
        sol[i] = arr_2[i] - arr_1[i] * sol[i+1]
    return sol


@njit
def periodic_tridiagonal_matrix_solver(n, diag, sub_a, sub_c, res):
    gamma = - diag[0]
    diag[0] = diag[0] - gamma
    diag[n-1] = diag[n-1] - sub_a[0] * sub_c[n-1] / gamma
    u = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    u[0], u[n-1] = gamma, sub_c[n-1]
    v[0], v[n-1] = 1, sub_a[0]/gamma
    #print(gamma, u, v)
    #print(diag)
    y = tridiagonal_matrix_solver(n, diag, sub_a, sub_c, res)
    #print(diag)
    q = tridiagonal_matrix_solver(n, diag, sub_a, sub_c, u)
    sol = y - q * np.dot(v, y) / (1 + np.dot(v, q))
    return sol
