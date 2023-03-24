import numpy as np
from numba import njit, prange

@njit
def tridiag_matrix_solver(sub_a, diag, sub_c, res_3D, n_x, n_y, n_z):
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

