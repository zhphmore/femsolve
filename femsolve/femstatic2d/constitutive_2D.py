import numpy as np


def constitutive_2D(dim_space, elem, B, properties, u):
    """
    Compute strain, stress and pressure from displacement
    Only support 2D, dim_space == 2
    
    Parameters:
    dim_space (int): Spatial dimension
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    B (ndarray): list of B matrices for all elements, shape: (num_elements, row_B, nne * dim_space)
        For 2D: row_B == 3
    properties (ndarray): Material properties
    u (ndarray): Nodal displacements, shape: (num_nodes, dim_space)

    Returns:
    Es (ndarray): Strain vector, shape: (num_elements, row_B)
    Ss (ndarray): Stress vector, shape: (num_elements, row_B)
    P (ndarray): Pressure vector, shape: (num_elements, )
    """
    num_elements, nne = elem.shape
    row_B = B.shape[1]

    # Material properties
    E = properties[1]  # Young's modulus
    nu = properties[2]  # Poisson's ratio
    G = E / (2 * (1 + nu))  # Shear modulus
    K = 2 * G / 3 * (1 + nu) / (1 - 2 * nu)  # Bulk modulus

    # Plane stress constitutive matrix
    # C (ndarray), shape: (row_B, row_B), for 2D: row_B == 3
    C = (E / (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    Ss = np.zeros((num_elements, row_B))
    Es = np.zeros((num_elements, row_B))
    P = np.zeros(num_elements)

    for i_elem in range(num_elements):
        # d (ndarray), shape: (nne, dim_space)
        d = np.zeros((nne, dim_space))
        for i_nne in range(nne):
            d[i_nne] = u[elem[i_elem, i_nne]]

        # EE (ndarray), shape: (row_B, )
        EE = B[i_elem] @ d.ravel()
        # S (ndarray), shape: (row_B, )
        S = C @ EE

        Es[i_elem] = EE
        Ss[i_elem] = S
        P[i_elem] = -(Ss[i_elem, 0] + Ss[i_elem, 1]) / 3

    return Es, Ss, P
