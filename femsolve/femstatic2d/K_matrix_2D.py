import numpy as np


def K_matrix_2D(num_nodes, dim_space, elem, jacobians, properties, B):
    """
    Assemble the global stiffness matrix
    Only support 2D, dim_space == 2
    
    Parameters:
    num_nodes (int): Total number of nodes
    dim_space (int): Spatial dimension
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    jacobians (ndarray): Jacobians of each element (Surface areas of all the elements if 2D), shape: (num_elements, )
    properties (ndarray): Material properties
    B (ndarray): list of B matrices for all elements, shape: (num_elements, row_B, nne * dim_space)\
        For 2D: row_B == 3
    
    Returns:
    K (ndarray): global stiffness matrix, shape: (num_nodes * dim_space, num_nodes * dim_space)
    """
    # Material properties
    E = properties[1]  # Young's modulus, Pa
    nu = properties[2]  # Poisson ratio

    # Plane stress stiffness matrix
    # C (ndarray), shape: (row_B, row_B), for 2D: row_B == 3
    C = (E / (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    num_elements, nne = elem.shape

    # K matrix
    K = np.zeros((num_nodes * dim_space, num_nodes * dim_space))

    for i_elem in range(num_elements):
        # Ke (ndarray): Element stiffness matrix, shape: (nne * dim_space, nne * dim_space)
        Ke = B[i_elem].T @ C @ B[i_elem] * jacobians[i_elem]

        # Assemble element stiffness matrix into global stiffness matrix
        # Based on node numbers contained in the element
        for i_nne in range(nne):
            for j_nne in range(nne):
                K[(elem[i_elem, i_nne] * dim_space):((elem[i_elem, i_nne] + 1) * dim_space),
                (elem[i_elem, j_nne] * dim_space):((elem[i_elem, j_nne] + 1) * dim_space)] += Ke[(i_nne * dim_space):(
                            (i_nne + 1) * dim_space), (j_nne * dim_space):((j_nne + 1) * dim_space)]

    return K
