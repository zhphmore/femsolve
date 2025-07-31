import numpy as np


def B_matrix(dim_space, num_elements, nne, dp):
    """
    Calculate the B matrix for all the elements
    
    Parameters:
    dim_space (int): Spatial dimension
    num_elements (int): Total number of elements
    nne (int): Number of nodal elements
    dp (ndarray): Derivatives of the shape function at the barycenters, shape: (num_elements, nne, dim_space)
    
    Returns:
    B (ndarray): list of B matrices for all elements, shape: (num_elements, row_B, nne * dim_space)
        For 2D: row_B == 3, for 3D: row_B == 6
    """
    row_B = int(0.5 * dim_space * (dim_space + 1))
    B = np.zeros((num_elements, row_B, nne * dim_space))

    # 2D plane problem
    # B matrix for each element: shape: (3, (2 * nne))
    # For C2D3: shape: (3, 6), for C2D4: shape: (3, 8)
    if dim_space == 2:
        for i_elem in range(num_elements):
            for i_nne in range(nne):
                for i in range(dim_space):
                    B[i_elem, i, i_nne * dim_space + i] = dp[i_elem, i_nne, i]
                B[i_elem, 2, i_nne * dim_space] = dp[i_elem, i_nne, 1]
                B[i_elem, 2, i_nne * dim_space + 1] = dp[i_elem, i_nne, 0]

    # 3D space problem
    # B matrix for each element: shape: (6, (3 * nne))
    # For C2D3: shape: (6, 9), for C2D4: shape: (6, 12)
    elif dim_space == 3:
        for i_elem in range(num_elements):
            for i_nne in range(nne):
                for i in range(dim_space):
                    B[i_elem, i, i_nne * dim_space + i] = dp[i_elem, i_nne, i]
                B[i_elem, 3, i_nne * dim_space] = dp[i_elem, i_nne, 1]
                B[i_elem, 3, i_nne * dim_space + 1] = dp[i_elem, i_nne, 0]
                B[i_elem, 4, i_nne * dim_space] = dp[i_elem, i_nne, 2]
                B[i_elem, 4, i_nne * dim_space + 2] = dp[i_elem, i_nne, 0]
                B[i_elem, 5, i_nne * dim_space + 1] = dp[i_elem, i_nne, 2]
                B[i_elem, 5, i_nne * dim_space + 2] = dp[i_elem, i_nne, 1]

    return B
