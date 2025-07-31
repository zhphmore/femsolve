import numpy as np


def shape_function(x_a, elem, xg, jacobians):
    """
    Calculate the B matrix for all the elements
    Only support C2D3 or C2D4, nne (number of nodal elements) == 3 or 4
    
    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    xg (ndarray): Barycenters of all the elements, shape: (num_elements, dim_space)
    jacobians (ndarray): Jacobians of each element (Surface areas of all the elements if 2D), shape: (num_elements, )
    
    Returns:
    p (ndarray): Values of the shape function at the barycenters, shape: (num_elements, nne)
    dp (ndarray): Derivatives of the shape function at the barycenters, shape: (num_elements, nne, dim_space)
    """
    num_nodes, dim_space = x_a.shape
    num_elements, nne = elem.shape

    p = np.zeros((num_elements, nne))
    dp = np.zeros((num_elements, nne, dim_space))

    # C2D3
    if nne == 3:
        p, dp = linear_interpolation(x_a, elem, xg, jacobians)
    # C2D4
    elif nne == 4:
        p, dp = bilinear_interpolation(x_a, elem, xg, jacobians)

    return p, dp


def linear_interpolation(x_a, elem, xg, jacobians):
    """
    Calculate linear shape functions and their derivatives for triangular elements
    
    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    xg (ndarray): Barycenters of all the elements, shape: (num_elements, dim_space)
    jacobians (ndarray): Jacobians of each element (Surface areas of all the elements if 2D), shape: (num_elements, )

    Returns:
    p (ndarray): Values of the shape function at the barycenters, shape: (num_elements, nne)
    dp (ndarray): Derivatives of the shape function at the barycenters, shape: (num_elements, nne, dim_space)
    """
    num_nodes, dim_space = x_a.shape
    num_elements, nne = elem.shape

    p = np.zeros((num_elements, nne))
    dp = np.zeros((num_elements, nne, dim_space))

    for i_elem in range(num_elements):
        x1 = x_a[elem[i_elem, 0], 0]
        x2 = x_a[elem[i_elem, 1], 0]
        x3 = x_a[elem[i_elem, 2], 0]
        y1 = x_a[elem[i_elem, 0], 1]
        y2 = x_a[elem[i_elem, 1], 1]
        y3 = x_a[elem[i_elem, 2], 1]

        N = np.zeros(3)
        N[0] = xg[i_elem, 0] * (y2 - y3) + xg[i_elem, 1] * (x3 - x2) + (x2 * y3 - x3 * y2)
        N[1] = xg[i_elem, 0] * (y3 - y1) + xg[i_elem, 1] * (x1 - x3) + (x3 * y1 - x1 * y3)
        N[2] = xg[i_elem, 0] * (y1 - y2) + xg[i_elem, 1] * (x2 - x1) + (x1 * y2 - x2 * y1)

        p[i_elem] = N / (2 * jacobians[i_elem])

        dN = np.array([[y2 - y3, x3 - x2],
                       [y3 - y1, x1 - x3],
                       [y1 - y2, x2 - x1]])

        dp[i_elem] = dN / (2 * jacobians[i_elem])

    return p, dp


def bilinear_interpolation(x_a, elem, xg, jacobians):
    """
    Calculate bilinear shape functions and their derivatives for quadrilateral elements
    
    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    xg (ndarray): Barycenters of all the elements, shape: (num_elements, dim_space)
    jacobians (ndarray): Jacobians of each element (Surface areas of all the elements if 2D), shape: (num_elements, )

    Returns:
    p (ndarray): Values of the shape function at the barycenters, shape: (num_elements, nne)
    dp (ndarray): Derivatives of the shape function at the barycenters, shape: (num_elements, nne, dim_space)
    """
    num_nodes, dim_space = x_a.shape
    num_elements, nne = elem.shape
    p = np.ones((num_elements, nne))
    dp = np.zeros((num_elements, nne, dim_space))

    p *= 0.25

    for i_elem in range(num_elements):
        elem_coord = x_a[elem[i_elem, :]]

        # Ni derivatives with respect to natural coordinates "z1" & "z2"
        z = np.array([0.5, 0.5])

        dN_i = np.array([
            [z[1] - 1, -z[1] + 1, 1 + z[1], -1 - z[1]],
            [z[0] - 1, -z[0] - 1, 1 + z[0], 1 - z[0]]
        ]) / 4

        # Jacobian Matrix
        J = dN_i @ elem_coord
        # detJ = np.linalg.det(J)

        dN = np.linalg.solve(J, dN_i)

        dp[i_elem] = dN.T

    return p, dp
