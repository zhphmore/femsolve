import numpy as np


def g_center(x_a, elem):
    """
    Calculate the barycenter and surface area of each element
    
    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
        Only support 2D, dim_space == 2
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    
    Returns:
    xg (ndarray): Barycenters of all the elements, shape: (num_elements, dim_space)
    jacobians (ndarray): Jacobians of each element (Surface areas of all the elements if 2D), shape: (num_elements, )
    """
    num_nodes, dim_space = x_a.shape
    num_elements, nne = elem.shape

    xg = np.zeros((num_elements, dim_space))
    jacobians = np.zeros(num_elements)

    # Calculate centroids
    for i_elem in range(num_elements):
        sum_coord = np.zeros(dim_space)
        for i_ne in range(nne):
            sum_coord += x_a[elem[i_elem, i_ne]]
        xg[i_elem] = sum_coord / nne

    # Calculate jacobians
    # Calculate areas
    for i_elem in range(num_elements):
        elem_coord = np.zeros((dim_space, nne + 1))

        # Shoelace formula
        for i_ne in range(nne):
            elem_coord[:, i_ne] = x_a[elem[i_elem, i_ne], :]
        elem_coord[:, nne] = x_a[elem[i_elem, 0], :]

        xi_yi1 = 0
        xi1_yi = 0
        for i_ne in range(nne):
            xi_yi1 += elem_coord[0, i_ne] * elem_coord[1, i_ne + 1]
            xi1_yi += elem_coord[1, i_ne] * elem_coord[0, i_ne + 1]
        jacobians[i_elem] = abs(xi_yi1 - xi1_yi) / 2

    return xg, jacobians
