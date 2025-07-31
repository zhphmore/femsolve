import numpy as np


def generate_mesh(geometry_set, mesh_set):
    """
    Generate mesh for finite element analysis
    
    Parameters:
    flag_elem_type (int): 1 for triangular elements, 2 for quadrilateral elements
    
    Returns:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    """
    flag_elem_type = mesh_set[0]

    # Cook's membrane
    x_a, elem = mesh_C2D4(geometry_set, num_elem_horizon=mesh_set[1], num_elem_vertical=mesh_set[2])

    # triangular elements, C2D3
    if flag_elem_type == 1:
        elem = split_C2D4_to_C2D3(x_a, elem)

    return x_a, elem


def mesh_C2D4(geometry, num_elem_horizon, num_elem_vertical):
    """
    Generate coordinates for Cook's membrane problem
    
    Parameters:
    geometry (int): Geometric parameters of trapezoidal panel
    num_elem_horizon (int): Number of elements in horizontal direction
    num_elem_vertical (int): Number of elements in vertical direction
    
    Returns:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space), dim_space == 2
    elem (ndarray): Element connectivity table, shape: (num_elements, nne), nne == 4
    """
    # geometry
    H = geometry[0]
    V_left = geometry[1]
    V_right = geometry[2]

    # Number of nodes
    num_node_horizon = num_elem_horizon + 1
    num_node_vertical = num_elem_vertical + 1

    # Element type: C2D4
    dim_space = 2
    nne = 4

    x_a = np.zeros((num_node_vertical * num_node_horizon, dim_space))
    elem = np.zeros(shape=(num_elem_vertical * num_elem_horizon, nne), dtype=int)

    # Number the nodes: from left to right, from bottom to top
    for i in range(num_node_horizon):
        for j in range(num_node_vertical):
            id_node = j * num_node_horizon + i

            x_a[id_node, 0] = i * (H / num_elem_horizon)
            Vj = (V_left - V_right) * (i / num_elem_horizon)
            x_a[id_node, 1] = j * ((V_left - Vj) / num_elem_vertical) + Vj

    # Number the nodes: from left to right, from bottom to top
    for i in range(num_elem_horizon):
        for j in range(num_elem_vertical):
            id_element = j * num_elem_horizon + i
            elem[id_element, 0] = j * num_node_horizon + i
            elem[id_element, 1] = j * num_node_horizon + i + 1
            elem[id_element, 2] = (j + 1) * num_node_horizon + i + 1
            elem[id_element, 3] = (j + 1) * num_node_horizon + i

    return x_a, elem


def split_C2D4_to_C2D3(x_a, elem):
    """
    Split quadrilateral elements into triangular elements

    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space), dim_space == 2
    elem (ndarray): Element connectivity table for quadrilateral elements, shape: (num_elements, nne), nne == 4

    Returns:
    elem_new (ndarray): New element connectivity table for triangular elements, shape: (num_elements * 2, nne - 1)
    """
    num_elements, nne = elem.shape
    num_nodes, dim_space = x_a.shape

    elem_new = np.zeros(shape=(num_elements * 2, nne - 1), dtype=int)

    for i_elem in range(num_elements):
        elem_coord = np.zeros((nne, dim_space))
        for i_nne in range(nne):
            elem_coord[i_nne] = x_a[elem[i_elem, i_nne]]

        flag_circumcircle = circunf3(elem_coord)

        if flag_circumcircle:
            elem_new[i_elem * 2, 0] = elem[i_elem, 0]
            elem_new[i_elem * 2, 1] = elem[i_elem, 1]
            elem_new[i_elem * 2, 2] = elem[i_elem, 2]

            elem_new[i_elem * 2 + 1, 0] = elem[i_elem, 2]
            elem_new[i_elem * 2 + 1, 1] = elem[i_elem, 3]
            elem_new[i_elem * 2 + 1, 2] = elem[i_elem, 0]
        else:
            elem_new[i_elem * 2, 0] = elem[i_elem, 1]
            elem_new[i_elem * 2, 1] = elem[i_elem, 2]
            elem_new[i_elem * 2, 2] = elem[i_elem, 3]

            elem_new[i_elem * 2 + 1, 0] = elem[i_elem, 3]
            elem_new[i_elem * 2 + 1, 1] = elem[i_elem, 0]
            elem_new[i_elem * 2 + 1, 2] = elem[i_elem, 1]

    return elem_new


def circunf3(x_a):
    """
    Determine the optimal way to split a quadrilateral element into triangles
    based on the circumcircle of three points

    Parameters:
    x_a (ndarray): Coordinates of the quadrilateral element vertices, shape: (num_nodes, dim_space)

    Returns:
    flag_circumcircle (bool): Indicator for the splitting pattern
    """
    flag_circumcircle = False

    A = np.ones((3, 3))
    C = np.zeros(3)

    for i in range(3):
        C[i] = -(x_a[i, 0] ** 2 + x_a[i, 1] ** 2)
        for j in range(1, 3):
            A[i, j] = x_a[i, j - 1]

    B = np.linalg.solve(A, C)

    c = np.array([-B[1] / 2, -B[2] / 2])
    r = np.sqrt(c[0] ** 2 + c[1] ** 2 - B[0])

    d = np.sqrt((x_a[3, 0] - c[0]) ** 2 + (x_a[3, 1] - c[1]) ** 2)

    if d >= r:
        flag_circumcircle = True

    return flag_circumcircle
