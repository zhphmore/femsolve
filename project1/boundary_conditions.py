import numpy as np


def boundary_conditions(x_a, elem, boundaries):
    """
    Set up the boundary conditions and initial conditions
    
    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
    x0 (float): Nodes on the line x=x0 are fixed
    yt (float): The top horizontal edge
    Ft (float): Traction is applied on the top horizontal edge
    
    Returns:
    fixed_area (ndarray): Displacement boundary conditions (1 for fixed, 0 for free), shape: (num_nodes, dim_space)
    u_init (ndarray): Initial nodal displacements, shape: (num_nodes, dim_space)
    force_external (ndarray): Global force vector, shape: (num_nodes, dim_space)
    """
    x0 = boundaries[0]
    yt = boundaries[1]
    Ft = boundaries[2]

    num_nodes, dim_space = x_a.shape
    num_elements, nne = elem.shape

    # Initialize boundary conditions array
    fixed_area = np.zeros(shape=(num_nodes, dim_space), dtype=int)
    u_init = np.zeros((num_nodes, dim_space))
    force_surface = np.zeros((num_nodes, dim_space))

    # fixed_area
    for i_node in range(num_nodes):
        # Nodes at x=x0 are fixed (left side of the trapezoidal plate)
        if x_a[i_node, 0] == x0:
            # Mark both x and y coordinates as fixed (1)
            fixed_area[i_node, :] = 1

    # force_surface
    sur_id = np.zeros(nne, dtype=int)
    for i_elem in range(num_elements):
        sur_cnt = 0
        for i_nne in range(nne):
            if abs(x_a[elem[i_elem, i_nne], 1] - yt) < 1e-9:
                sur_id[sur_cnt] = elem[i_elem, i_nne]
                sur_cnt += 1
                if sur_cnt >= 2:
                    delta_area = abs(x_a[sur_id[0], 0] - x_a[sur_id[1], 0]) / 2
                    force_surface[sur_id[0], 1] += delta_area
                    force_surface[sur_id[1], 1] += delta_area
                    break

    force_external = Ft * force_surface

    return fixed_area, u_init, force_external
