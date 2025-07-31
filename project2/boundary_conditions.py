import numpy as np


def boundary_conditions(x_a, boundaries):
    """
    Set up the boundary conditions and initial conditions
    
    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
    x0 (float): Nodes on the line x=x0 are fixed
    xt (float): Nodes on the line x=xt are initialized with velocity (0.0, v0)
    v0 (float): Initial velocity
    
    Returns:
    fixed_area (ndarray): Displacement boundary conditions (1 for fixed, 0 for free), shape: (num_nodes, dim_space)
    u_init (ndarray): Initial nodal displacements, shape: (num_nodes, dim_space)
    velocity_init (ndarray): Initial nodal velocity, shape: (num_nodes, dim_space)
    force_external (ndarray): Global force vector, shape: (num_nodes, dim_space)
    """
    x0 = boundaries[0]
    xt = boundaries[1]
    v0 = boundaries[2]

    num_nodes, dim_space = x_a.shape

    # Initialize boundary conditions array
    # Each node has dim degrees of freedom
    fixed_area = np.zeros(shape=(num_nodes, dim_space), dtype=int)
    u_init = np.zeros((num_nodes, dim_space))
    velocity_init = np.zeros((num_nodes, dim_space))
    force_external = np.zeros((num_nodes, dim_space))

    for i_node in range(num_nodes):
        # Nodes at x=x0 are fixed (left side of the trapezoidal plate)
        if x_a[i_node, 0] == x0:
            # Mark both x and y coordinates as fixed (1)
            fixed_area[i_node, :] = 1

        # Nodes at x=xt are given initial velocity (right side of the plate)
        if x_a[i_node, 0] == xt:
            # Set initial velocity in y-direction
            velocity_init[i_node, 1] = v0

    return fixed_area, u_init, velocity_init, force_external
