import numpy as np


def boundary_conditions(x_a, boundaries):
    """
    Set up the boundary conditions and initial conditions
    
    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
    x0 (float): Nodes on the line x=x0 are fixed
    y0 (float): Nodes on the line y=y0 are fixed
    v0 (float): Initial velocity
    
    Returns:
    fixed_area (ndarray): Displacement boundary conditions (1 for fixed, 0 for free), shape: (num_nodes, dim_space)
    u_init (ndarray): Initial nodal displacements, shape: (num_nodes, dim_space)
    velocity_init (ndarray): Initial nodal velocity, shape: (num_nodes, dim_space)
    force_external (ndarray): Global force vector, shape: (num_nodes, dim_space)
    """
    x0 = boundaries[0]
    y0 = boundaries[1]
    v0 = boundaries[2]

    num_nodes, dim_space = x_a.shape

    # Initialize boundary conditions array
    fixed_area = np.zeros(shape=(num_nodes, dim_space), dtype=int)
    u_init = np.zeros((num_nodes, dim_space))
    velocity_init = np.zeros((num_nodes, dim_space))
    force_external = np.zeros((num_nodes, dim_space))

    for i_node in range(num_nodes):
        # Nodes at x=x0 are fixed (left side of the trapezoidal plate)
        if x_a[i_node, 0] == x0:
            # Mark x as fixed (1)
            fixed_area[i_node, 0] = 1

        if x_a[i_node, 1] == y0:
            # Mark y as fixed (1)
            fixed_area[i_node, 1] = 1

        # An initial velocity v0 in the left direction is applied to the entire domain at time t = 0
        # Set initial velocity in x-direction
        velocity_init[i_node, 0] = v0

    return fixed_area, u_init, velocity_init, force_external
