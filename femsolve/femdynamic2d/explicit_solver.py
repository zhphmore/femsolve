import numpy as np
from tqdm import tqdm

from .force_internal import force_internal


def explicit_solver(x_a, elem, fixed_area, u_init, velocity_init, force_external, jacobians, DN, properties, Mass,
                    gamma, time_step_val, total_steps):
    """
    The Newmark Explicit Dynamics Analysis
    
    Parameters:
    x_a (ndarray): Nodal coordinates, shape: (num_nodes, dim_space)
    elem (ndarray): Connectivity table, shape: (num_elements, nne)
    fixed_area (ndarray): Displacement boundary conditions (1 for fixed, 0 for free), shape: (num_nodes, dim_space)
    u_init (ndarray): Initial nodal displacements, shape: (num_nodes, dim_space)
    velocity_init (ndarray): Initial nodal velocity, shape: (num_nodes, dim_space)
    force_external (ndarray): Global force vector, shape: (num_nodes, dim_space)
    jacobians (ndarray): Surface area associated with each element, shape: (num_elements, )
    dp (ndarray): Derivatives of shape functions, shape: (num_elements, nne, dim_space)
    properties (ndarray): Material properties
    Mass (ndarray): Mass matrix, shape: (dim_space * num_nodes, dim_space * num_nodes)
    gamma (float): Time integration parameter (0.5 for central difference)
    time_step_val (float): Time step size
    total_steps (int): Total number of time steps

    Returns:
    u (ndarray): Nodal displacements at each time step, shape: (total_steps, num_nodes, dim_space)
    epsilon (ndarray): Total strain at each time step, shape: (total_steps, num_elements, dim_space ** 2)
    epsilon_p (ndarray): Plastic strain at each time step, shape: (total_steps, num_elements, dim_space ** 2)
    sigma_SP (ndarray): Stress at each time step, shape: (total_steps, num_elements, dim_space ** 2)
    ep_eff (ndarray): Effective plastic strain at each time step, shape: (total_steps, num_elements)
    sig_eff (ndarray): Effective von-Mises stress at each time step, shape: (total_steps, num_elements)
    """
    num_nodes, dim_space = x_a.shape
    num_elements, nne = elem.shape

    # Initialize visualization variables
    u = np.zeros((total_steps, num_nodes, dim_space))  # nodal displacements
    u[0] = u_init
    ep_eff = np.zeros((total_steps, num_elements))  # effective plastic strain
    sig_eff = np.zeros((total_steps, num_elements))  # von-Mises stresses

    # Initialize internal variables
    epsilon = np.zeros((total_steps, num_elements, dim_space ** 2))
    epsilon_p = np.zeros((total_steps, num_elements, dim_space ** 2))
    sigma_SP = np.zeros((total_steps, num_elements, dim_space ** 2))

    # Initialize velocity
    v = np.zeros((total_steps, num_nodes, dim_space))
    v[0] = velocity_init

    # Initialize nodal acceleration and forces
    a = np.zeros((num_nodes, dim_space))

    # F_ext (ndarray): External forces, shape: (num_nodes * dim_space, )
    F_ext = force_external.reshape((num_nodes * dim_space))

    # Time integration loop
    for step in tqdm(range(1, total_steps), desc="Explicit solver"):
        # 1. Predictor
        # print('u: ', u)
        u[step] = u[step - 1] + v[step - 1] * time_step_val + 0.5 * a * (time_step_val ** 2)
        u[step][fixed_area == 1] = 0
        x_a_u = x_a + u[step]

        # 2. Calculate internal forces
        F_int, epsl_ans, epsl_p_ans, sigma_SP_ans, ep_eff_ans, sig_eff_ans = force_internal(x_a_u, elem, jacobians, DN,
                                                                                            properties,
                                                                                            epsilon_p[step - 1])

        # Record strains and stresses
        epsilon[step] = epsl_ans
        epsilon_p[step] = epsl_p_ans
        sigma_SP[step] = sigma_SP_ans
        ep_eff[step] = ep_eff_ans
        sig_eff[step] = sig_eff_ans

        # 3. Corrector
        # Total force = external force - internal force
        F_total = F_ext - F_int

        # Apply boundary conditions
        F_total[fixed_area.ravel() == 1] = 0

        # Calculate next acceleration
        a_next = np.linalg.solve(Mass, F_total)

        a_next = a_next.reshape((num_nodes, dim_space))

        # Update velocity
        v[step] = v[step - 1] + 0.5 * (a + a_next) * time_step_val

        # Update acceleration
        a = a_next

    return u, epsilon, epsilon_p, sigma_SP, ep_eff, sig_eff
