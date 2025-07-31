import numpy as np
from numpy.linalg import det, inv, eigh

from .constitutive import constitutive


def force_internal(x_a_u, elem, jacobians, DN, properties, epsilon_p):
    """
    Assemble the nodal internal force vector
    
    Parameters:
    x_a_u (ndarray): Current nodal coordinates, shape: (num_nodes, dim_space)
    elem (ndarray): Connectivity table, shape: (num_elements, nne)
    jacobians (ndarray): Surface area associated with each element, shape: (num_elements, )
    dp (ndarray): Derivatives of shape functions at each element, shape: (num_elements, nne, dim_space)
    properties (ndarray): Material properties
    epsilon_p (ndarray): Plastic strain tensor from last time step, shape: (num_elements, dim_space ** 2)
    
    Returns:
    F_int (ndarray): Global nodal force vector , shape: (num_nodes, dim_space)
    epsl_ans (ndarray): Total strain, shape: (num_elements, dim_space ** 2)
    epsl_p_ans (ndarray): Plastic strain, shape: (num_elements, dim_space ** 2)
    sigma_SP_ans (ndarray): Stress tensor, shape: (num_elements, dim_space ** 2)
    ep_eff_ans (ndarray): Effective plastic strain, shape: (num_elements, )
    sig_eff_ans (ndarray): von-Mises stress, shape: (num_elements, )
    """
    num_nodes, dim_space = x_a_u.shape
    num_elements, nne = elem.shape

    F_int = np.zeros((num_nodes, dim_space))
    ep_eff_ans = np.zeros(num_elements)
    sig_eff_ans = np.zeros(num_elements)
    # Initialize strain arrays
    epsl_ans = np.zeros((num_elements, dim_space ** 2))
    epsl_p_ans = np.zeros((num_elements, dim_space ** 2))
    sigma_SP_ans = np.zeros((num_elements, dim_space ** 2))

    # Linear hardening parameters
    sig_0 = properties[3]
    E_h = properties[4]
    power_a = properties[5]

    for i_elem in range(num_elements):
        # 1. Calculate deformation gradient
        # F_dg (ndarray): Deformation gradient, shape: (dim_space, dim_space)
        F_dg = x_a_u[elem[i_elem, :]].T @ DN[i_elem]

        # 2. Convert to Hencky strain tensor
        # epsl (ndarray): Hencky strain tensor of total deformation, shape: (dim_space, dim_space)
        U_srr = F_dg.T @ F_dg
        epsl = np.multiply(0.5, logm_eig(U_srr))

        # 3. Evaluate stress at element
        # Get plastic strain from previous step
        # epsl_p (ndarray): Plastic strain tensor, shape: (dim_space, dim_space)
        epsl_p = epsilon_p[i_elem].reshape((dim_space, dim_space))
        sig_S, sig_P, ep_eff, sig_eff = constitutive(epsl, epsl_p, properties)

        # Calculate hardening
        sig_c = sig_0 * ((1 + E_h * ep_eff) ** power_a)

        # Check if yielding occurs
        if sig_eff > sig_c:
            # Calculate elastic strain
            epsl_e = epsl - epsl_p
            ee_eff = np.sqrt((2 / 3) * np.sum(epsl_e ** 2))

            # Solve for plastic strain increment using Newton iteration
            delta_ep_eff = solve_eq(properties, ep_eff, ee_eff)

            # Calculate plastic strain increment
            delta_ep = delta_ep_eff * np.sqrt(3 / 2) * (sig_S / np.sqrt(np.sum(sig_S ** 2)))

            # Update plastic strain
            epsl_p = epsl_p + delta_ep

            # Recalculate stress
            sig_S, sig_P, ep_eff, sig_eff = constitutive(epsl, epsl_p, properties)

        # Record strains and stresses
        epsl_ans[i_elem] = epsl.ravel()
        epsl_p_ans[i_elem] = epsl_p.ravel()
        ep_eff_ans[i_elem] = ep_eff
        sig_eff_ans[i_elem] = sig_eff

        # Calculate total stress
        sigma_SP = sig_S + sig_P
        sigma_SP_ans[i_elem] = sigma_SP.ravel()

        # 4. Compute nodal force contribution
        # PK (ndarray): First Piola-Kirchhoff stress, shape: (dim_space, dim_space)
        PK = det(F_dg) * (sigma_SP @ inv(F_dg.T))

        # F_int_elem (ndarray): Element internal force, shape: (nne, dim_space)
        F_int_elem = (PK @ DN[i_elem].T * jacobians[i_elem]).T

        # 5. Assemble to global force vector
        for i_ne in range(nne):
            F_int[elem[i_elem, i_ne]] += F_int_elem[i_ne]

    F_int = F_int.flatten()

    return F_int, epsl_ans, epsl_p_ans, sigma_SP_ans, ep_eff_ans, sig_eff_ans


def solve_eq(properties, ep_eff, ee_eff):
    """
    Newton iteration to solve for plastic strain increment
    
    Parameters:
    properties (ndarray): Material properties
    ep_eff (float): Current effective plastic strain
    ee_eff (float): Parameter for yield function
    
    Returns:
    delta_ep_eff (float): Plastic strain increment
    """
    # Material properties
    E = properties[1]  # Young's modulus
    nu = properties[2]  # Poisson's ratio
    sig_0 = properties[3]
    E_h = properties[4]
    power_a = properties[5]
    G = E / (2 * (1 + nu))  # Shear modulus

    delta_ep_eff = 0

    # Newton iteration parameters
    num_tolerance = 1e-10
    num_iteration = 100

    # Newton iteration
    for i in range(num_iteration):
        # Function to solve
        f = sig_0 * ((1 + E_h * (ep_eff + delta_ep_eff)) ** power_a) - 3 * G * (ee_eff - delta_ep_eff)

        # Derivative
        df = sig_0 * power_a * ((1 + E_h * (ep_eff + delta_ep_eff)) ** (power_a - 1)) * E_h + 3 * G

        if abs(f) < num_tolerance or abs(f / df) < num_tolerance:
            break

        delta_ep_eff = delta_ep_eff - f / df

    return delta_ep_eff


def logm_eig(M):
    """
    Compute the matrix logarithm of real symmetric matrix using eigen-decomposition method

    Parameters:
    M (ndarray): must be real symmetric matrix
    """
    eig_value, eig_vector = eigh(M)

    logm_M = eig_vector @ np.diag(np.log(eig_value)) @ eig_vector.T

    return logm_M
