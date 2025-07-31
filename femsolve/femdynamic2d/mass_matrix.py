import numpy as np


def mass_matrix(alpha_mass, num_nodes, dim_space, elem, p, jacobians, density):
    """
    Calculate mass matrix

    Parameters:
    alpha_mass (float): Combination factor of consistent mass matrix and lumped mass matrix
    num_nodes (int): Total number of nodes
    dim_space (int): Spatial dimension
    elem (ndarray): Connectivity table, shape: (num_elements, nne)
    p (ndarray): Values of shape functions at element barycenters, shape: (num_elements, nne)
    jacobians (ndarray): Jacobians of each element, shape: (num_elements, )
    density (float): Material density

    Returns:
    Mass (ndarray): Mass matrix, shape: (dim_space * num_nodes, dim_space * num_nodes)
    """
    if alpha_mass == 0:
        Mass = lumped_mass(num_nodes, dim_space, elem, jacobians, density)
    elif alpha_mass == 1:
        Mass = consistent_mass(num_nodes, dim_space, elem, p, jacobians, density)
    else:
        cMass = consistent_mass(num_nodes, dim_space, elem, p, jacobians, density)
        lMass = lumped_mass(num_nodes, dim_space, elem, jacobians, density)
        Mass = alpha_mass * cMass + (1 - alpha_mass) * lMass

    return Mass


def consistent_mass(num_nodes, dim_space, elem, p, jacobians, density):
    """
    Calculate the consistent mass matrix
    
    Parameters:
    num_nodes (int): Total number of nodes
    dim_space (int): Spatial dimension
    elem (ndarray): Connectivity table, shape: (num_elements, nne)
    p (ndarray): Values of shape functions at element barycenters, shape: (num_elements, nne)
    jacobians (ndarray): Jacobians of each element, shape: (num_elements, )
    density (float): Material density
    
    Returns:
    cMass (ndarray): Consistent mass matrix, shape: (dim_space * num_nodes, dim_space * num_nodes)
    """
    num_elements, nne = elem.shape
    cMass = np.zeros((dim_space * num_nodes, dim_space * num_nodes))

    # Calculate consistent mass matrix
    for i_elem in range(num_elements):
        # Create shape function matrix for current element
        NE = np.zeros((dim_space, dim_space * nne))

        for i_ne in range(nne):
            NE[:, (i_ne * dim_space):((i_ne + 1) * dim_space)] = p[i_elem, i_ne] * np.eye(dim_space)

        # Calculate element mass matrix
        Mi = NE.T @ NE * jacobians[i_elem] * density

        # Assemble into global mass matrix
        for i_ne in range(nne):
            for j_ne in range(nne):
                cMass[(elem[i_elem, i_ne] * dim_space):((elem[i_elem, i_ne] + 1) * dim_space),
                (elem[i_elem, j_ne] * dim_space):((elem[i_elem, j_ne] + 1) * dim_space)] = Mi[(i_ne * dim_space):(
                        (i_ne + 1) * dim_space), (j_ne * dim_space):((j_ne + 1) * dim_space)]

    return cMass


def lumped_mass(num_nodes, dim_space, elem, jacobians, density):
    """
    Calculate the lumped mass matrix

    Parameters:
    num_nodes (int): Total number of nodes
    dim_space (int): Spatial dimension
    elem (ndarray): Connectivity table, shape: (num_elements, nne)
    jacobians (ndarray): Jacobians of each element, shape: (num_elements, )
    density (float): Material density

    Returns:
    lMass (ndarray): Lumped mass matrix (diagonal matrix), shape: (dim_space * num_nodes, dim_space * num_nodes)
    """
    num_elements, nne = elem.shape
    lMass = np.zeros((num_nodes * dim_space, num_nodes * dim_space))

    # Calculate lumped mass matrix
    for i_elem in range(num_elements):
        # For each element, distribute mass equally to its num_nodes
        for i_ne in range(nne):
            lMass[(elem[i_elem, i_ne] * dim_space):((elem[i_elem, i_ne] + 1) * dim_space),
            (elem[i_elem, i_ne] * dim_space):((elem[i_elem, i_ne] + 1) * dim_space)] += np.eye(dim_space) * jacobians[
                i_elem] * density / nne

    return lMass
