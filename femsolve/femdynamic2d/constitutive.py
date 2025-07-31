import numpy as np


def constitutive(epsl, epsl_p, properties):
    """
    Compute the stress tensor
    
    Parameters:
    epsl (ndarray): Hencky strain tensor of total deformation, shape: (dim_space, dim_space)
    epsl_p (ndarray): Plastic strain tensor from last time step, shape: (dim_space, dim_space)
    properties (ndarray): Material properties
    
    Returns:
    sig_S (ndarray): Deviatoric part of stress tensor, shape: (dim_space, dim_space)
    sig_P (ndarray): Volumetric part of stress tensor, shape: (dim_space, dim_space)
    ep_eff (float): Effective plastic strain, scalar
    sig_eff (float): Effective von-Mises stress, scalar
    """
    # Material properties
    E = properties[1]  # Young's modulus
    nu = properties[2]  # Poisson's ratio
    G = E / (2 * (1 + nu))  # Shear modulus
    K = 2 * G / 3 * (1 + nu) / (1 - 2 * nu)  # Bulk modulus

    dim_space = epsl.shape[0]

    # Calculate elastic strain
    epsl_e = epsl - epsl_p

    # Calculate deviatoric elastic strain
    epsl_e_dev = epsl_e - (1 / dim_space) * np.trace(epsl_e) * np.eye(dim_space)

    # Calculate deviatoric stress
    sig_S = 2 * G * epsl_e_dev

    # Calculate volumetric stress
    sig_P = K * np.trace(epsl_e) * np.eye(dim_space)

    # Calculate effective plastic strain
    ep_eff = np.sqrt((2 / 3) * np.sum(epsl_p ** 2))

    # Calculate effective von-Mises stress
    sig_eff = np.sqrt((3 / 2) * np.sum(sig_S ** 2))

    return sig_S, sig_P, ep_eff, sig_eff
