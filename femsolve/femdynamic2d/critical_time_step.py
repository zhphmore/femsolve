import numpy as np


def critical_time_step(jacobians, density, E, time_step_safe_factor=0.2):
    """
    Calculate the critical time step
    
    Parameters:
    jacobians (ndarray): Jacobians of each element (Surface areas of all the elements if 2D)
    density (float): Material density, kg/m3
    E (float): Young's modulus, Pa
    time_step_safe_factor (float): Safety factor
    
    Returns:
    time_step_val (float): Critical time step
    """
    # Minimum element size
    h_min = np.min(np.sqrt(jacobians))

    # Wave speed in solid
    velocity_solid = np.sqrt(E / density)

    # Critical time step
    time_step_val = time_step_safe_factor * h_min / velocity_solid

    return time_step_val
