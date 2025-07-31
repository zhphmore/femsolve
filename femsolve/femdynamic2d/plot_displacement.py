import numpy as np
import matplotlib.pyplot as plt


def plot_displacement(x_a, elem, u, drawing_amplify, ts=0):
    """
    Plot displacement at certain time

    Parameters:
    x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    u (ndarray): Nodal displacements, shape: (num_nodes, dim_space)
    ts (int): Time for plotting
    drawing_amplify (float): Amplification factor

    """
    num_elements, nne = elem.shape

    x_max = max(x_a[:, 0])
    x_min = min(x_a[:, 0])
    x_range = x_max - x_min
    axis_x_max = x_max + 0.1 * x_range
    axis_x_min = x_min - 0.1 * x_range
    y_max = max(x_a[:, 1])
    y_min = min(x_a[:, 1])
    y_range = y_max - y_min
    axis_y_max = y_max + 0.1 * y_range
    axis_y_min = y_min - 0.1 * y_range

    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis([axis_x_min, axis_x_max, axis_y_min, axis_y_max])

    x_a_u = x_a + drawing_amplify * u[ts]

    polygon_line, = ax.plot([], [], 'k-', lw=0.5)

    # segs (ndarray), shape: (num_elements, nne, dim_space)
    segs = x_a_u[elem]
    # Connect the edges end to end, shape: (num_elements, nne + 1, dim_space)
    # segs[:, 0, :] is the same with segs[:, nne, :]
    segs = np.column_stack([segs, segs[:, :1, :]])
    # Insert nan between elements, shape: (num_elements, nne + 2, dim_space)
    # segs[:, nne + 1, :] is np.nan
    segs = np.insert(segs, (nne + 1), np.nan, axis=1)
    # Use ravel() is better than flatten()
    # segs[..., 0].ravel(), shape: (num_elements * (nne + 1), )
    polygon_line.set_data(segs[..., 0].ravel(), segs[..., 1].ravel())

    ax.relim()

    plt.show()
