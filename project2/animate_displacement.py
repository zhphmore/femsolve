import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_displacement(x_a, elem, u, drawing_amplify, plot_interval_step, path_save):
    """
    Animate displacement

    Parameters:
    x_a (ndarray): Initial nodal coordinates, shape: (num_nodes, dim_space)
    elem (ndarray): Element connectivity table, shape: (num_elements, nne)
    u (ndarray): Nodal displacements, shape: (total_steps, num_nodes, dim_space)
    drawing_amplify (float): Amplification factor
    plot_interval_step (int): Plot every plot_interval_step steps
    """
    num_elements, nne = elem.shape
    total_steps, num_nodes, dim_space = u.shape

    fig, ax = plt.subplots()

    ax.set_xlabel('x')
    ax.set_ylabel('y')

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
    axis_max = max(axis_x_max, axis_y_max)
    axis_min = min(axis_x_min, axis_y_min)

    # ax.set_xlim(axis_min, axis_max)
    # # ax.set_ylim(axis_min, axis_max)
    # ax.set_aspect('equal')
    ax.axis([axis_x_min, axis_x_max, axis_y_min, axis_y_max])

    # Use Line2D to update animation
    polygon_line, = ax.plot([], [], 'k-', lw=0.5)

    def update(i_frame):
        x_a_u = x_a + drawing_amplify * u[i_frame]

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

        # Reset axis range
        # ax.relim()
        # ax.autoscale_view()

        return polygon_line,

    # Start animation
    ani = FuncAnimation(fig, update, frames=total_steps, interval=plot_interval_step, blit=True)

    plt.show()

    print('Saving gif ...')
    os.makedirs(path_save, exist_ok=True)
    ani.save(os.path.join(path_save, 'transform.gif'), writer='pillow', fps=60)
