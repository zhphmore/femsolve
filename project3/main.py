import os

import femsolve

from generate_mesh import generate_mesh
from boundary_conditions import boundary_conditions
from animate_displacement import animate_displacement


def main():
    # ********************
    # 0. Set the path

    file_settings = 'settings.json'
    folder_data = 'data\\'

    path_current = os.path.dirname(os.path.realpath(__file__))
    path_settings = os.path.join(path_current, file_settings)
    path_data = os.path.join(path_current, folder_data)

    # ********************
    # 1. Read settings from json

    print('Reading settings from json ...')
    geometry_set, mesh_set, boundaries_set, properties_set, solver_set, plotter_set = femsolve.read_json(path_settings)

    # ********************
    # 2. Generate mesh, initialize coordinates of num_nodes and element connectivity table
    # flag_elem_type: 1 for triangular element, 2 for quadrilateral element

    print('Generating mesh ...')
    # flag_elem_type: 1 for C2D3 triangular element, 2 for C2D4 quadrilateral element
    x_a, elem = generate_mesh(geometry_set, mesh_set)
    print("x_a shape: ", x_a.shape)
    print("elem shape: ", elem.shape)

    # ********************
    # 3. Boundary conditions

    print('Reading boundary conditions ...')
    fixed_area, u_init, velocity_init, force_external = boundary_conditions(x_a, boundaries_set)

    # ********************
    # 4. Initialize

    fem_solver = femsolve.FEMDynamic2D(x_a, elem, properties_set)
    # fem_solver.plotMesh()
    fem_solver.setBoundaryCondition(fixed_area, u_init, velocity_init, force_external)
    fem_solver.priorCompute(alpha_mass=solver_set[0], time_step_safe_factor=solver_set[1])
    fem_solver.savePriorData(path_save=path_data)

    # ********************
    # 5. Do explicit solution

    fem_solver.explicitSolver(gamma=solver_set[2], total_steps=solver_set[3])
    fem_solver.saveSolveResult(path_save=path_data)

    # ********************
    # 6. Show results
    u, epsilon, epsilon_p, sigma_SP, ep_eff, sig_eff = fem_solver.getResults()
    animate_displacement(x_a, elem, u, drawing_amplify=plotter_set[0], plot_interval_step=plotter_set[1],
                         path_save=path_data)

    print('Complete')


if __name__ == "__main__":
    main()
