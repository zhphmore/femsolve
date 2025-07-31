import os

import femsolve

from generate_mesh import generate_mesh
from boundary_conditions import boundary_conditions


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
    fixed_area, u_init, force_external = boundary_conditions(x_a, elem, boundaries_set)

    # ********************
    # 4. Initialize

    fem_solver = femsolve.FEMStatic2D(x_a, elem, properties_set)
    # fem_solver.plotMesh()
    fem_solver.setBoundaryCondition(fixed_area, u_init, force_external)
    fem_solver.priorCompute()
    fem_solver.savePriorData(path_save=path_data)

    # ********************
    # 5. Do solution

    fem_solver.staticSolver()
    fem_solver.saveSolveResult(path_save=path_data)

    # ********************
    # 6. Show and results

    fem_solver.plotDisplacement(drawing_amplify=plotter_set[0])

    print('Complete')


if __name__ == "__main__":
    main()
