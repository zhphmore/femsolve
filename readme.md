# Non-linear Finite Element Method Solver

This is a python code for 2-spatial dimension problems using the non-linear Finite Element Method

The python code was developed by P.H. Zhang (2025) and B. Li (2010).

## Overview

This implementation solves 2-spatial dimension problems using the non-linear Finite Element Method with:

- C2D3 (triangular) or C2D4 (quadrilateral) elements
- Linear and bilinear shape functions
- Plane stress constitutive relations
- Explicit dynamic method

## File Structure

```
├── femsolve                            # Central python package for Finite Element Method
    └── ...
├── project1                            # Demonstration for femsolve
    ├── ... .pdf                        # Introdution to this project
    ├── main.py                         # Run to get data
    ├── generate_mesh.py                # Generate mesh for finite element analysis
    ├── boundary_conditions.py          # Set up the boundary conditions and initial conditions
    ├── settings.json                   # Configuations for this sample
    └── data                            # Store data
        ├── x_a.csv                     # Run main.py to get data
        └── ...
├── project2
    ├── ...
    ├── animate_displacement.py         # Animate displacement
    └── ...
├── project3
    └── ...
```

## How to run demonstration

You can regard /project1 as a demonstration.

1. **Read pdf to understand the problem**
2. **Modify settings.json to do settings**
3. **Run main.py to get data**
4. **Enter ./data to check the data**

## Project Introduction

### project1

A static linear elasticity problem on the trapezoidal panel (Cook's membrane) domain.
The vertical left edge is fixed. The bottom and the right vertical edges are traction free.
Traction is applied on the top horizontal.

### project2

A dynamic elastoplastic problem on the trapezoidal panel (Cook's membrane) domain
The vertical left edge is fixed. All other edges are traction free.
An initial velocity v0 in the vertical direction is applied to the right edge of the domain.

### project3

A dynamic elastoplastic problem on the rectangle panel domain.
There is no vertical displacement on the bottom and no horizontal displacement on the left.
All other edges are traction free.
Plane strain condition is considered.
An initial velocity v0 in the left direction is applied to the entire domain at time t == 0.

## Introduction to femsolve package

Please read /readme_femsolve.md carefully.

### Package file Structure

```
├── __init__.py
├── read_json.py                   # Read settings from json file
├── femstatic2d                    # Finite Element Method static problem
    ├── __init__.py
    ├── FEMStatic2D.py             # For static problem
    ├── g_center.py                # Calculate the barycenter and surface area of each element
    ├── shape_function.py          # Calculate linear and bilinear shape functions and their derivatives
    ├── B_matrix.py                # Calculate the B matrix
    ├── K_matrix_2D.py             # Global stiffness matrix assembly
    ├── constitutive_2D.py         # Compute strain, stress and pressure from displacement
    ├── plot_mesh.py               # Plot mesh
    └── plot_displacement.py       # Plot displacement
├── femdynamic2d                   # # Finite Element Method dynamic problem
    ├── __init__.py
    ├── FEMDynamic2D.py            # For dynamic problem
    ├── g_center.py                # Calculate the barycenter and surface area of each element
    ├── shape_function.py          # Calculate linear and bilinear shape functions and their derivatives
    ├── mass_matrix.py             # Calculate mass matrix
    ├── critical_time_step.py      # Calculate the critical time step
    ├── explicit_solver.py         # The Newmark Explicit Dynamics Analysis
    ├── force_internal.py          # Assemble the nodal internal force
    ├── constitutive.py            # Compute the stress tensor
    ├── plot_mesh.py               # Plot mesh
    └── plot_displacement.py       # Plot displacement at certain time
├── pyproject.toml                 # Package installation
├── requirements.txt               # Python dependencies
└── readme_femsolve.md             # This file  
```

### Package installation

1. **Clone or download the repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install numpy pandas matplotlib tqdm
   ```

## License

This code is provided as-is for educational and research purposes.
