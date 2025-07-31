# Finite Element Method Solver

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

## Installation

1. **Clone or download the repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install numpy pandas matplotlib tqdm
   ```

## Dependencies

- **NumPy**: Numerical computations and array operations
- **Pandas**: Data analysis and data manipulation 
- **Matplotlib**: Plotting and visualization
- **Tqdm**: Turns any iterable or loop into a smart progress bar, from the Arabic word taqaddum

## Notes

- The implementation assumes plane stress conditions

## License

This code is provided as-is for educational and research purposes.
