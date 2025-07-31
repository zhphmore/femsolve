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
├── femsolve                            # Central python package for Finite Element Method
    └── ...
├── project1                            # Demonstration for femsolve
    ├── midterm_exam_2024.pdf           # Introdution to this project
    ├── main.py                         # Run to get data
    ├── generate_mesh.py                # Generate mesh for finite element analysis
    ├── boundary_conditions.py          # Set up the boundary conditions and initial conditions
    ├── settings.json                   # Configuations for this sample
    └── data                            # Store data
        ├── x_a.csv                     # Run main.py to get data
        └── ...
├── project2
    └── ...
├── project3
    └── ...
```

## Introduction to femsolve package

Please read /readme_femsolve.md carefully.

## Installation

**Clone or download the repository**

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

## License

This code is provided as-is for educational and research purposes.
