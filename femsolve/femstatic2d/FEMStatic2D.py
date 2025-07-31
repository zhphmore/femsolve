import os
import numpy as np
import pandas as pd

from .g_center import g_center
from .shape_function import shape_function
from .B_matrix import B_matrix
from .K_matrix_2D import K_matrix_2D
from .constitutive_2D import constitutive_2D

from .plot_mesh import plot_mesh
from .plot_displacement import plot_displacement


class FEMStatic2D:
    def __init__(self, x_a, elem, properties):
        """
        Parameters:
        x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
            num_nodes (int): Total number of nodes
            dim_space (int): Spatial dimension
            Only support 2D, dim_space == 2
        elem (ndarray): Element connectivity table, shape: (num_elements, nne)
            num_elements (int): Total number of elements
            nne (int): Number of nodal elements
            Only support C2D3 or C2D4, nne == 3 or 4
        properties (ndarray): Material properties, including:
            0. Density, kg/m3
            1. Young's modulus, Pa
            2. Poisson's ratio
            3. Initial yield stress, Pa
            4. Hardening modulus
            5. Power law exponent
        """
        super(FEMStatic2D, self).__init__()

        # Mesh
        self.x_a, self.elem = None, None
        self.num_nodes, self.dim_space, self.num_elements, self.nne = None, None, None, None
        self.setMesh(x_a, elem)

        # Material properties
        self.properties = np.zeros(6)
        self.properties[:min(len(properties), 6)] = properties

        # Boundary conditions
        self.fixed_area, self.u_init, self.force_external = None, None, None

        # Geometric center and jacobians
        self.xg, self.jacobians = None, None

        # Shape functions and their derivatives at quadrature points
        self.p, self.dp = None, None

        # B matrix
        self.B = None
        # K matrix
        self.K = None

        # Solution result
        self.u = None  # nodal displacements
        self.Es = None  # strain vector
        self.Ss = None  # stress vector
        self.P = None  # pressure vector

    def setMesh(self, x_a, elem):
        """
        Set up mesh for finite element analysis

        Parameters:
        x_a (ndarray): Node coordinates, shape: (num_nodes, dim_space)
        elem (ndarray): Element connectivity table, shape: (num_elements, nne)
        """
        # check input
        if not x_a.shape[1] == 2:
            raise ValueError("x_a must be 2D, expect shape (:, 2), but found shape (:, {})".format(self.dim_space))
        if not (elem.shape[1] == 3 or 4):
            raise ValueError(
                "number of nodal elements must be 2D, expect shape (:, 3) or (:, 4), but found shape (:, {})".format(
                    self.nne))

        self.x_a = x_a
        self.num_nodes, self.dim_space = self.x_a.shape
        self.elem = elem
        self.num_elements, self.nne = self.elem.shape

    def setBoundaryCondition(self, fixed_area, u_init, force_external):
        """
        Set up the boundary conditions and initial conditions

        Parameters:
        fixed_area (ndarray): Displacement boundary conditions (1 for fixed, 0 for free), shape: (num_nodes, dim_space)
        u_init (ndarray): Initial nodal displacements, shape: (num_nodes, dim_space)
        force_external (ndarray): Global force vector, shape: (num_nodes, dim_space)
        """
        # check input
        if not (fixed_area.shape == (self.num_nodes, self.dim_space)):
            raise ValueError(
                "expect fixed_area shape ({}, {}), but found shape {}".format(self.num_nodes, self.dim_space,
                                                                              fixed_area.shape))
        if not (u_init.shape == (self.num_nodes, self.dim_space)):
            raise ValueError(
                "expect u_init shape ({}, {}), but found shape {}".format(self.num_nodes, self.dim_space,
                                                                          u_init.shape))
        if not (force_external.shape == (self.num_nodes, self.dim_space)):
            raise ValueError(
                "expect force_external shape ({}, {}), but found shape {}".format(self.num_nodes, self.dim_space,
                                                                                  force_external.shape))

        self.fixed_area, self.u_init, self.force_external = fixed_area, u_init, force_external

    def priorCompute(self):
        """
        Do some necessary computations

        Returns:
        xg (ndarray): Barycenters of all the elements, shape: (num_elements, dim_space)
        jacobians (ndarray): Jacobians of each element, shape: (num_elements, )
        p (ndarray): Values of the shape function at the barycenters, shape: (num_elements, nne)
        dp (ndarray): Derivatives of the shape function at the barycenters, shape: (num_elements, nne, dim_space)
        B (ndarray): list of B matrices for all elements, shape: (num_elements, row_B, nne * dim_space)
        K (ndarray): global stiffness matrix, shape: (num_nodes * dim_space, num_nodes * dim_space)
        """
        # Geometric center and jacobians
        print('Computing barycenters and jacobians ...')
        self.xg, self.jacobians = g_center(self.x_a, self.elem)

        # Shape functions and their derivatives at quadrature points
        print('Computing shape function ...')
        self.p, self.dp = shape_function(self.x_a, self.elem, self.xg, self.jacobians)

        # Compute B matrix
        print('Computing B matrix ...')
        self.B = B_matrix(self.dim_space, self.num_elements, self.nne, self.dp)
        print("B matrix shape: ", self.B.shape)

        # Compute K matrix
        print('Computing K matrix ...')
        self.K = K_matrix_2D(self.num_nodes, self.dim_space, self.elem, self.jacobians, self.properties, self.B)

    def staticSolver(self):
        """
        The Newmark Explicit Dynamics Analysis

        Returns:
        u (ndarray): Nodal displacements, shape: (num_nodes, dim_space)
        Es (ndarray): Strain vector, shape: (num_elements, row_B)
        Ss (ndarray): Stress vector, shape: (num_elements, row_B)
        P (ndarray): Pressure vector, shape: (num_elements, )
        """
        # check value
        if (self.fixed_area is None) or (self.u_init is None) or (self.force_external is None):
            raise ValueError("Please use setBoundaryCondition() first")
        if self.K is None:
            raise ValueError("Please use baseCompute() first")

        # Zero-one method
        # For fixed position points, set K matrix diagonal element to 1, other row/column elements to 0
        # Also set corresponding external force force_external to 0
        force_total = self.force_external.ravel()

        for i in range(self.num_nodes * self.dim_space):
            if self.fixed_area.ravel()[i] == 1:
                self.K[:, i] = 0
                self.K[i, :] = 0
                self.K[i, i] = 1
                force_total[i] = 0

        self.u = np.linalg.solve(self.K, force_total)
        self.u = self.u.reshape(self.num_nodes, self.dim_space)

        self.Es, self.Ss, self.P = constitutive_2D(self.dim_space, self.elem, self.B, self.properties, self.u)

    def savePriorData(self, path_save):
        """
        Save the prior data to disk

        path_save (string): the saving path
        """
        print('Save the prior data to csv ...')
        os.makedirs(path_save, exist_ok=True)

        # x_a (ndarray): Nodal coordinates
        df_x_a = pd.DataFrame(self.x_a)
        df_x_a.to_csv(os.path.join(path_save, 'x_a.csv'), header=False, index=False, encoding='utf-8')

        # elem (ndarray): Connectivity table
        df_elem = pd.DataFrame(self.elem)
        df_elem.to_csv(os.path.join(path_save, 'elem.csv'), header=False, index=False, encoding='utf-8')

        # jacobians (ndarray): Jacobians of each element (Surface areas of all the elements if 2D)
        df_jacobians = pd.DataFrame(self.jacobians)
        df_jacobians.to_csv(os.path.join(path_save, 'jacobians.csv'), header=False, index=False, encoding='utf-8')

        # p (ndarray): Values of the shape function at the barycenters
        df_N = pd.DataFrame(self.p)
        df_N.to_csv(os.path.join(path_save, 'p.csv'), header=False, index=False, encoding='utf-8')

        # dp (ndarray): Derivatives of the shape function at the barycenters
        df_DN = pd.DataFrame(self.dp.reshape(self.num_elements, -1))
        df_DN.to_csv(os.path.join(path_save, 'dp.csv'), header=False, index=False, encoding='utf-8')

        # B (ndarray): B matrix
        df_B = pd.DataFrame(self.B.reshape(self.num_elements, -1))
        df_B.to_csv(os.path.join(path_save, 'B.csv'), header=False, index=False, encoding='utf-8')

        # K (ndarray): K matrix
        df_K = pd.DataFrame(self.K)
        df_K.to_csv(os.path.join(path_save, 'K.csv'), header=False, index=False, encoding='utf-8')

        print('Prior data saved, path: {}'.format(path_save))

    def saveSolveResult(self, path_save):
        """
        Save the solution results to disk

        path_save (string): the saving path
        """
        print('Save the solution results to csv ...')
        os.makedirs(path_save, exist_ok=True)

        # u (ndarray): Nodal displacements
        df_u = pd.DataFrame(self.u.reshape(self.num_nodes, self.dim_space))
        df_u.to_csv(os.path.join(path_save, 'u.csv'), header=False, index=False, encoding='utf-8')

        # Es (ndarray): strain vector
        df_Es = pd.DataFrame(self.Es)
        df_Es.to_csv(os.path.join(path_save, 'Es.csv'), header=False, index=False, encoding='utf-8')

        # Ss (ndarray): stress vector
        df_Ss = pd.DataFrame(self.Ss)
        df_Ss.to_csv(os.path.join(path_save, 'Ss.csv'), header=False, index=False, encoding='utf-8')

        # P (ndarray): pressure vector
        df_P = pd.DataFrame(self.P)
        df_P.to_csv(os.path.join(path_save, 'P.csv'), header=False, index=False, encoding='utf-8')

        print('Solution results saved, path: {}'.format(path_save))

    def plotMesh(self):
        """
        Plot mesh
        """
        plot_mesh(self.x_a, self.elem)

    def plotDisplacement(self, drawing_amplify):
        """
        Plot dynamic results

        Parameters:
        drawing_amplify (float): Amplification factor
        path_save (string): the saving path
        """
        plot_displacement(self.x_a, self.elem, self.u, drawing_amplify)

    def getResults(self):
        """
        Get computation results

        Returns:
        u (ndarray): Nodal displacements, shape: (num_nodes, dim_space)
        Es (ndarray): Strain vector, shape: (num_elements, row_B)
        Ss (ndarray): Stress vector, shape: (num_elements, row_B)
        P (ndarray): Pressure vector, shape: (num_elements, )
        """
        return self.u, self.Es, self.Ss, self.P
