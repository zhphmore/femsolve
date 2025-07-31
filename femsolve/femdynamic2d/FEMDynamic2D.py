import os
import numpy as np
import pandas as pd

from .g_center import g_center
from .shape_function import shape_function
from .mass_matrix import mass_matrix
from .critical_time_step import critical_time_step
from .explicit_solver import explicit_solver

from .plot_mesh import plot_mesh
from .plot_displacement import plot_displacement


class FEMDynamic2D:
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
        super(FEMDynamic2D, self).__init__()

        # Mesh
        self.x_a, self.elem = None, None
        self.num_nodes, self.dim_space, self.num_elements, self.nne = None, None, None, None
        self.setMesh(x_a, elem)

        # Material properties
        self.properties = np.zeros(6)
        self.properties[:min(len(properties), 6)] = properties

        # Boundary conditions
        self.fixed_area, self.u_init, self.velocity_init, self.force_external = None, None, None, None

        # Geometric center and jacobians
        self.xg, self.jacobians = None, None

        # Shape functions and their derivatives at quadrature points
        self.p, self.dp = None, None

        # Mass matrix
        self.alpha_mass = 0
        self.Mass = None

        # Critical time step
        self.time_step_safe_factor = 0.2
        self.time_step_val = None

        # Solution result
        self.u = None  # nodal displacements
        self.epsilon = None
        self.epsilon_p = None
        self.sigma_SP = None
        self.ep_eff = None  # effective plastic strain
        self.sig_eff = None  # von-Mises stresses

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

    def setBoundaryCondition(self, fixed_area, u_init, velocity_init, force_external):
        """
        Set up the boundary conditions and initial conditions

        Parameters:
        fixed_area (ndarray): Displacement boundary conditions (1 for fixed, 0 for free), shape: (num_nodes, dim_space)
        u_init (ndarray): Initial nodal displacements, shape: (num_nodes, dim_space)
        velocity_init (ndarray): Initial nodal velocity, shape: (num_nodes, dim_space)
        force_external (ndarray): Global force vector, shape: (num_nodes, dim_space)
        """
        if not (fixed_area.shape == (self.num_nodes, self.dim_space)):
            raise ValueError(
                "expect fixed_area shape ({}, {}), but found shape {}".format(self.num_nodes, self.dim_space,
                                                                              fixed_area.shape))
        if not (u_init.shape == (self.num_nodes, self.dim_space)):
            raise ValueError(
                "expect u_init shape ({}, {}), but found shape {}".format(self.num_nodes, self.dim_space,
                                                                          u_init.shape))
        if not (velocity_init.shape == (self.num_nodes, self.dim_space)):
            raise ValueError(
                "expect velocity_init shape ({}, {}), but found shape {}".format(self.num_nodes, self.dim_space,
                                                                                 velocity_init.shape))
        if not (force_external.shape == (self.num_nodes, self.dim_space)):
            raise ValueError(
                "expect force_external shape ({}, {}), but found shape {}".format(self.num_nodes, self.dim_space,
                                                                                  force_external.shape))

        self.fixed_area, self.u_init, self.velocity_init, self.force_external = fixed_area, u_init, velocity_init, force_external

    def priorCompute(self, alpha_mass=0, time_step_safe_factor=0.2):
        """
        Do some necessary computations

        Parameters:
        alpha_mass (float): Combination factor of consistent mass matrix and lumped mass matrix
        time_step_safe_factor (float): Safety factor

        Returns:
        xg (ndarray): Barycenters of all the elements, shape: (num_elements, dim_space)
        jacobians (ndarray): Jacobians of each element, shape: (num_elements, )
        p (ndarray): Values of the shape function at the barycenters, shape: (num_elements, nne)
        dp (ndarray): Derivatives of the shape function at the barycenters, shape: (num_elements, nne, dim_space)
        Mass (ndarray): Mass matrix, shape: (dim_space * num_nodes, dim_space * num_nodes)
        time_step_val (float): Critical time step
        """
        # check input
        if not (0 <= alpha_mass <= 1):
            raise ValueError("alpha_mass must be between 0 and 1, but found {}".format(alpha_mass))
        if not (0 < time_step_safe_factor <= 1):
            raise ValueError(
                "time_step_safe_factor must be between 0 and 1, but found {}".format(time_step_safe_factor))

        self.alpha_mass = alpha_mass
        self.time_step_safe_factor = time_step_safe_factor

        # Geometric center and jacobians
        print('Computing barycenters and jacobians ...')
        self.xg, self.jacobians = g_center(self.x_a, self.elem)

        # Shape functions and their derivatives at quadrature points
        print('Computing shape function ...')
        self.p, self.dp = shape_function(self.x_a, self.elem, self.xg, self.jacobians)

        # Mass matrix
        print('Computing mass matrix ...')
        self.Mass = mass_matrix(self.alpha_mass, self.num_nodes, self.dim_space, self.elem, self.p, self.jacobians,
                                self.properties[0])
        print("Mass shape: ", self.Mass.shape)

        # Calculate critical time step
        print('Setting critical time step ...')
        self.time_step_val = critical_time_step(self.jacobians, self.properties[0], self.properties[1],
                                                self.time_step_safe_factor)
        print('Critical time step: {}'.format(self.time_step_val))

    def explicitSolver(self, gamma, total_steps):
        """
        The Newmark Explicit Dynamics Analysis

        Parameters:
        time_step_val (float): Time step size
        gamma (float): Time integration parameter (0.5 for central difference)

        Returns:
        u (ndarray): Nodal displacements at each time step, shape: (total_steps, num_nodes, dim_space)
        epsilon (ndarray): Total strain at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        epsilon_p (ndarray): Plastic strain at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        sigma_SP (ndarray): Stress at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        ep_eff (ndarray): Effective plastic strain at each time step, shape: (total_steps, num_elements)
        sig_eff (ndarray): Effective von-Mises stress at each time step, shape: (total_steps, num_elements)
        """
        # check value
        if (self.fixed_area is None) or (self.u_init is None) or (self.velocity_init is None) or (
                self.force_external is None):
            raise ValueError("Please use setBoundaryCondition() first")
        if self.Mass is None:
            raise ValueError("Please use baseCompute() first")

        # check input
        total_steps = max(int(total_steps), 1)

        self.u, self.epsilon, self.epsilon_p, self.sigma_SP, self.ep_eff, self.sig_eff = explicit_solver(self.x_a,
                                                                                                         self.elem,
                                                                                                         self.fixed_area,
                                                                                                         self.u_init,
                                                                                                         self.velocity_init,
                                                                                                         self.force_external,
                                                                                                         self.jacobians,
                                                                                                         self.dp,
                                                                                                         self.properties,
                                                                                                         self.Mass,
                                                                                                         gamma,
                                                                                                         self.time_step_val,
                                                                                                         total_steps)

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

        # p (list): Values of the shape function at the barycenters
        df_N = pd.DataFrame(self.p)
        df_N.to_csv(os.path.join(path_save, 'p.csv'), header=False, index=False, encoding='utf-8')

        # dp (list): Derivatives of the shape function at the barycenters
        df_DN = pd.DataFrame(self.dp.reshape(self.num_elements, -1))
        df_DN.to_csv(os.path.join(path_save, 'dp.csv'), header=False, index=False, encoding='utf-8')

        # Mass (ndarray): Mass matrix
        df_Mass = pd.DataFrame(self.Mass)
        df_Mass.to_csv(os.path.join(path_save, 'Mass.csv'), header=False, index=False, encoding='utf-8')

        print('Prior data saved, path: {}'.format(path_save))

    def saveSolveResult(self, path_save):
        """
        Save the solution results to disk

        path_save (string): the saving path
        """
        print('Save the solution results to csv ...')
        os.makedirs(path_save, exist_ok=True)

        # u (ndarray): Nodal displacements at each time step
        df_u = pd.DataFrame(self.u.reshape(-1, self.num_nodes * self.dim_space))
        df_u.to_csv(os.path.join(path_save, 'u.csv'), header=False, index=False, encoding='utf-8')

        # epsilon (ndarray): Total strain at each time step
        df_epsilon = pd.DataFrame(self.epsilon.reshape(-1, self.num_elements * self.dim_space * self.dim_space))
        df_epsilon.to_csv(os.path.join(path_save, 'epsilon.csv'), header=False, index=False, encoding='utf-8')

        # epsilon_p (ndarray): Plastic strain at each time step
        df_epsilon_p = pd.DataFrame(self.epsilon_p.reshape(-1, self.num_elements * self.dim_space * self.dim_space))
        df_epsilon_p.to_csv(os.path.join(path_save, 'epsilon_p.csv'), header=False, index=False, encoding='utf-8')

        # sigma_SP (ndarray): Stress at each time step
        df_sigma_SP = pd.DataFrame(self.sigma_SP.reshape(-1, self.num_elements * self.dim_space * self.dim_space))
        df_sigma_SP.to_csv(os.path.join(path_save, 'sigma_SP.csv'), header=False, index=False, encoding='utf-8')

        # ep_eff (ndarray): Effective plastic strain at each time step
        df_ep_eff = pd.DataFrame(self.ep_eff.reshape(-1, self.num_elements))
        df_ep_eff.to_csv(os.path.join(path_save, 'ep_eff.csv'), header=False, index=False, encoding='utf-8')

        # sig_eff (ndarray): Effective von-Mises stress at each time step
        df_sig_eff = pd.DataFrame(self.sig_eff.reshape(-1, self.num_elements))
        df_sig_eff.to_csv(os.path.join(path_save, 'sig_eff.csv'), header=False, index=False, encoding='utf-8')

        print('Solution results saved, path: {}'.format(path_save))

    def plotMesh(self):
        """
        Plot mesh

        Returns:
        u (ndarray): Nodal displacements at each time step, shape: (total_steps, num_nodes, dim_space)
        epsilon (ndarray): Total strain at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        epsilon_p (ndarray): Plastic strain at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        sigma_SP (ndarray): Stress at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        ep_eff (ndarray): Effective plastic strain at each time step, shape: (total_steps, num_elements)
        sig_eff (ndarray): Effective von-Mises stress at each time step, shape: (total_steps, num_elements)
        """
        plot_mesh(self.x_a, self.elem)

    def plotDisplacement(self, drawing_amplify, ts=0):
        """
        Plot dynamic results

        Parameters:
        drawing_amplify (float): Amplification factor
        plot_interval_step (int): Plot every plot_interval_step steps
        path_save (string): the saving path
        """
        plot_displacement(self.x_a, self.elem, self.u, drawing_amplify, ts)

    def getResults(self):
        """
        Get computation results

        Returns:
        u (ndarray): Nodal displacements at each time step, shape: (total_steps, num_nodes, dim_space)
        epsilon (ndarray): Total strain at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        epsilon_p (ndarray): Plastic strain at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        sigma_SP (ndarray): Stress at each time step, shape: (total_steps, num_elements, dim_space ** 2)
        ep_eff (ndarray): Effective plastic strain at each time step, shape: (total_steps, num_elements)
        sig_eff (ndarray): Effective von-Mises stress at each time step, shape: (total_steps, num_elements)
        """
        return self.u, self.epsilon, self.epsilon_p, self.sigma_SP, self.ep_eff, self.sig_eff
