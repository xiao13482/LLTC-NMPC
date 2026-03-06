"""
Vehicle Dynamics Models

Implements the Kinematic Bicycle Model for the vehicle.
Equations:
    dot_x = v * cos(phi)
    dot_y = v * sin(phi)
    dot_phi = delta

Author: Xiao Guoliang
        Harbin Institute of Technology, Harbin
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import casadi as ca
import config
import numpy as np

class CasADi_Dynamic:
    """
    Symbolic dynamics model using CasADi.
    Used for formulating the NMPC optimization problem.
    """
    def __init__(self):
        cfg = config.Config()
        self.dt_MPC = cfg.DT_SIM

    def dynamic(self, current_state, inputs):
        x_k = current_state[0]
        y_k = current_state[1]
        phi_k = current_state[2]
        v_k = inputs[0]
        delta_k = inputs[1]
        dot_x = v_k * ca.cos(phi_k)
        dot_y = v_k *ca.sin(phi_k)
        dot_phi = delta_k

        return ca.vertcat(dot_x, dot_y, dot_phi)

    def rk4_step(self, x, u, dt):
        """
        4th-order Runge-Kutta (RK4) numerical integration.
        """
        k1 = self.dynamic(x, u)
        k2 = self.dynamic(x + 0.5 * dt * k1, u)
        k3 = self.dynamic(x + 0.5 * dt * k2, u)
        k4 = self.dynamic(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_casadi_functions(self):
        state_sym = ca.MX.sym('states', 3)
        inputs_sym = ca.MX.sym('inputs', 2)

        state_next = self.rk4_step(state_sym, inputs_sym, self.dt_MPC)
        return ca.Function('dynamic_step', [state_sym, inputs_sym], [state_next],
                           ['x', 'u'], ['x_next'])

class Sim_Dynamic:
    """
    Symbolic dynamics model using numpy.
    Used for simulating the system.
    """
    def __init__(self):
        cfg = config.Config()
        self.dt_MPC = cfg.DT_SIM

    def dynamic(self, current_state, inputs):
        x_k = current_state[0]
        y_k = current_state[1]
        phi_k = current_state[2]
        v_k = inputs[0]
        delta_k = inputs[1]
        dot_x = v_k * np.cos(phi_k)
        dot_y = v_k * np.sin(phi_k)
        dot_phi = delta_k

        return np.array([dot_x, dot_y, dot_phi])

    def rk4_step(self, x, u, dt):
        """
        4th-order Runge-Kutta (RK4) numerical integration.
        """
        k1 = self.dynamic(x, u)
        k2 = self.dynamic(x + 0.5 * dt * k1, u)
        k3 = self.dynamic(x + 0.5 * dt * k2, u)
        k4 = self.dynamic(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
