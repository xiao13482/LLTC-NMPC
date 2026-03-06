"""
Nonlinear Model Predictive Control (NMPC) Solver

This module formulates and solves the NMPC optimization problem using CasADi and IPOPT.
It supports both standard quadratic terminal costs and Neural Network-based Lyapunov terminal costs.

Author: Xiao Guoliang
        Harbin Institute of Technology, Harbin
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import casadi as ca
import config
from .dynamic import CasADi_Dynamic
from Lyapunov_train import LowerTriangularMatrixNet
import torch
import torch.nn as nn


class MPC_solver:
    def __init__(self, print_time=False, neural_cost = False):
        """
        Initialize the NMPC solver.

        Args:
            print_time (bool): Whether to print the solver execution time.
            neural_cost (bool): If True, use the learned Lyapunov neural network for terminal cost.
                                If False, use standard quadratic terminal cost.
        """
        ## MPC params ##
        # --- Global Configurations ---
        self.cfg = config.Config()
        self.state_dim = self.cfg.state_dim  # Dimension of state (3)
        self.inputs_dim = self.cfg.inputs_dim  # Dimension of inputs
        self.dt_MPC = self.cfg.DT_SIM  # Time step
        # When using Neural Cost, prediction horizon can be significantly reduced (e.g., N=1)
        self.N = 1 if neural_cost else self.cfg.PREDICTION_HORIZON
        self.neural_cost = neural_cost          #neural terminal cost

        # --- Neural Network Initialization (If enabled) ---
        self.device = torch.device('cpu')
        pt_dim = self.cfg.state_dim * 2 + self.cfg.inputs_dim

        self.model = LowerTriangularMatrixNet(
            input_dim=pt_dim,
            state_dim=self.cfg.state_dim,
            hidden_layers=[40, 40, 40],
            epsilon=0.001,
            activation=nn.ReLU()
        ).to(self.device)

        model_dir = os.path.join(project_root, 'models')
        model_path = os.path.join(model_dir, 'lyapunov_terminal_cost_best.pth')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # --- Performance Index Weight Matrices ---
        self.Q = np.diag(self.cfg.Q)
        self.R = np.diag(self.cfg.R)
        self.R_2 = np.diag(self.cfg.R_2)

        # --- Dynamics Setup ---
        dynamic = CasADi_Dynamic()
        self.f_dynamic = dynamic.get_casadi_functions()  # casadi符号形式动力学

        # Initialize CasADi Opti stack
        self.opti = ca.Opti()

        # --- Optimization Variables ---
        self.X = self.opti.variable(self.state_dim, self.N + 1)  # 状态序列，(N+1)*dim_x维
        self.U = self.opti.variable(self.inputs_dim, self.N)  # 控制序列，N*dim_u维

        # --- Optimization Parameters (Values set dynamically at each solving step) ---
        self.x_init = self.opti.parameter(self.state_dim)  # MPC中的x_0
        self.x_ref = self.opti.parameter(self.state_dim)  # 参考轨迹
        self.u_ref = self.opti.parameter(self.inputs_dim) # 参考输入
        self.P = self.opti.parameter(self.state_dim, self.state_dim)

        # Build the optimization objective and constraints
        self._build_optimization_problem()

        # --- Solver Configuration (IPOPT) ---
        opts = {
            'ipopt.print_level': 0,  # 0: Silent, 5: Detailed IPOPT logs
            'ipopt.sb': 'yes',  # Suppress startup banner
            'print_time': 0 if not print_time else 1,
            'ipopt.max_iter': 500,  # Maximum iterations
            'ipopt.tol': 1e-6,  # Convergence tolerance
            'expand': True  # Expand mathematical operations for faster solving
        }
        self.opti.solver('ipopt', opts)

        # Variables to store results for warm-starting the next MPC step
        self.last_X = np.zeros([self.state_dim, self.N + 1])
        self.last_U = np.zeros([self.inputs_dim, self.N])

    def get_terminal_cost_matrix(self, current_state, ref_state, ref_input=np.zeros(2)):
        """
        Uses the Neural Network to predict the positive definite terminal cost matrix P.
        """
        # Concatenate parameter vector p_t = [x, x_r, u_r]
        p_t_numpy = np.concatenate([current_state, ref_state, ref_input])

        # Convert to PyTorch Tensor and add batch dimension (shape: [1, 8])
        p_t_tensor = torch.tensor(p_t_numpy, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass (No gradient computation needed during inference)
        with torch.no_grad():
            _, P_tensor = self.model.forward(p_t_tensor)

        P_numpy = P_tensor.squeeze(0).cpu().numpy()

        return P_numpy

    def _build_optimization_problem(self):
        """
        Constructs the NLP problem using CasADi Opti.
        Defines the cost function (minimize) and constraints (subject_to).
        """
        # --- 1. Stage Cost (Running Cost) ---
        cost = 0
        # stage cost
        for k in range(self.N):
            # J = (x - x_ref)^T * Q * (x - x_ref) + u^T * R * u
            state_error = self.X[:, k] - self.x_ref
            cost += ca.mtimes([state_error.T, self.Q, state_error])

            # Control effort cost
            control = self.U[:, k]
            cost += ca.mtimes([control.T, self.R, control])

        # terminal cost
        state_error = self.X[:, self.N] - self.x_ref
        cost += ca.mtimes([state_error.T, self.P, state_error])
        self.opti.minimize(cost)

        # --- Constraints ---
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.x_init)

        for k in range(self.N):
            # System Dynamics Constraint (Multiple Shooting method)
            x_next = self.f_dynamic(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k + 1] == x_next)

            # State Bounds Constraints
            self.opti.subject_to(self.opti.bounded(self.cfg.x_min, self.X[0, k], self.cfg.x_max))
            self.opti.subject_to(self.opti.bounded(self.cfg.y_min, self.X[1, k], self.cfg.y_max))
            self.opti.subject_to(self.opti.bounded(self.cfg.phi_min, self.X[2, k], self.cfg.phi_max))

            # Control Bounds Constraints
            self.opti.subject_to(self.opti.bounded(self.cfg.v_min, self.U[0, k], self.cfg.v_max))
            self.opti.subject_to(self.opti.bounded(self.cfg.delta_min, self.U[1, k], self.cfg.delta_max))

            # Obstacle Avoidance Constraint (Cylindrical keep-out zone)
            distance = (self.X[0, k] - self.cfg.cylinder_cx)**2 + (self.X[1, k] - self.cfg.cylinder_cy)**2
            self.opti.subject_to(distance >= self.cfg.cylinder_r ** 2)

        # Terminal State Constraints (Including obstacle avoidance for the terminal state)
        self.opti.subject_to(self.opti.bounded(self.cfg.x_min, self.X[0, self.N], self.cfg.x_max))
        self.opti.subject_to(self.opti.bounded(self.cfg.y_min, self.X[1, self.N], self.cfg.y_max))
        self.opti.subject_to(self.opti.bounded(self.cfg.phi_min, self.X[2, self.N], self.cfg.phi_max))
        distance_N = (self.X[0, self.N] - self.cfg.cylinder_cx) ** 2 + (self.X[1, self.N] - self.cfg.cylinder_cy) ** 2
        self.opti.subject_to(distance_N >= self.cfg.cylinder_r ** 2)

    def solve(self, current_state, ref_state, u_ref = np.zeros(2), detail = False, raise_on_fail = False):
        """
        Solves the MPC optimization problem for the current time step.

        Args:
            current_state (ndarray): Current vehicle state [x, y, phi].
            ref_state (ndarray): Target reference state.
            u_ref (ndarray): Target reference control input.
            detail (bool): If True, returns full optimal trajectory and control sequence.
            raise_on_fail (bool): If True, raises RuntimeError upon solver failure.

        Returns:
            Optimal control input (if detail=False) OR full solution matrices (if detail=True).
        """
        # Set current parameters for the optimization problem
        self.opti.set_value(self.x_init, current_state)
        self.opti.set_value(self.x_ref, ref_state)
        self.opti.set_value(self.u_ref, u_ref)

        # Compute terminal cost matrix dynamically if using Neural Cost
        if self.neural_cost:
            P = self.get_terminal_cost_matrix(current_state, ref_state, u_ref)
        else:
            P = np.diag(self.cfg.P)
        self.opti.set_value(self.P, P)

        #Warm Start
        # U_guess = ca.horzcat(self.last_U[:, 1:], np.zeros([self.inputs_dim, 1]))
        # X_guess = ca.horzcat(self.last_X[:, 1:], self.last_X[:, -1])
        # self.opti.set_initial(self.U, U_guess)
        # self.opti.set_initial(self.X, X_guess)

        # solve
        try:
            sol = self.opti.solve()

            # Convert CasADi numerical values to NumPy arrays
            u_optimal = sol.value(self.U[:, 0])  # 取第一个控制量
            path_predict = sol.value(self.X)  # 取预测轨迹用于画图
            self.last_X = sol.value(self.X)   # 存储当前求解结果用于下次热启动
            self.last_U = sol.value(self.U)

            if detail:
                return sol.value(self.U), sol.value(self.X)
            else:
                return u_optimal, path_predict
        except RuntimeError as e:
            # Error handling strategy
            if raise_on_fail:
                error_msg = f"[Warning] MPC Solver Failed"
                raise RuntimeError(error_msg) from e
            else:
                print(f'current_state: {current_state}, ref_state: {ref_state}')
                sys.exit(f"[Warning] MPC Solver Failed: {e}")



