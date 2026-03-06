"""
Dataset Collection Module

Executes Algorithm 1 from the reference paper to generate training data.
Iteratively samples initial states and computes the optimal open-loop trajectory
using standard MPC to collect state-cost pairs.
"""

import sys
import numpy as np
import random
import pickle
import os
import config
import utils
from MPC.NMPC_solver import MPC_solver

def data_collect(cfg = config.Config()):
    """
    Executes the data collection loop (Algorithm 1).
    :return: Dataset D containing training samples.
    """
    random.seed(cfg.seed)       # Fix random seed for reproducibility
    np.random.seed(cfg.seed)

    mpc = MPC_solver()
    d_solver = utils.dp_solver()
    Q = np.diag(cfg.Q)
    R = np.diag(cfg.R)
    P = np.diag(cfg.P)

    Dataset = []                # List to store successful data samples

    i = 0
    while True:

        low = np.array([cfg.x_min, cfg.y_min, cfg.phi_min])
        high = np.array([cfg.x_max, cfg.y_max, cfg.phi_max])
        state_init = np.random.uniform(low=low, high=high)
        state_target = np.zeros(3)
        inputs_target = np.zeros(2)

        # Construct parameter vector dictionary
        pt_i = {
            'x_0': state_init,
            'x_r': state_target,
            'u_r': inputs_target
        }

        # Solve CDA_NMPC to get the optimal control sequence (U_i) and state trajectory (X_i)
        try:
            U_i, X_i = mpc.solve(state_init, state_target, detail = True, raise_on_fail = True)
            x_1 = X_i[:, 1]
            state_error_0 = X_i[:, 0] - state_target
            control_0 = U_i[:, 0]
            l_0 = state_error_0.T @ Q @ state_error_0 + control_0.T @ R @ control_0

            # Compute the total optimal cost J_CDA over the horizon
            J_CDA = 0
            for j in range(cfg.PREDICTION_HORIZON):
                state_error = X_i[:, j] - state_target
                J_CDA += state_error.T @ Q @ state_error
                control = U_i[:, j]
                J_CDA += control.T @ R @ control
            # terminal cost
            state_error_N = X_i[:, -1] - state_target
            J_CDA += state_error_N.T @ P @ state_error_N

            V_1 = J_CDA - l_0
            # dp = d_solver.solve(state_target, inputs_target)
            # C_N = (cfg.PREDICTION_HORIZON - 1)*dp + cfg.l_T
            C_N = 78.6

            Data_sample = {
                'p_t': pt_i,
                'x_1': x_1,
                'l_0': l_0,
                'V_1': V_1,
                'C_N': C_N
            }
            # Acceptance condition: Data is valid if it stays within the defined level set C_N
            if V_1 <= C_N:
                state_init_str = utils.fmt_array(state_init)  # 数组
                u0_str = utils.fmt_array(U_i[:,0])  # 标量
                x1_str = utils.fmt_array(x_1)  # 数组
                l0_str = utils.fmt_scalar(l_0)
                V1_str = utils.fmt_scalar(V_1)
                C_N_str = utils.fmt_scalar(C_N)

                print(f'Successfully collected {i + 1}/{cfg.M} data point:'
                      f'x_0:{state_init_str}, u_0:{u0_str}, x_1:{x1_str}, l_0:{l0_str}, V_1:{V1_str}, C_N:{C_N_str}')
                Dataset.append(Data_sample)
                i = i + 1
        except RuntimeError as e:
            print(f"\033[91m[Data Collection Failed] current_state: {state_init}, ref_state: {state_target}\033[0m")
            i = i + 1

        # Break loop once sufficient data is collected
        if i>=cfg.M:
            return Dataset

if __name__ == '__main__':
    Dataset = data_collect()
    save_path = os.path.join(os.path.dirname(__file__), 'data_set.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(Dataset, f)
    print(f"All data successfully saved to: {save_path}")