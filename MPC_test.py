"""
MPC Testing and Simulation Script

Runs the closed-loop simulation of the vehicle reaching the target point using NMPC,
and visualizes the map, state trajectories, and control inputs dynamically using matplotlib.
"""
import numpy as np
import matplotlib
import torch
import torch.nn as nn
import os

from MPC.dynamic import Sim_Dynamic

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from MPC.NMPC_solver import MPC_solver
from Lyapunov_train import LowerTriangularMatrixNet
import config



def main():
    # --- 1. Initialization ---
    cfg = config.Config()
    dynamic = Sim_Dynamic()
    solver = MPC_solver(print_time=True, neural_cost = True)# Enable Neural Terminal Cost
    
    # Load neural terminal cost model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = cfg.state_dim * 2 + cfg.inputs_dim  # p_t = [x_0, x_r, u_r]
    neural_cost_model = LowerTriangularMatrixNet(
        input_dim=input_dim,
        state_dim=cfg.state_dim,
        hidden_layers=[40, 40, 40],
        epsilon=0.001,
        activation=nn.ReLU()
    ).to(device)
    
    # Load pre-trained weights
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'lyapunov_terminal_cost_best.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        neural_cost_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Neural terminal cost model loaded from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
    neural_cost_model.eval()

    current_state = np.array([0.41, 1.25, -1.61])# Initial parking position
    ref_state = np.array([0, 0, 0])                # Target parking position

    # Data logging arrays for plotting
    history_x, history_y, history_phi = [current_state[0]], [current_state[1]], [current_state[2]]
    history_v, history_delta = [], []
    history_neural_cost = []
    time_steps = []

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 2)
    ax_map = fig.add_subplot(gs[:, 0])  
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_phi = fig.add_subplot(gs[1, 1])
    ax_ctrl = fig.add_subplot(gs[2, 1])
    ax_neural = fig.add_subplot(gs[3, 1])

    ax_map.set_xlim(cfg.x_min - 0.5, cfg.x_max + 0.5)
    ax_map.set_ylim(cfg.y_min - 0.5, cfg.y_max + 0.5)
    ax_map.set_aspect('equal')
    ax_map.grid(True, linestyle=':', alpha=0.6)
    ax_map.add_patch(patches.Circle((cfg.cylinder_cx, cfg.cylinder_cy), cfg.cylinder_r, color='red', alpha=0.3))
    ax_map.plot(ref_state[0], ref_state[1], 'gx', markersize=10, label='Goal')
    line_run, = ax_map.plot([], [], 'b-', linewidth=1.5, label='Real Path')
    line_predict, = ax_map.plot([], [], 'r--', alpha=0.5, label='MPC Predict')
    car_arrow = ax_map.quiver(current_state[0], current_state[1], np.cos(current_state[2]), np.sin(current_state[2]),
                              color='darkblue', scale=20, width=0.005, zorder=5)

    line_x, = ax_pos.plot([], [], 'r-', label='x')
    line_y, = ax_pos.plot([], [], 'b-', label='y')
    ax_pos.set_ylabel("Position")
    ax_pos.legend(loc='right')
    ax_pos.grid(True, alpha=0.3)

    line_phi, = ax_phi.plot([], [], 'k-', label='phi (rad)')
    ax_phi.set_ylabel("Heading")
    ax_phi.set_ylim(cfg.phi_min - 0.5, cfg.phi_max + 0.5)
    ax_phi.legend(loc='right')
    ax_phi.grid(True, alpha=0.3)

    line_v, = ax_ctrl.plot([], [], 'g-', label='v (m/s)')
    line_delta, = ax_ctrl.plot([], [], 'm-', label='delta (rad)')
    ax_ctrl.set_ylabel("Control")
    ax_ctrl.set_xlabel("Steps")
    ax_ctrl.legend(loc='right')
    ax_ctrl.grid(True, alpha=0.3)

    line_neural_cost, = ax_neural.plot([], [], 'c-', label='Neural Terminal Cost')
    ax_neural.set_ylabel("Neural Terminal Cost")
    ax_neural.set_xlabel("Steps")
    ax_neural.legend(loc='right')
    ax_neural.grid(True, alpha=0.3)

    def update(frame):
        nonlocal current_state

        # MPC solve
        u_opt, path_predict = solver.solve(current_state, ref_state)
        v, delta = u_opt[0], u_opt[1]

        history_v.append(v)
        history_delta.append(delta)
        time_steps.append(frame)

        # dynamic update
        new_x = current_state[0] + np.cos(current_state[2]) * v * cfg.DT_SIM
        new_y = current_state[1] + np.sin(current_state[2]) * v * cfg.DT_SIM
        new_phi = current_state[2] + delta * cfg.DT_SIM
        current_state = dynamic.rk4_step(current_state, u_opt, cfg.DT_SIM)

        history_x.append(current_state[0])
        history_y.append(current_state[1])
        history_phi.append(current_state[2])
        
        # Compute neural terminal cost
        with torch.no_grad():
            # Construct p_t = [x_0_initial, x_r, u_r] for neural cost model
            # For simplicity, using current_state and ref_state
            x_state = torch.tensor([current_state], dtype=torch.float32).to(device)
            x_ref = torch.tensor([ref_state], dtype=torch.float32).to(device)
            
            # p_t = [x_0, x_r, u_r] - we use current state as x_0
            u_r = np.array([0, 0])  # Reference input (both zero)
            p_t = torch.tensor([np.concatenate([current_state, ref_state, u_r])], dtype=torch.float32).to(device)
            
            neural_cost = neural_cost_model.compute_terminal_cost(x_state, x_ref, p_t)
            history_neural_cost.append(neural_cost.item())

        # update plot
        line_run.set_data(history_x, history_y)
        line_predict.set_data(path_predict[0, :], path_predict[1, :])
        car_arrow.set_offsets([current_state[0], current_state[1]])
        car_arrow.set_UVC(np.cos(current_state[2]), np.sin(current_state[2]))

        line_x.set_data(time_steps, history_x[:-1])
        line_y.set_data(time_steps, history_y[:-1])
        line_phi.set_data(time_steps, history_phi[:-1])
        
        # 为v和delta生成阶梯状数据（零阶保持器）
        if len(history_v) > 0 and len(time_steps) > 0:
            v_step_times = []
            v_step_values = []
            for i in range(len(history_v)):
                v_step_times.append(time_steps[i])
                v_step_values.append(history_v[i])
                if i < len(history_v) - 1:
                    v_step_times.append(time_steps[i + 1])
                    v_step_values.append(history_v[i])
            line_v.set_data(v_step_times, v_step_values)

        if len(history_delta) > 0 and len(time_steps) > 0:
            delta_step_times = []
            delta_step_values = []
            for i in range(len(history_delta)):
                delta_step_times.append(time_steps[i])
                delta_step_values.append(history_delta[i])
                if i < len(history_delta) - 1:
                    delta_step_times.append(time_steps[i + 1])
                    delta_step_values.append(history_delta[i])
            line_delta.set_data(delta_step_times, delta_step_values)

        # Update neural cost curve
        if len(history_neural_cost) > 0:
            line_neural_cost.set_data(time_steps, history_neural_cost)

        for ax in [ax_pos, ax_phi, ax_ctrl, ax_neural]:
            ax.set_xlim(0, max(20, frame + 5))

        ax_pos.relim()
        ax_pos.autoscale_view()

        ax_phi.relim()
        ax_phi.autoscale_view()

        ax_ctrl.relim()
        ax_ctrl.autoscale_view()
        
        ax_neural.relim()
        ax_neural.autoscale_view()

        return line_run, line_predict, car_arrow, line_x, line_y, line_phi, line_v, line_delta, line_neural_cost

    sim_steps = 200
    ani = FuncAnimation(fig, update, frames=sim_steps, interval=100, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()