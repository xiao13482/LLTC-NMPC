"""
MPC Testing and Simulation Script

Runs the closed-loop simulation of the vehicle reaching the target point using NMPC,
and visualizes the map, state trajectories, and control inputs dynamically using matplotlib.
"""
import numpy as np
import matplotlib

from MPC.dynamic import Sim_Dynamic

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from MPC.NMPC_solver import MPC_solver
import config



def main():
    # --- 1. Initialization ---
    cfg = config.Config()
    dynamic = Sim_Dynamic()
    solver = MPC_solver(print_time=True, neural_cost = True)# Enable Neural Terminal Cost

    current_state = np.array([0.41, 1.25, -1.61])# Initial parking position
    ref_state = np.array([0, 0, 0])                # Target parking position

    # Data logging arrays for plotting
    history_x, history_y, history_phi = [current_state[0]], [current_state[1]], [current_state[2]]
    history_v, history_delta = [], []
    time_steps = []

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 2)
    ax_map = fig.add_subplot(gs[:, 0])  
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_phi = fig.add_subplot(gs[1, 1])
    ax_ctrl = fig.add_subplot(gs[2, 1])

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

        # update plot
        line_run.set_data(history_x, history_y)
        line_predict.set_data(path_predict[0, :], path_predict[1, :])
        car_arrow.set_offsets([current_state[0], current_state[1]])
        car_arrow.set_UVC(np.cos(current_state[2]), np.sin(current_state[2]))

        line_x.set_data(time_steps, history_x[:-1])
        line_y.set_data(time_steps, history_y[:-1])
        line_phi.set_data(time_steps, history_phi[:-1])

          

        for ax in [ax_pos, ax_phi, ax_ctrl]:
            ax.set_xlim(0, max(20, frame + 5))

        ax_pos.relim()
        ax_pos.autoscale_view()

        ax_phi.relim()
        ax_phi.autoscale_view()

        ax_ctrl.relim()
        ax_ctrl.autoscale_view()

        return line_run, line_predict, car_arrow, line_x, line_y, line_phi, line_v, line_delta

    sim_steps = 200
    ani = FuncAnimation(fig, update, frames=sim_steps, interval=100, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()