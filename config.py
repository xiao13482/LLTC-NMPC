"""
Global Configuration Parameters
"""
import numpy as np

class Config:
    def __init__(self):
        # --- System Settings ---
        self.DT_SIM = 0.1  # Simulation time step (seconds)
        self.PREDICTION_HORIZON = 60  # MPC prediction horizon steps

        # --- MPC Parameters ---
        # State vector: [x, y, phi] (Position and Heading)
        # Input vector: [v, delta] (Velocity and Steering Angle)
        self.state_dim = 3
        self.inputs_dim = 2
        self.Q = [3.0, 5.0, 0.01]   # State stage cost weight
        self.P = [3.0, 5.0, 0.01]   # Terminal cost weight (used if not using Neural Cost)
        self.R = [0.5, 0.01]        # Control input weight
        self.R_2 = [0.0, 0.0]       # Control rate weight (to smooth control)
        self.l_T = 1.31             # Threshold value for terminal constraint

        # --- State Constraints ---
        self.x_max = 2.0; self.x_min = -2.0
        self.y_max = 2.0; self.y_min = -2.0
        self.phi_max = np.pi; self.phi_min = -np.pi

        # --- Control Constraints ---
        self.v_max = 0.6            # Max forward velocity
        self.v_min = -0.2           # Max reverse velocity
        self.delta_max = np.pi/3    # Max steering angle
        self.delta_min = -np.pi/3

        # --- Obstacle Avoidance (Cylinder) ---
        self.cylinder_cx = 0        # Center x
        self.cylinder_cy = 1        # Center y
        self.cylinder_r = 0         # Radius (set to 0 for no obstacle)

        # --- Dataset Parameters ---
        self.M = 5000               # Number of data points to collect
        self.seed = 42              # Random seed for reproducibility