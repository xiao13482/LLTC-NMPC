
# LLTC-NMPC: Learning Lyapunov Terminal Costs from Data

This repository is an unofficial Python/PyTorch reimplementation of the algorithm from the paper [**Learning Lyapunov terminal costs from data for complexity reduction in nonlinear model predictive control**](https://doi.org/10.1002/rnc.7411) (Abdufattokhov et al., 2024, *Int J Robust Nonlinear Control*), primarily reproducing the **Autonomous parking problem** from the paper.

## Overview
By constructing a neural network-based Lyapunov terminal cost (LTC), the prediction horizon of Nonlinear Model Predictive Control (NMPC) is reduced from $N=60$ to $N=1$. This significantly decreases the computational burden of online optimization while strictly guaranteeing the asymptotic stability of the closed-loop system. This project reproduces the **LLTC-NMPC** framework from the paper:

- **Data Generation**: Collects expert data (states $x_0$, $x_1$, stage cost $l_0$, remaining cost $V_1$) by solving long-horizon NMPC offline.
- **Constrained Learning**: Designs a neural network structure satisfying lower triangular decomposition $\hat{P}(p_t) = \hat{L}(p_t)\hat{L}^T(p_t) + \epsilon I$, and introduces **Domain Constraints** and **Decay Constraints** in the training loss to ensure the network output strictly adheres to the Lyapunov stability theorem.
- **Extremely Fast Inference**: During online control, the trained neural network performs forward inference to obtain the terminal weight matrix $P$. Combined with an ultra-short prediction horizon of $N=1$, it achieves millisecond-level real-time control based on CasADi + IPOPT.
- ---

## Installation Environment

It is recommended to create a clean virtual environment using Conda by running:

```bash
conda create -n lltc_nmpc python=3.9
conda activate lltc_nmpc
pip install torch numpy matplotlib
pip install casadi
```
## Project Structure
- `MPC/dynamic.py`: Defines the dynamic model required for MPC solving and simulation.
- `MPC/NMPC_solver.py`: The NMPC solver, performing single-step MPC solutions using CasADi + IPOPT.
- `config.py`: Configures global parameters.
- `data_collect.py`: Implements *Algorithm 1* from the paper to collect training data.
- `Lyapunov_train.py`: Trains the neural network.
- `MPC_test.py`: Implements closed-loop control for testing and visualization.
## How to run
### 1. Collect Training Data
Run the script `data_collect.py`. The dataset will be saved as `data_set.pkl` in the current root directory.
### 2. Lyapunov Terminal Cost Training
Run the script `Lyapunov_train.py`. Key information about the training progress will be printed to the console.
### 3. LLTC-MPC Closed-Loop Control
Run the script `MPC_test.py`. A pop-up animation window will display the real-time trajectory of the car.
