"""
Learning Lyapunov terminal costs from data.
Implementation of equations (19)-(26) from:
Abdufattokhov et al. (2024) - Learning Lyapunov terminal costs from data.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import sys
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import config


class LyapunovTerminalCostDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the data collected via Algorithm 1.
    Each sample contains:
        p_t: Parameter vector (x_0, x_r, u_r)
        x_1: Optimal state at the next time step
        l_0: Stage cost at the initial step
        V_1: Value function (cost-to-go) at step 1
        C_N: The level set value representing the domain constraint
    """

    def __init__(self, data_path: str, transform=None):
        """
        Args:
            data_path: Path to the serialized dataset file (.pkl)
            transform: Optional data transformations
        """
        super(LyapunovTerminalCostDataset, self).__init__()

        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.transform = transform
        self._process_data()

    def _process_data(self):
        """ Processes raw dictionary data into PyTorch Tensors """
        self.p_t_list = []  # 参数向量 [x_0, x_r, u_r]
        self.x_1_list = []  # 下一时刻最优状态
        self.l_0_list = []  # 初始阶段代价
        self.V_1_list = []  # 成本到go函数值
        self.C_N_list = []  # C_N值

        for sample in self.raw_data:
            p_t = sample['p_t']
            # Concatenate parameter vector p_t = [x_0, x_r, u_r]
            x_0 = p_t['x_0']
            x_r = p_t['x_r']
            u_r = p_t['u_r']
            p_t_vec = np.concatenate([x_0, x_r, u_r])

            self.p_t_list.append(p_t_vec)
            self.x_1_list.append(sample['x_1'])
            self.l_0_list.append(sample['l_0'])
            self.V_1_list.append(sample['V_1'])
            self.C_N_list.append(sample['C_N'])

        # Convert to Tensors
        self.p_t_tensor = torch.tensor(np.array(self.p_t_list), dtype=torch.float32)
        self.x_1_tensor = torch.tensor(np.array(self.x_1_list), dtype=torch.float32)
        self.l_0_tensor = torch.tensor(np.array(self.l_0_list), dtype=torch.float32).reshape(-1, 1)
        self.V_1_tensor = torch.tensor(np.array(self.V_1_list), dtype=torch.float32).reshape(-1, 1)
        self.C_N_tensor = torch.tensor(np.array(self.C_N_list), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.p_t_list)

    def __getitem__(self, idx):
        return {
            'p_t': self.p_t_tensor[idx],
            'x_1': self.x_1_tensor[idx],
            'l_0': self.l_0_tensor[idx],
            'V_1': self.V_1_tensor[idx],
            'C_N': self.C_N_tensor[idx]
        }


class LowerTriangularMatrixNet(nn.Module):
    """
    Neural Network implementation of \hat{L}_\theta(p_t).
    Outputs the vectorized form of a lower triangular matrix L, which is
    then used to construct a positive definite matrix \hat{P} (Cholesky factorization style).

    Paper Equations (19)-(20):
    \hat{V}^{LLTC}(x, p_t) = (x - x_r)^T \hat{P}(p_t)(x - x_r)
    \hat{P}(p_t) = \hat{L}(p_t)\hat{L}^T(p_t) + \epsilon I
    """
    def __init__(self, input_dim: int, state_dim: int, hidden_layers: List[int],
                 epsilon: float = 0.001, activation=nn.ReLU()):
        """
        Args:
             input_dim: Dimension of p_t (state_dim*2 + inputs_dim = 8)
             state_dim: Dimension of the state (3)
             hidden_layers: List containing number of neurons in hidden layers
             epsilon: Small constant to strictly guarantee positive definiteness
             activation: Activation function applied between layers
        """
        super(LowerTriangularMatrixNet, self).__init__()

        self.state_dim = state_dim
        self.epsilon = epsilon

        # Number of non-zero elements in an n x n lower triangular matrix
        self.triangular_elements = state_dim * (state_dim + 1) // 2

        # Build Multi-Layer Perceptron (MLP
        layer = []
        pre_dim = input_dim

        for hidden_dim in hidden_layers:
            layer.append(nn.Linear(pre_dim, hidden_dim))
            layer.append(activation)
            pre_dim = hidden_dim

        layer.append(nn.Linear(pre_dim, self.triangular_elements))

        self.network = nn.Sequential(*layer)

    def forward(self, p_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute L and P matrices.

        Args:
            p_t: Parameter tensor of shape [batch_size, input_dim]

        Returns:
            L: Lower triangular matrix [batch_size, state_dim, state_dim]
            P: Positive definite weight matrix [batch_size, state_dim, state_dim]
        """
        batch_size = p_t.shape[0]

        # Get lower triangular elements from the network
        tri_elements = self.network(p_t)

        # Reconstruct the lower triangular matrix L
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=p_t.device)
        idx = 0
        for i in range(self.state_dim):
            for j in range(i + 1):
                L[:, i, j] = tri_elements[:, idx]
                idx += 1

        # Construct the positive definite matrix P = L * L^T + epsilon * I
        L_transpose = L.transpose(1, 2)
        P = torch.bmm(L, L_transpose) + self.epsilon * torch.eye(self.state_dim, device=p_t.device).unsqueeze(0)

        return L, P

    def compute_terminal_cost(self, x: torch.Tensor, x_r: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        """
        Computes the terminal cost \hat{V}^{LLTC}(x, p_t) using the quadratic form.

        Args:
            x: State tensor [batch_size, state_dim]
            x_r: Reference state tensor [batch_size, state_dim]
            p_t: Parameter tensor [batch_size, input_dim]

        Returns:
            cost: Calculated terminal cost [batch_size, 1]
        """
        _, P = self.forward(p_t)

        # Compute state error
        state_error = x - x_r  # [batch_size, state_dim]

        # Compute quadratic form: (x - x_r)^T * P * (x - x_r)
        P_error = torch.bmm(P, state_error.unsqueeze(-1))
        cost = torch.bmm(state_error.unsqueeze(1), P_error)

        return cost.squeeze(-1)


class LyapunovLoss(nn.Module):
    """
    Loss function for learning the Lyapunov terminal cost.

    Paper Equation (26):
    L(θ) = γ||θ||² + 1/M ∑[φ(V_1^i, \hat{V}^{LLTC}(x_1^i,p_t^i))
           + λ₁E_dom^i(θ) + λ₂E_dec^i(θ)]

    Where constraints are formulated as ReLU penalties:
    - Domain Constraint E_dom: max{\hat{V}(x_1) - C_N, 0}
    - Decay Constraint E_dec: max{\hat{V}(x_1) - \hat{V}(x_0) + l_0, 0}
    """

    def __init__(self, lambda_1: float = 1e4, lambda_2: float = 1e4,
                 gamma: float = 1e-6, loss_type: str = 'mse'):
        """
        Args:
            lambda_1: Penalty coefficient for the Domain constraint
            lambda_2: Penalty coefficient for the Decay condition constraint
            gamma: L2 Regularization coefficient
            loss_type: Type of fitting loss ('mse', 'mae', 'huber')
        """
        super(LyapunovLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gamma = gamma

        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='sum')
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='sum')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='sum', delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, model: LowerTriangularMatrixNet,
                    p_t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor,
                    x_r: torch.Tensor, l_0: torch.Tensor, V_1_true: torch.Tensor,
                    C_N: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        """
        batch_size = p_t.shape[0]
        # Compute predicted terminal costs for x_1 and x_0
        V_1_hat = model.compute_terminal_cost(x_1, x_r, p_t)  # \hat{V}(x_1, p_t)
        V_0_hat = model.compute_terminal_cost(x_0, x_r, p_t)  # \hat{V}(x_0, p_t)

        # 1. Fitting Loss (Data fidelity)
        fit_loss = self.base_loss(V_1_hat, V_1_true)/batch_size

        # 2. Domain Violation Penalty (Ensure V_hat is bounded by C_N)
        dom_violation = torch.clamp(V_1_hat - C_N, min=0)
        dom_penalty = dom_violation.sum() / batch_size

        # 3. Decay Condition Penalty (Ensure V_hat strictly decreases: V_1 - V_0 <= -l_0)
        dec_violation = torch.clamp(V_1_hat - V_0_hat + l_0, min=0)
        dec_penalty = dec_violation.sum() / batch_size

        # 4. L2 Regularization
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)
        l2_reg = self.gamma * l2_reg

        # total_loss
        total_loss = fit_loss + self.lambda_1 * dom_penalty + self.lambda_2 * dec_penalty + l2_reg
        # total_loss = fit_loss + l2_reg

        return {
            'total': total_loss,
            'fit': fit_loss,
            'dom_penalty': dom_penalty,
            'dec_penalty': dec_penalty,
            'l2_reg': l2_reg,
            'dom_violation_count': (dom_violation > 0).sum().item(),
            'dec_violation_count': (dec_violation > 0).sum().item()
        }


class LyapunovTerminalCostTrainer:
    """
    trainer
    """

    def __init__(self, model: LowerTriangularMatrixNet,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 cfg: config.Config):
        """
        Args:
            model: nn forward model
            train_loader: train data loader
            val_loader: valuation data loader
            cfg: config parameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        # loss function
        self.criterion = LyapunovLoss(
            lambda_1=1e4,
            lambda_2=1e4,
            gamma=1e-6,
            loss_type='mse'
        )

        # optimizer (Adam)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.995, 0.999)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.999, patience=50
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_fit_loss': [],
            'val_fit_loss': [],
            'train_dom_penalty': [],
            'val_dom_penalty': [],
            'train_dec_penalty': [],
            'val_dec_penalty': [],
            'dom_violations': [],
            'dec_violations': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """train an epoch"""
        self.model.train()

        total_loss = 0
        total_fit = 0
        total_dom = 0
        total_dec = 0
        total_dom_count = 0
        total_dec_count = 0
        num_batches = 0

        device = next(self.model.parameters()).device

        for batch in self.train_loader:
            p_t = batch['p_t'].to(device)
            x_1 = batch['x_1'].to(device)
            l_0 = batch['l_0'].to(device)
            V_1 = batch['V_1'].to(device)
            C_N = batch['C_N'].to(device)

            # p_t = [x_0, x_r, u_r]
            state_dim = self.cfg.state_dim
            x_0 = p_t[:, :state_dim]
            x_r = p_t[:, state_dim:2 * state_dim]

            self.optimizer.zero_grad()

            loss_dict = self.criterion(
                self.model, p_t, x_0, x_1, x_r, l_0, V_1, C_N
            )

            loss_dict['total'].backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss_dict['total'].item()
            total_fit += loss_dict['fit'].item()
            total_dom += loss_dict['dom_penalty'].item()
            total_dec += loss_dict['dec_penalty'].item()
            total_dom_count += loss_dict['dom_violation_count']
            total_dec_count += loss_dict['dec_violation_count']
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'fit': total_fit / num_batches,
            'dom': total_dom / num_batches,
            'dec': total_dec / num_batches,
            'dom_count': total_dom_count / len(self.train_loader.dataset),
            'dec_count': total_dec_count / len(self.train_loader.dataset)
        }

    def validate(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0
        total_fit = 0
        total_dom = 0
        total_dec = 0
        total_dom_count = 0
        total_dec_count = 0
        num_batches = 0

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in self.val_loader:
                p_t = batch['p_t'].to(device)
                x_1 = batch['x_1'].to(device)
                l_0 = batch['l_0'].to(device)
                V_1 = batch['V_1'].to(device)
                C_N = batch['C_N'].to(device)

                state_dim = self.cfg.state_dim
                x_0 = p_t[:, :state_dim]
                x_r = p_t[:, state_dim:2 * state_dim]

                loss_dict = self.criterion(
                    self.model, p_t, x_0, x_1, x_r, l_0, V_1, C_N
                )

                total_loss += loss_dict['total'].item()
                total_fit += loss_dict['fit'].item()
                total_dom += loss_dict['dom_penalty'].item()
                total_dec += loss_dict['dec_penalty'].item()
                total_dom_count += loss_dict['dom_violation_count']
                total_dec_count += loss_dict['dec_violation_count']
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'fit': total_fit / num_batches,
            'dom': total_dom / num_batches,
            'dec': total_dec / num_batches,
            'dom_count': total_dom_count / len(self.val_loader.dataset),
            'dec_count': total_dec_count / len(self.val_loader.dataset)
        }

    def train(self, num_epochs: int = 5000, save_path: str = None):
        """
        train the network

        Args:
            num_epochs: train epochs
            save_path: save path
        """
        print("start training...")
        print(f"number of model parameters: {sum(p.numel() for p in self.model.parameters())}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # train
            train_metrics = self.train_epoch()

            # validating
            val_metrics = self.validate()

            # update the studying rate
            self.scheduler.step(val_metrics['loss'])

            # save training history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_fit_loss'].append(train_metrics['fit'])
            self.history['val_fit_loss'].append(val_metrics['fit'])
            self.history['train_dom_penalty'].append(train_metrics['dom'])
            self.history['val_dom_penalty'].append(val_metrics['dom'])
            self.history['train_dec_penalty'].append(train_metrics['dec'])
            self.history['val_dec_penalty'].append(val_metrics['dec'])
            self.history['dom_violations'].append(val_metrics['dom_count'])
            self.history['dec_violations'].append(val_metrics['dec_count'])

            # save the best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'history': self.history
                    }, save_path.replace('.pth', '_best.pth'))

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}]")
                print(f"  Train Loss: {train_metrics['loss']:.6f}")
                print(f"  Val Loss: {val_metrics['loss']:.6f}")
                print(f"  Fit Loss: {val_metrics['fit']:.6f}")
                print(f"  Domain Violations: {val_metrics['dom_count']:.2%}")
                print(f"  Decay Violations: {val_metrics['dec_count']:.2%}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 50)

        # save model
        if save_path:
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'history': self.history
            }, save_path)
            print(f"model has saved in{save_path}")

        print("training finished!")

    def plot_training_history(self, save_path: str = None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # total loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # fit loss
        axes[0, 1].plot(epochs, self.history['train_fit_loss'], label='Train')
        axes[0, 1].plot(epochs, self.history['val_fit_loss'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Fit Loss')
        axes[0, 1].set_title('Fitting Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Violation Rate
        axes[0, 2].plot(epochs, self.history['dom_violations'], label='Domain')
        axes[0, 2].plot(epochs, self.history['dec_violations'], label='Decay')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Violation Rate')
        axes[0, 2].set_title('Constraint Violations')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Domain Penalty
        axes[1, 0].plot(epochs, self.history['train_dom_penalty'], label='Train')
        axes[1, 0].plot(epochs, self.history['val_dom_penalty'], label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Domain Penalty')
        axes[1, 0].set_title('Domain Constraint Penalty')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Decay Penalty
        axes[1, 1].plot(epochs, self.history['train_dec_penalty'], label='Train')
        axes[1, 1].plot(epochs, self.history['val_dec_penalty'], label='Validation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Decay Penalty')
        axes[1, 1].set_title('Decay Constraint Penalty')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # studying rate
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def evaluate_model(model: LowerTriangularMatrixNet,
                   test_loader: DataLoader,
                   cfg: config.Config) -> Dict[str, float]:

    model.eval()
    device = next(model.parameters()).device

    all_V_true = []
    all_V_pred = []
    dom_violations = 0
    dec_violations = 0

    with torch.no_grad():
        for batch in test_loader:
            p_t = batch['p_t'].to(device)
            x_1 = batch['x_1'].to(device)
            l_0 = batch['l_0'].to(device)
            V_true = batch['V_1'].to(device)
            C_N = batch['C_N'].to(device)

            state_dim = cfg.state_dim
            x_0 = p_t[:, :state_dim]
            x_r = p_t[:, state_dim:2 * state_dim]

            V_pred = model.compute_terminal_cost(x_1, x_r, p_t)
            V_0_pred = model.compute_terminal_cost(x_0, x_r, p_t)

            # 检查约束违反
            dom_violations += (V_pred > C_N).sum().item()
            dec_violations += (V_pred - V_0_pred + l_0 > 0).sum().item()

            all_V_true.append(V_true.cpu().numpy())
            all_V_pred.append(V_pred.cpu().numpy())

    V_true = np.concatenate(all_V_true).flatten()
    V_pred = np.concatenate(all_V_pred).flatten()

    rmse = np.sqrt(np.mean((V_true - V_pred) ** 2))
    nrmse = rmse / (V_true.max() - V_true.min())

    ss_res = np.sum((V_true - V_pred) ** 2)
    ss_tot = np.sum((V_true - np.mean(V_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    total_samples = len(test_loader.dataset)

    return {
        'NRMSE': nrmse,
        'R2_score': r2,
        'C_dom': dom_violations,
        'C_dec': dec_violations,
        'dom_violation_rate': dom_violations / total_samples,
        'dec_violation_rate': dec_violations / total_samples
    }


def main():
    """
    Main function to execute the full pipeline:
    1. Load and split the dataset.
    2. Initialize the neural network model.
    3. Train the model to learn the Lyapunov terminal cost.
    4. Evaluate and visualize the training performance.
    """
    cfg = config.Config()

    # Device Setup
    # Automatically select GPU (CUDA) if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset Loading & Preparation
    data_path = os.path.join(os.path.dirname(__file__), 'data_set.pkl')
    dataset = LyapunovTerminalCostDataset(data_path)

    # Split the dataset into training and testing sets (80% / 20% split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create PyTorch DataLoaders to handle batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Calculate input dimension: p_t = [x_0, x_r, u_r]
    input_dim = cfg.state_dim * 2 + cfg.inputs_dim  # x_0 + x_r + u_r

    # Instantiate the Neural Network model
    # Note: Using the optimal architecture from Table 1 of the reference paper
    # (3 hidden layers with 40 neurons each)
    model = LowerTriangularMatrixNet(
        input_dim=input_dim,
        state_dim=cfg.state_dim,
        hidden_layers=[40, 40, 40],  # 3层隐藏层
        epsilon=0.001,
        activation=nn.ReLU()
    ).to(device)

    # Training Initialization
    trainer = LyapunovTerminalCostTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        cfg=cfg
    )

    # Define directories and paths for saving model checkpoints
    save_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'lyapunov_terminal_cost.pth')

    trainer.train(num_epochs=8000, save_path=model_save_path)

    # Plot and save the training history metrics (Losses, Penalties, Violations)
    plot_path = os.path.join(save_dir, 'training_history.png')
    trainer.plot_training_history(save_path=plot_path)

    # Print final evaluation metrics
    metrics = evaluate_model(model, test_loader, cfg)

    print("\n" + "=" * 50)
    print("Model Evaluation Results:")
    print(f"  NRMSE (train/test): {metrics['NRMSE']:.4f}")
    print(f"  R² score (train/test): {metrics['R2_score']:.4f}")
    print(f"  C_dom (train/test): {metrics['C_dom']}")
    print(f"  C_dec (train/test): {metrics['C_dec']}")
    print(f"  Domain violation rate: {metrics['dom_violation_rate']:.2%}")
    print(f"  Decay violation rate: {metrics['dec_violation_rate']:.2%}")
    print("=" * 50)


if __name__ == '__main__':
    main()
