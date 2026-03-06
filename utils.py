import numpy as np
import casadi as ca
import sys
from config import Config


def fmt_scalar(val):
    """格式化标量为保留三位小数的字符串"""
    return f"{val:.3f}" if val is not None else "None"

def fmt_array(arr):
    """格式化 NumPy 数组，每个元素保留三位小数"""
    return np.array2string(arr, precision=3, separator=', ', suppress_small=True)

class dp_solver:
    """
    求解论文式（13）对应的非线性规划问题
    """
    def __init__(self, cfg = Config()):
        self.cfg = cfg

        #MPC参数
        self.Q = np.diag(cfg.Q)
        self.R = np.diag(cfg.R)
        self.P = np.diag(cfg.P)
        self.l_T = cfg.l_T

        self.opti = ca.Opti()

        #优化变量
        self.x = self.opti.variable(self.cfg.state_dim, 1)

        #外部参数
        self.x_r = self.opti.parameter(self.cfg.state_dim)   #目标状态

        self._build_optimization_problem()

        # 求解器配置
        opts = {
            'ipopt.print_level': 0,  # 0: 静默, 5: 详细
            'ipopt.sb': 'yes',  # 抑制横幅
            'print_time': 0 ,  # 打印时间
            'ipopt.max_iter': 2000,  # 最大迭代次数
            'ipopt.tol': 1e-4,  # 求解容差
            'expand': True  # 展开数学运算，加速求解
        }  # 详细配置参数
        self.opti.solver('ipopt', opts)  # 内点法求解NLP问题


    def _build_optimization_problem(self):

        #优化变量
        state_error = self.x[:, 0] - self.x_r[:,0]

        # l = ca.mtimes([state_error.T, self.Q, state_error]) + ca.mtimes([control.T, self.R, control])
        l = ca.mtimes([state_error.T, self.Q, state_error])
        self.opti.minimize(l)

        #状态约束
        self.opti.subject_to(self.opti.bounded(self.cfg.x_min, self.x[0, 0], self.cfg.x_max))
        self.opti.subject_to(self.opti.bounded(self.cfg.y_min, self.x[1, 0], self.cfg.y_max))
        self.opti.subject_to(self.opti.bounded(self.cfg.phi_min, self.x[2, 0], self.cfg.phi_max))



        #状态空间内终端约束的补集
        V_N = ca.mtimes([state_error.T, self.P, state_error])
        self.opti.subject_to(V_N == self.cfg.l_T)

    def solve(self, x_r, u_r):
        #设置外部参数
        self.opti.set_value(self.x_r, x_r)

        #求解
        try:
            sol = self.opti.solve()
            u_opti = sol.value(self.u)
            x_opti = sol.value(self.x)
            state_error = x_opti - x_r
            control = u_opti - u_r
            d = state_error.T @ self.Q @ state_error + control.T @ self.R @ control
            return d
        except RuntimeError as e:
            sys.exit(f"[Warning] d Solver Failed: {e}")



