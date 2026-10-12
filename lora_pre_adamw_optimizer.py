import torch
from torch.optim.optimizer import Optimizer

class LoRAPreAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.0, rank=16):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rank=rank)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            rank = group['rank']

            # 论文中的系数耦合
            gamma1 = 1.0 - (beta1 ** 0.5)
            gamma2 = 1.0 - (beta2 ** 0.25)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to(torch.float32)  # <--- 加上 .to(torch.float32)
                if grad.is_sparse:
                    raise RuntimeError('LoRAPreAdamW does not support sparse gradients')

                state = self.state[p]

                # --- 状态初始化 ---
                if len(state) == 0:
                    # 根据权重矩阵的形状决定低秩矩阵的形状
                    if len(p.shape) == 2:
                        p_dim, q_dim = p.shape
                        r = min(rank, p_dim, q_dim)
                    else:
                        # 1D参数（如 bias）保留原维度，或者降级为普通 Adam
                        r = 1
                        p_dim, q_dim = 1, p.numel()
                    
                    state['step'] = 0
                    # 一阶动量低秩因子 (不要全零，给微小随机值，避免“冷启动”)
                    state['m_B'] = torch.randn((p_dim, r), device=p.device, dtype=torch.float32) * 0.02
                    state['m_A'] = torch.randn((r, q_dim), device=p.device, dtype=torch.float32) * 0.02
                    # 二阶动量低秩因子（使用平方形式保证非负，同样给随机初值）
                    state['v_B'] = torch.randn((p_dim, r), device=p.device, dtype=torch.float32).abs() * 0.02
                    state['v_A'] = torch.randn((r, q_dim), device=p.device, dtype=torch.float32).abs() * 0.02

                state['step'] += 1
                step = state['step']

                # 【新增】如果是第一步，用梯度做 SVD 初始化，彻底解决冷启动
                if step == 1 and len(p.shape) == 2:
                    U, S, V = torch.svd_lowrank(grad, q=r)
                    state['m_B'] = (U @ torch.diag(S)).to(torch.float32)
                    state['m_A'] = V.t().to(torch.float32)
                    
                    abs_grad = grad.abs()
                    U_v, S_v, V_v = torch.svd_lowrank(abs_grad, q=r)
                    state['v_B'] = (U_v @ torch.diag(S_v)).to(torch.float32)
                    state['v_A'] = V_v.t().to(torch.float32)

                # 如果是1D参数，直接使用标准Adam
                if len(p.shape) != 2:
                    m = state.get('m', torch.zeros_like(p))
                    v = state.get('v', torch.zeros_like(p))
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad * grad
                    state['m'], state['v'] = m, v
                    m_hat = m / (1 - beta1 ** step)
                    v_hat = v / (1 - beta2 ** step)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                    if weight_decay != 0:
                        p.add_(p, alpha=-lr * weight_decay)
                    continue

                # --- 2D 参数的低秩更新 ---
                m_B, m_A = state['m_B'], state['m_A']
                v_B, v_A = state['v_B'], state['v_A']
                r = m_A.shape[0]

                # 计算矩阵求逆所需的单位矩阵
                I_r = torch.eye(r, device=p.device, dtype=torch.float32)
                
                # 一阶动量更新 (论文公式 11)
                try:
                    # 使用正则化求逆（对应论文防止矩阵奇异）
                    temp_B = grad @ m_A.t() @ torch.linalg.inv(m_A @ m_A.t() + eps * I_r)
                    temp_A = torch.linalg.inv(m_B.t() @ m_B + eps * I_r) @ m_B.t() @ grad
                except RuntimeError:
                    continue  # 如果矩阵奇异则跳过该步

                m_B = (1 - gamma1) * m_B + gamma1 * temp_B
                m_A = (1 - gamma1) * m_A + gamma1 * temp_A
                
                # 二阶动量更新 (论文公式 17)
                abs_grad = grad.abs()
                try:
                    temp_vB = abs_grad @ v_A.t() @ torch.linalg.inv(v_A @ v_A.t() + eps * I_r)
                    temp_vA = torch.linalg.inv(v_B.t() @ v_B + eps * I_r) @ v_B.t() @ abs_grad
                except RuntimeError:
                    continue

                v_B = (1 - gamma2) * v_B + gamma2 * temp_vB
                v_A = (1 - gamma2) * v_A + gamma2 * temp_vA

                # 保存状态
                state['m_B'], state['m_A'] = m_B, m_A
                state['v_B'], state['v_A'] = v_B, v_A

                # 重组有效动量，用于参数更新
                m_hat = (m_B @ m_A) / (1 - beta1 ** step)
                v_hat = (v_B @ v_A).pow(2) / (1 - beta2 ** step) # 论文使用了平方

                # 更新参数
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)
                
                # 权重衰减（AdamW解耦）
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss