import torch
from torch.optim.optimizer import Optimizer

# 正交化算法 
def newton_schulz_5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)

# 标准 Muon 
class Muon(Optimizer):
    def __init__(self, params, lr=2e-2, momentum=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mu, lr, wd = group['momentum'], group['lr'], group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                state['m'].mul_(mu).add_(grad, alpha=1-mu)
                m_orth = newton_schulz_5(state['m'])
                adj = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(m_orth, alpha=-lr * adj)
                if wd != 0: p.add_(p, alpha=-lr * wd)
        return loss

class HybridMuon(Optimizer):
    def __init__(self, model, lr=2e-2, momentum=0.95, weight_decay=0.0):
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim == 2 and 'embed' not in name.lower() and 'lm_head' not in name.lower():
                muon_params.append(p)
            else:
                adamw_params.append(p)
        self.muon_opt = Muon(muon_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.adamw_opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.muon_opt.param_groups + self.adamw_opt.param_groups
        self.state = {}
        super().__init__(self.param_groups, {})
    def zero_grad(self, set_to_none=True):
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adamw_opt.zero_grad(set_to_none=set_to_none)
    def step(self, closure=None):
        loss = self.muon_opt.step(closure)
        self.adamw_opt.step()
        return loss

# LoRA-Pre-Muon
class LoRAPreMuon(Optimizer):
    def __init__(self, params, lr=2e-2, momentum=0.95, weight_decay=0.0, rank=16):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rank=rank)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mu, lr, wd, rank = group['momentum'], group['lr'], group['weight_decay'], group['rank']
            gamma1 = 1.0 - (mu ** 0.5)
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.to(torch.float32)
                state = self.state[p]
                if len(state) == 0:
                    p_dim, q_dim = p.shape
                    r = min(rank, p_dim, q_dim)
                    state['step'] = 0
                    state['m_B'] = torch.zeros((p_dim, r), device=p.device, dtype=torch.float32)
                    state['m_A'] = torch.randn((r, q_dim), device=p.device, dtype=torch.float32) * 0.02
                state['step'] += 1
                m_B, m_A = state['m_B'], state['m_A']
                r = m_A.shape[0]
                I_r = torch.eye(r, device=p.device, dtype=torch.float32)
                if state['step'] == 1:
                    U, S, V = torch.svd_lowrank(grad, q=r)
                    state['m_B'], state['m_A'] = (U @ torch.diag(S)).to(torch.float32), V.t().to(torch.float32)
                    m_B, m_A = state['m_B'], state['m_A']
                try:
                    # 提取正则化矩阵并计算特征值
                    M_A = m_A @ m_A.t() + 1e-6 * I_r
                    M_B = m_B.t() @ m_B + 1e-6 * I_r
                    
                    # 每 50 步打印一次特征值
                    if state['step'] % 50 == 0:
                        eig_A = torch.linalg.eigvalsh(M_A)
                        eig_B = torch.linalg.eigvalsh(M_B)
                        print(f"[Muon] Step {state['step']} | M_A max_eig: {eig_A.max().item():.4f}, min_eig: {eig_A.min().item():.6f} | M_B max_eig: {eig_B.max().item():.4f}, min_eig: {eig_B.min().item():.6f}")

                    temp_B = grad @ m_A.t() @ torch.linalg.inv(M_A)
                    temp_A = torch.linalg.inv(M_B) @ m_B.t() @ grad
                except RuntimeError: continue
                m_B = (1 - gamma1) * m_B + (gamma1 / (1 - mu)) * temp_B
                m_A = (1 - gamma1) * m_A + (gamma1 / (1 - mu)) * temp_A
                state['m_B'], state['m_A'] = m_B, m_A
                m_full = m_B @ m_A
                O = newton_schulz_5(m_full)
                adj = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(O, alpha=-lr * adj)
                if wd != 0: p.add_(p, alpha=-lr * wd)
        return loss

class HybridLoRAPreMuon(Optimizer):
    def __init__(self, model, lr=2e-2, momentum=0.95, weight_decay=0.0, rank=16):
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim == 2 and 'embed' not in name.lower() and 'lm_head' not in name.lower():
                muon_params.append(p)
            else:
                adamw_params.append(p)
        self.muon_opt = LoRAPreMuon(muon_params, lr=lr, momentum=momentum, weight_decay=weight_decay, rank=rank)
        self.adamw_opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.muon_opt.param_groups + self.adamw_opt.param_groups
        self.state = {}
        super().__init__(self.param_groups, {})
    def zero_grad(self, set_to_none=True):
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adamw_opt.zero_grad(set_to_none=set_to_none)
    def step(self, closure=None):
        loss = self.muon_opt.step(closure)
        self.adamw_opt.step()
        return loss