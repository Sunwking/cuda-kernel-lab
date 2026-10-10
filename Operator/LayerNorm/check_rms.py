import torch, torch.nn as nn

class MyRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))
        ms = (x * x).mean(dim=dims, keepdim=True)
        return self.weight * (x * torch.rsqrt(ms + self.eps))

torch.manual_seed(0)
x = torch.randn(2, 4, 8, dtype=torch.float64)
w = torch.randn(8, dtype=torch.float64)
eps = 1e-6

def run(m, w):
    with torch.no_grad():
        m.weight.copy_(w)
    xx = x.clone().requires_grad_(True)
    m(xx).sum().backward()
    return xx.grad, m.weight.grad

mine, off = MyRMSNorm(8, eps=eps), nn.RMSNorm(8, eps=eps)
dx_m, dw_m = run(mine, w)
dx_o, dw_o = run(off, w)
print("dx allclose:", torch.allclose(dx_m, dx_o, atol=1e-12))
print("dw allclose:", torch.allclose(dw_m, dw_o, atol=1e-12))

# 手推闭式反向 vs autograd（同一个 w）
g = torch.ones_like(x)
ms = (x * x).mean(-1, keepdim=True)
r = torch.rsqrt(ms + eps)
h = g * w
dx_manual = r * (h - x * (h * x).mean(-1, keepdim=True) / (ms + eps))
print("manual dx allclose:", torch.allclose(dx_manual, dx_m, atol=1e-12))
print("manual dx max err :", (dx_manual - dx_m).abs().max().item())
