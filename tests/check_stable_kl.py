"""kldiv_stable equals kldiv on ordinary logits, and keeps finite gradients where kldiv's are NaN."""
import torch
from utils.modules import kldiv, kldiv_stable
torch.manual_seed(0)
for scale in [1.0, 5.0, 20.0]:
    a = (torch.randn(256, 10) * scale).double().requires_grad_()
    b = (torch.randn(256, 10) * scale).double().requires_grad_()
    v0 = kldiv(a, b, reduction='none').sum(1); v1 = kldiv_stable(a, b, reduction='none').sum(1)
    g0 = torch.autograd.grad(v0.sum(), [a, b]); g1 = torch.autograd.grad(v1.sum(), [a, b])
    print(f"scale {scale}: max |value diff| {(v0-v1).abs().max().item():.2e}, max |grad diff| "
          f"{max((x-y).abs().max().item() for x, y in zip(g0, g1)):.2e}")
# a teacher so confident that softmax underflows to exactly 0 in float32
t = torch.tensor([[200.0, -200.0] + [-200.0] * 8]).requires_grad_()
s = torch.zeros(1, 10).requires_grad_()
print("softmax(teacher) has exact zeros:", bool((torch.softmax(t, 1) == 0).any()))
for name, f in [("kldiv", kldiv), ("kldiv_stable", kldiv_stable)]:
    g = torch.autograd.grad(f(s, t, reduction='none').sum(), [t])[0]
    print(f"{name}: teacher grad finite = {bool(torch.isfinite(g).all())}")
