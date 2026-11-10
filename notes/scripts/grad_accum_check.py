"""Gradient accumulation across two backward passes (no zero_grad) must equal the CPU sum: guards the plan's persistent grad storage."""
import copy, torch, torch.nn.functional as F, omarchy_mac_torch
from omarchy_mac_torch.models import SeparableCNN, fake_data
torch.manual_seed(0)
ref = SeparableCNN(); mod = copy.deepcopy(ref).to("omarchy_mac_torch")
(X1, y1), (X2, y2) = fake_data(64, 1), fake_data(64, 2)
for xb, yb in ((X1, y1), (X2, y2)):
    F.cross_entropy(ref(xb), yb).backward()
    F.cross_entropy(mod(xb.to("omarchy_mac_torch")), yb.to("omarchy_mac_torch")).backward()
d = max((a.grad - b.grad.cpu()).abs().max().item() for a, b in zip(ref.parameters(), mod.parameters()))
print("max grad diff after 2 accumulated backward passes:", d)
