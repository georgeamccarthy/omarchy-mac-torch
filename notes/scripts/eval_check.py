"""Eval-mode outputs of device models (after a few training steps, so BN running stats are non-trivial) vs the CPU."""
import copy, torch, torch.nn.functional as F, omarchy_mac_torch
from omarchy_mac_torch.models import SeparableCNN, PlainCNN, fake_data
for cls in (SeparableCNN, PlainCNN):
    torch.manual_seed(0)
    ref = cls(); mod = copy.deepcopy(ref).to("omarchy_mac_torch")
    X, y = fake_data(64)
    for _ in range(3):
        for m, dev in ((ref, "cpu"), (mod, "omarchy_mac_torch")):
            F.cross_entropy(m(X.to(dev)), y.to(dev)).backward()
    ref.eval(); mod.eval(); Xe, _ = fake_data(200, 5)
    with torch.no_grad():
        a = ref(Xe); b = mod(Xe.to("omarchy_mac_torch")).cpu()
    print(cls.__name__, "eval max |diff|:", (a - b).abs().max().item(), " fallbacks:", dict(omarchy_mac_torch.FALLBACKS))
