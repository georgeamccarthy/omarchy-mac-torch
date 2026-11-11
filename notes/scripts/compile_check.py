"""torch.compile(backend="omarchy_mac_torch") vs the CPU (losses and gradients, training and eval) and vs eager speed
(with and without the fused nn.Sequential stacks). usage: compile_check.py [steps]"""
import copy, sys, time, torch, torch.nn.functional as F, omarchy_mac_torch as o
from omarchy_mac_torch import stacks
from omarchy_mac_torch.models import PlainCNN, SeparableCNN, ResNet18, MobileNetV3Small
D = o.DEVICE; steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for name, mk, shape in (("separablecnn", SeparableCNN, (128, 1, 28, 28)), ("plaincnn", PlainCNN, (128, 1, 28, 28)),
                        ("resnet18", ResNet18, (32, 3, 64, 64))):
    torch.manual_seed(0)
    ref = mk(); x, y = torch.randn(*shape), torch.randint(0, 10, (shape[0],))
    # correctness: one training step's gradients and an eval forward, compiled on the device vs the CPU
    m = copy.deepcopy(ref).to(D); cm = torch.compile(m, backend=D)
    F.cross_entropy(ref(x), y).backward(); F.cross_entropy(cm(x.to(D)), y.to(D)).backward()
    gerr = max(((a.grad - b.grad.cpu()).abs().max() / a.grad.abs().max().clamp_min(1e-6)).item() for a, b in zip(ref.parameters(), m.parameters()))
    ref.eval(); m.eval()
    with torch.no_grad():
        eerr = ((ref(x) - cm(x.to(D)).cpu()).abs().max() / ref(x).abs().max()).item()
    # speed: training steps
    res = {}
    for mode in ("eager", "eager+stacks", "compiled"):
        torch.manual_seed(0)
        mm = copy.deepcopy(ref).train().to(D)
        if mode == "eager+stacks": stacks.accelerate(mm)
        run = torch.compile(mm, backend=D) if mode == "compiled" else mm
        opt = torch.optim.SGD(mm.parameters(), lr=0.01, momentum=0.9); xd, yd = x.to(D), y.to(D)
        def step():
            loss = F.cross_entropy(run(xd), yd); opt.zero_grad(); loss.backward(); opt.step()
        for _ in range(3): step()
        o.synchronize(); t = time.perf_counter()
        for _ in range(steps): step()
        o.synchronize(); res[mode] = (time.perf_counter() - t) / steps * 1000
    print(f"{name:13s} grad rel err {gerr:.1e}  eval rel err {eerr:.1e}  train ms/step: " + "  ".join(f"{k} {v:.1f}" for k, v in res.items()), flush=True)
print("fallbacks:", dict(o.FALLBACKS))
