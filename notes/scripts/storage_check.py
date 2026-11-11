"""Storage-level behaviour that PyTorch implements through storages, not ops: torch.save / torch.load (with views,
int64, bool and map_location), deepcopy, .view(dtype), generators and seeding, events. No monkeypatching is involved."""
import copy, io, torch, omarchy_mac_torch as o
D = o.DEVICE
t = torch.randn(3, 4).to(D)
buf = io.BytesIO()
torch.save({"t": t, "i": torch.arange(3).to(D), "b": t > 0, "v": t[1:, 2:]}, buf)
buf.seek(0); r = torch.load(buf)
assert r["t"].device.type == D and torch.equal(r["t"].cpu(), t.cpu()) and r["i"].cpu().tolist() == [0, 1, 2]
assert torch.equal(r["b"].cpu(), (t > 0).cpu()) and torch.equal(r["v"].cpu(), t[1:, 2:].cpu())
buf.seek(0); assert torch.load(buf, map_location="cpu")["t"].device.type == "cpu"
m = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2)).to(D)
m2 = copy.deepcopy(m)
assert m2[0].weight.device.type == D and torch.equal(m2[0].weight.cpu(), m[0].weight.cpu())
v = torch.randn(4).to(D)
assert torch.equal(v.view(torch.int32).cpu(), v.cpu().view(torch.int32))
assert torch.arange(3, dtype=torch.int64).mul(3_000_000_000).to(D).cpu().tolist() == [0, 3_000_000_000, 6_000_000_000]
g = torch.Generator(device=D)
g.manual_seed(1); a = torch.empty(3, device=D).uniform_(generator=g)
g.manual_seed(1); b = torch.empty(3, device=D).uniform_(generator=g)
assert torch.equal(a.cpu(), b.cpu())
torch.manual_seed(3); a = torch.randn(3, device=D).cpu(); torch.manual_seed(3); b = torch.randn(3, device=D).cpu()
assert torch.equal(a, b)
e = torch.Event(device=D); (torch.randn(500, 500, device=D) @ torch.randn(500, 500, device=D)); e.record(); e.synchronize()
assert e.query()
print("storage checks passed; fallbacks:", dict(o.FALLBACKS))
