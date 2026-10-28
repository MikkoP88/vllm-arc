import torch, torch.xpu
def flag():
    try: return torch.xpu.is_current_stream_capturing()
    except Exception as e: return f"ERR{e!r}"
def dezombie(tag):
    if flag() is True:
        torch.xpu.set_stream(torch.xpu.Stream())
        print(f"  [{tag}] DEZOMBIE swapped to fresh stream")
    return flag()
def exec_check(tag):
    try:
        t = torch.full((4,), 123.456, device="xpu"); torch.xpu.synchronize()
        ok = all(abs(v-123.456)<1e-2 for v in t.cpu().tolist())
        print(f"  [{tag}] exec_ok={ok} vals={t.cpu().tolist()[:2]}")
    except Exception as e:
        print(f"  [{tag}] exec ERR: {e!r}")

pool = torch.xpu.graph_pool_handle()
print("== T-A: 5 pool-shared captures on current(default) stream ==")
for i in range(5):
    try:
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, pool=pool, stream=torch.xpu.current_stream()):
            a = torch.randn(64,64,device="xpu"); b = a @ a
        print(f"  cap{i}: end returned, flag={flag()}")
    except Exception as e:
        print(f"  cap{i} EXC {e!r} flag={flag()}"); break
exec_check("T-A")

print("== T-B: event record INSIDE capture ==")
dezombie("pre-TB")
try:
    g = torch.xpu.XPUGraph(); ev = torch.xpu.Event()
    with torch.xpu.graph(g, stream=torch.xpu.current_stream()):
        x = torch.full((4,), 5.0, device="xpu"); y = x+2.0
        ev.record()
    print(f"  end returned, flag={flag()}")
    exec_check("T-B")
    try: ev.synchronize(); print("  ev.sync OK")
    except Exception as e: print(f"  ev.sync RAISED: {e!r}")
except Exception as e:
    print(f"  T-B EXC {e!r} flag={flag()}")
exec_check("T-B-post")

print("== T-C: record + side.wait_event INSIDE capture ==")
dezombie("pre-TC")
try:
    g = torch.xpu.XPUGraph(); ev2 = torch.xpu.Event(); side = torch.xpu.Stream()
    with torch.xpu.graph(g, stream=torch.xpu.current_stream()):
        x = torch.full((4,), 6.0, device="xpu"); y = x+3.0
        ev2.record()
        side.wait_event(ev2)
    print(f"  end returned, flag={flag()}")
except Exception as e:
    print(f"  T-C EXC {e!r} flag={flag()}")
exec_check("T-C")
print("PROBE2_DONE")
