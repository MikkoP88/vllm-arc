import torch, torch.xpu, traceback
print("torch", torch.__version__, "dev", torch.xpu.get_device_name(0))
def flag():
    try: return torch.xpu.is_current_stream_capturing()
    except Exception as e: return f"ERR {e!r}"
def q(s): return hex(s.sycl_queue) if hasattr(s, "sycl_queue") else str(s)
def exec_check(tag):
    try:
        t = torch.full((4,), 123.456, device="xpu")
        torch.xpu.synchronize()
        ok = all(abs(v-123.456) < 1e-2 for v in t.cpu().tolist())
        print(f"[{tag}] exec_ok={ok} vals={t.cpu().tolist()}")
    except Exception as e:
        print(f"[{tag}] exec ERR: {e!r}")
d0 = torch.xpu.current_stream(); print("default stream", q(d0), "flag_before:", flag())

print("=== T3: SIDE-stream capture first (pristine default) ===")
try:
    g2 = torch.xpu.XPUGraph()
    with torch.xpu.graph(g2):
        x2 = torch.full((4,), 9.0, device="xpu"); y2 = x2 + 1.0
    print("T3 capture_end ok. flag(cur=default):", flag(), "cur still default:", torch.xpu.current_stream() == d0)
    exec_check("T3-after")
    try:
        g2.replay(); torch.xpu.synchronize(); print("T3 replay vals:", y2.cpu().tolist())
    except Exception as e: print("T3 replay EXC:", repr(e))
except Exception as e:
    print("T3 EXC:", repr(e)); traceback.print_exc()

print("=== T1: capture on CURRENT (default) stream, vllm style ===")
try:
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g, stream=torch.xpu.current_stream()):
        x = torch.full((4,), 7.0, device="xpu"); y = x * 3.0
    print("T1 capture_end RETURNED CLEAN. flag(cur):", flag(), "cur==default:", torch.xpu.current_stream() == d0)
except Exception as e:
    print("T1 capture EXC:", repr(e), "flag:", flag())
exec_check("T1-after")
try:
    g.replay(); torch.xpu.synchronize(); print("T1 replay vals:", y.cpu().tolist())
except Exception as e: print("T1 replay EXC:", repr(e))

print("=== T2: dezombie = set_stream(fresh) ===")
try:
    f1 = torch.xpu.Stream(); torch.xpu.set_stream(f1)
    print("T2 swapped to", q(f1), "flag:", flag())
    exec_check("T2-fresh")
except Exception as e: print("T2 EXC:", repr(e))
print("PROBE_DONE")
