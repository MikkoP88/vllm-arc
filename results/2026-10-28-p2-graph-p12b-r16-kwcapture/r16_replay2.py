# r16b — two-process oneCCL mini-serve: the last untested device work in
# defect-b's poison window (first real step's preamble = embedding gemm +
# TP allreduce) after r16a exonerated the GDN kernel, the projection gemms
# and graph capture/replay standalone (all clean with REAL kwargs).
#
# Serve chain replicated per rank (torchrun --nproc_per_node 2, ccl):
#   S0  stream state baseline
#   S1  EAGER allreduce (embedding-style)                 -> expect clean
#   S2  XPUGraph capture CONTAINING an allreduce (decode graphs on TP2
#       capture every layer's allreduce) + replays         -> health warmup
#   S3  EAGER allreduce AGAIN (the first real prefill)    -> TRIGGER?
#   S4  real projection-shape gemms + real GDN op (captured kwargs)
# Stream state printed after every stage. Verdict:
#   flip at/after S2 or S3  -> oneCCL x command-graph = defect-b mechanism
#   all clean               -> poison needs vllm's runner context
import glob
import os

import torch
import torch.distributed as dist

import vllm  # noqa: F401
import vllm._xpu_ops  # noqa: F401

FLIPS = []
RANK = int(os.environ.get("RANK", "0"))


def state(tag):
    try:
        rec = torch.xpu.is_current_stream_capturing()
    except Exception as e:  # noqa: BLE001
        print(f"[r{RANK}] {tag}: query-err {e!r}", flush=True)
        return None
    mark = "FLIP <<< RECORDING" if rec else "clean"
    if rec:
        FLIPS.append(tag)
    print(f"[r{RANK}] {tag}: {mark}", flush=True)
    return rec


def allreduce(x):
    dist.all_reduce(x)
    return x


def main():
    dist.init_process_group(backend="xccl")
    torch.xpu.set_device(RANK)
    dev = f"xpu:{RANK}"
    gen = torch.Generator().manual_seed(20261028 + RANK)

    # small stand-in for the embedding output
    x = (torch.randn(832, 5120, generator=gen) * 0.02).to(dev).to(torch.float16)
    w = (torch.randn(5120, 8192, generator=gen) * 0.02).to(dev).to(torch.float16)

    state("S0 baseline")
    allreduce(x)
    torch.xpu.synchronize()
    state("S1 eager allreduce")

    # S2a: COMPLETED command graphs in the runtime (decode-graph analog,
    # no collective inside — collectives are capture-illegal, see S2c)
    try:
        side = torch.xpu.Stream(device=dev)
        torch.xpu.graph.default_capture_stream = side
        g = torch.xpu.graphs.XPUGraph()
        with torch.xpu.graph(g):
            y0 = x @ w
        torch.xpu.synchronize()
        for _ in range(3):
            g.replay()
        torch.xpu.synchronize()
        state("S2a completed graph + replays")
    except Exception as e:  # noqa: BLE001
        print(f"[r{RANK}] S2a ERR: {type(e).__name__}: {str(e)[:300]}",
              flush=True)

    # S2c: prove the capture-illegality of the collective (expected error)
    try:
        gc_ = torch.xpu.graphs.XPUGraph()
        with torch.xpu.graph(gc_):
            y1 = x @ w
            allreduce(y1)
        torch.xpu.synchronize()
        state("S2c graph WITH allreduce (unexpected success)")
    except Exception as e:  # noqa: BLE001
        print(f"[r{RANK}] S2c capture-illegal as expected: "
              f"{type(e).__name__}: {str(e)[:200]}", flush=True)

    # first real prefill analog: eager collective AFTER completed graphs —
    # THE untested combination (serve's exact order: captures -> warmup
    # replays -> first big prefill's xccl allreduce)
    try:
        allreduce(x)
        torch.xpu.synchronize()
        state("S3 eager allreduce AFTER completed graphs")
    except Exception as e:  # noqa: BLE001
        print(f"[r{RANK}] S3 ERR: {type(e).__name__}: {str(e)[:300]}",
              flush=True)

    try:
        q = x @ w[:, :48]
        torch.xpu.synchronize()
        state("S4 gemms")
    except Exception as e:  # noqa: BLE001
        print(f"[r{RANK}] S4 ERR: {type(e).__name__}: {str(e)[:300]}",
              flush=True)

    # real GDN op from this rank's r16a capture, if present
    try:
        capdir = os.environ.get("R16_CAP_DIR", "/root/r16cap/out")
        pf = sorted(glob.glob(f"{capdir}/cap_*_01p.pt"))
        ef = sorted(glob.glob(f"{capdir}/cap_*_01e.pt"))
        if pf and ef:
            cap = torch.load(pf[RANK % len(pf)], map_location="cpu",
                             weights_only=False)
            cap.update(torch.load(ef[RANK % len(ef)], map_location="cpu",
                                  weights_only=False))
            m, meta = cap["module"], cap["meta"]
            hid = cap["hidden_states"].to(dev)
            tok = hid.size(0)
            nv_r = m["num_v_heads"] // m["tp_size"]
            pools = {k: torch.zeros(tuple(m[f"{k}_shape"]), device=dev,
                                    dtype=torch.float16)
                     for k in ("conv_pool", "ssm_pool")}
            kw = dict(
                core_attn_out=torch.zeros(tok, nv_r, m["head_v_dim"],
                                          device=dev, dtype=hid.dtype),
                z=torch.empty(tok, nv_r, m["head_v_dim"], device=dev,
                              dtype=hid.dtype),
                projected_states_qkvz=cap["qkvz"].to(dev),
                projected_states_ba=cap["ba"].to(dev),
                num_k_heads=m["num_k_heads"], num_v_heads=m["num_v_heads"],
                head_k_dim=m["head_k_dim"], head_v_dim=m["head_v_dim"],
                conv_state=pools["conv_pool"], ssm_state=pools["ssm_pool"],
                conv_weights=m["conv_weight"].to(dev),
                conv_bias=(m["conv_bias"].to(dev)
                           if m.get("conv_bias") is not None else None),
                activation=m["activation"], A_log=m["A_log"].to(dev),
                dt_bias=m["dt_bias"].to(dev),
                num_prefills=meta.get("num_prefills", 1),
                num_decodes=meta.get("num_decodes", 0),
                num_spec_decodes=meta.get("num_spec_decodes", 0),
                has_initial_state=meta.get("has_initial_state"),
                non_spec_query_start_loc=meta.get(
                    "non_spec_query_start_loc"),
                non_spec_token_indx=meta.get("non_spec_token_indx"),
                non_spec_state_indices_tensor=meta.get(
                    "non_spec_state_indices_tensor"),
                spec_query_start_loc=meta.get("spec_query_start_loc"),
                spec_token_indx=meta.get("spec_token_indx"),
                spec_state_indices_tensor=meta.get(
                    "spec_state_indices_tensor"),
                num_accepted_tokens=meta.get("num_accepted_tokens"),
                num_actual_tokens=meta.get("num_actual_tokens", tok),
                tp_size=m["tp_size"],
                reorder_input=not m["gqa_interleaved_layout"])
            for k, v in list(kw.items()):
                if torch.is_tensor(v) and v.device.type == "cpu":
                    kw[k] = v.to(dev)
            torch.ops._xpu_C.gdn_attention(**kw)
            torch.xpu.synchronize()
            state("S5 real GDN op (captured kwargs)")
    except Exception as e:  # noqa: BLE001
        print(f"[r{RANK}] S5 ERR: {type(e).__name__}: {str(e)[:300]}",
              flush=True)

    try:
        allreduce(x)
        torch.xpu.synchronize()
        state("S6 final allreduce")
    except Exception as e:  # noqa: BLE001
        print(f"[r{RANK}] S6 ERR: {type(e).__name__}: {str(e)[:300]}",
              flush=True)

    print(f"[r{RANK}] RESULT flips={FLIPS if FLIPS else 'NONE'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if not FLIPS else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
