# r16 replay v6 — standalone replay of the REAL captured layer-0 call.
# Discriminates defect-b (r16v5 evidence: flip detected inside the plain
# in_proj_ba F.linear BEFORE this step's SYCL op; retry on a fresh stream
# flipped again):
#   S2 flip / S3 crash  -> graph-machinery x prefill-gemm interaction
#   S3 crash standalone-no-graph -> kernel bug on real TP2 inputs
#   all clean           -> poison needs fuller serve context
# Stages: S0 baseline; S1 trivial graph capture+replay; S1d decode-shaped
# GDN op captured into an XPUGraph + replays (what warmup does before the
# first real prefill); S2 the REAL projection gemms (captured weights) —
# the exact serve flip window; S3 the REAL op call (captured qkvz/ba +
# metadata + reconstructed pools); S4 second op call. Stream state printed
# after every stage. --nograph drops S1/S1d.
import glob
import os
import sys

import torch

import vllm  # noqa: F401  registers custom ops
import vllm._xpu_ops  # noqa: F401  explicit

FLIPS = []


def state(tag):
    try:
        rec = torch.xpu.is_current_stream_capturing()
    except Exception as e:  # noqa: BLE001
        print(f"{tag}: query-err {e!r}", flush=True)
        return None
    mark = "FLIP <<< RECORDING" if rec else "clean"
    if rec:
        FLIPS.append(tag)
    print(f"{tag}: {mark}", flush=True)
    return rec


def load_caps(capdir):
    """Merge cap_<pid>_<idx>{e,p}.pt per idx: e = hidden + module weights
    (entry, clean stream), p = meta + qkvz/ba + pool rows (post-orig)."""
    merged = {}
    for path in sorted(glob.glob(f"{capdir}/*.pt")):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"load-fail {path}: {e!r}", flush=True)
            continue
        base = os.path.basename(path)          # cap_12327_00e.pt
        stem = base[:-4] if base.endswith(".pt") else base
        idx = stem.split("_")[-1][:-1] if stem[-1] in "ep" else stem.split("_")[-1]
        merged.setdefault(idx, {}).update(blob)
        merged[idx]["_files"] = merged[idx].get("_files", []) + [base]
    return merged


def to_xpu(v, dtype=None):
    if v is None or isinstance(v, (int, bool, float, str)):
        return v
    return v.to("xpu") if dtype is None else v.to("xpu").to(dtype)


def build_pools(module, rows):
    """Fresh-request semantics: has_initial_state=False -> kernel does not
    READ prior state, only writes it. Zeros of the captured geometry plus
    the captured referenced rows (fidelity for the retry/continuation case)."""
    pools = {}
    for key in ("conv_pool", "ssm_pool"):
        shape = module.get(f"{key}_shape")
        if shape is None:
            return None
        r = (rows or {}).get(f"{key}_rows")
        p = torch.zeros(tuple(shape), device="xpu",
                        dtype=r.dtype if r is not None else torch.float16)
        if r is not None and (rows or {}).get("row_ids") is not None:
            ids = rows["row_ids"].to("xpu").long()
            p[ids] = r.to("xpu").to(p.dtype)
        pools[key] = p
    return pools


def replay(idx, cap, with_graphs):
    module, meta, info = cap["module"], cap.get("meta", {}), cap["info"]
    hid = cap["hidden_states"].to("xpu")
    tok = hid.size(0)
    dt = hid.dtype
    print(f"\n=== cap idx={idx} files={cap.get('_files')} info={info} "
          f"with_graphs={with_graphs}", flush=True)
    state("S0 baseline")

    nv_r = module["num_v_heads"] // module["tp_size"]
    hv = module["head_v_dim"]

    if with_graphs:
        try:
            side = torch.xpu.Stream()
            torch.xpu.graph.default_capture_stream = side
            dummy = torch.zeros(64, 64, device="xpu", dtype=dt)
            g = torch.xpu.graphs.XPUGraph()
            with torch.xpu.graph(g):
                y = dummy @ dummy
            torch.xpu.synchronize()
            for _ in range(3):
                g.replay()
            torch.xpu.synchronize()
            state("S1 trivial graph + replays")
        except Exception as e:  # noqa: BLE001
            print(f"S1 ERR: {type(e).__name__}: {str(e)[:300]}", flush=True)

        # decode-shaped GDN op inside a graph + replays = warmup's decode
        # graph activity that always precedes the first real prefill
        try:
            tok_d = 4
            gen = torch.Generator().manual_seed(20261028)
            qkvz_dim = (cap["qkvz"].size(1) if cap.get("qkvz") is not None else
                        (2 * module["num_k_heads"] // module["tp_size"])
                        * module["head_k_dim"] + nv_r * hv
                        + nv_r * module["head_k_dim"] + module["conv_weight"].shape[0])
            ba_dim = (cap["ba"].size(1) if cap.get("ba") is not None
                      else 2 * nv_r)
            qkvz_d = (torch.randn(tok_d, qkvz_dim, generator=gen)
                      * 0.02).to("xpu").to(dt)
            ba_d = (torch.randn(tok_d, ba_dim, generator=gen)
                    * 0.02).to("xpu").to(dt)
            core_d = torch.zeros(tok_d, nv_r, hv, device="xpu", dtype=dt)
            z_d = torch.empty_like(core_d)
            pools_d = build_pools(module, None) or {
                "conv_pool": torch.zeros(64, 3, 1024, device="xpu", dtype=dt),
                "ssm_pool": torch.zeros(64, nv_r, hv * 2, device="xpu", dtype=dt)}
            kw_d = dict(
                core_attn_out=core_d, z=z_d,
                projected_states_qkvz=qkvz_d, projected_states_ba=ba_d,
                num_k_heads=module["num_k_heads"],
                num_v_heads=module["num_v_heads"],
                head_k_dim=module["head_k_dim"], head_v_dim=hv,
                conv_state=pools_d["conv_pool"],
                ssm_state=pools_d["ssm_pool"],
                conv_weights=module["conv_weight"].to("xpu"),
                conv_bias=(module["conv_bias"].to("xpu")
                           if module.get("conv_bias") is not None else None),
                activation=module["activation"],
                A_log=module["A_log"].to("xpu"),
                dt_bias=module["dt_bias"].to("xpu"),
                num_prefills=0, num_decodes=1, num_spec_decodes=0,
                has_initial_state=torch.zeros(1, dtype=torch.bool, device="xpu"),
                non_spec_query_start_loc=torch.tensor(
                    [0, tok_d], device="xpu", dtype=torch.int32),
                non_spec_token_indx=torch.arange(
                    tok_d, device="xpu", dtype=torch.int32),
                non_spec_state_indices_tensor=torch.tensor(
                    [0], device="xpu", dtype=torch.int32),
                spec_query_start_loc=None, spec_token_indx=None,
                spec_state_indices_tensor=None,
                num_accepted_tokens=None, num_actual_tokens=tok_d,
                tp_size=module["tp_size"],
                reorder_input=not module["gqa_interleaved_layout"])
            gd = torch.xpu.graphs.XPUGraph()
            with torch.xpu.graph(gd):
                torch.ops._xpu_C.gdn_attention(**kw_d)
            torch.xpu.synchronize()
            for _ in range(3):
                gd.replay()
            torch.xpu.synchronize()
            state("S1d decode-graph + replays")
        except Exception as e:  # noqa: BLE001
            print(f"S1d ERR (decode-graph): {type(e).__name__}: "
                  f"{str(e)[:300]}", flush=True)

    # S2: the REAL projection gemms — serve's exact flip window
    # (onset stack: in_proj_ba F.linear, qwen_gdn_linear_attn.py:992).
    # fp8-quantized weights cannot F.linear against fp16 hidden (the serve
    # path goes through the fp8 quant method) — fall back to a
    # shape-matched fp16 gemm: window fidelity, not value fidelity.
    try:
        def w_or_shape(wkey):
            w = module[wkey].to("xpu")
            if w.dtype != dt:
                print(f"S2 note: {wkey} is {w.dtype} — shape-fidelity gemm "
                      f"only", flush=True)
                return torch.randn(tuple(w.shape), dtype=dt, device="xpu") \
                    * 0.02
            return w
        wq = w_or_shape("in_proj_qkvz.weight")
        wb = w_or_shape("in_proj_ba.weight")
        bq = (module["in_proj_qkvz.bias"].to("xpu")
              if module.get("in_proj_qkvz.bias") is not None else None)
        bb = (module["in_proj_ba.bias"].to("xpu")
              if module.get("in_proj_ba.bias") is not None else None)
        qkvz = torch.nn.functional.linear(hid, wq, bq)
        ba = torch.nn.functional.linear(hid, wb, bb)
        torch.xpu.synchronize()
        state("S2 real projection gemms")
    except Exception as e:  # noqa: BLE001
        print(f"S2 ERR: {type(e).__name__}: {str(e)[:300]}", flush=True)
        qkvz = to_xpu(cap["qkvz"])
        ba = to_xpu(cap["ba"])

    if cap.get("qkvz") is not None:  # captured recompute wins if present
        qkvz = to_xpu(cap["qkvz"])
        ba = to_xpu(cap["ba"])

    pools = build_pools(module, cap)
    core = torch.zeros(tok, nv_r, hv, device="xpu", dtype=dt)
    z = torch.empty_like(core)
    kw = dict(
        core_attn_out=core, z=z,
        projected_states_qkvz=qkvz, projected_states_ba=ba,
        num_k_heads=module["num_k_heads"],
        num_v_heads=module["num_v_heads"],
        head_k_dim=module["head_k_dim"], head_v_dim=hv,
        conv_state=(pools or {}).get("conv_pool"),
        ssm_state=(pools or {}).get("ssm_pool"),
        conv_weights=module["conv_weight"].to("xpu"),
        conv_bias=(module["conv_bias"].to("xpu")
                   if module.get("conv_bias") is not None else None),
        activation=module["activation"],
        A_log=module["A_log"].to("xpu"),
        dt_bias=module["dt_bias"].to("xpu"),
        num_prefills=meta.get("num_prefills", 1),
        num_decodes=meta.get("num_decodes", 0),
        num_spec_decodes=meta.get("num_spec_decodes", 0),
        has_initial_state=to_xpu(meta.get("has_initial_state")),
        non_spec_query_start_loc=to_xpu(meta.get("non_spec_query_start_loc")),
        non_spec_token_indx=to_xpu(meta.get("non_spec_token_indx")),
        non_spec_state_indices_tensor=to_xpu(
            meta.get("non_spec_state_indices_tensor")),
        spec_query_start_loc=to_xpu(meta.get("spec_query_start_loc")),
        spec_token_indx=to_xpu(meta.get("spec_token_indx")),
        spec_state_indices_tensor=to_xpu(
            meta.get("spec_state_indices_tensor")),
        num_accepted_tokens=meta.get("num_accepted_tokens"),
        num_actual_tokens=meta.get("num_actual_tokens", tok),
        tp_size=module["tp_size"],
        reorder_input=not module["gqa_interleaved_layout"])
    # schema declares Tensor? optionals — None values are legal (the real
    # step passed non_spec_token_indx=None; v5 lesson: never drop them)
    try:
        print(f"S3 kwargs: tok={tok} meta_keys={sorted(meta)} "
              f"pool_shapes="
              f"{[tuple(pools[k].shape) if pools else None for k in ('conv_pool', 'ssm_pool')]}",
              flush=True)
        torch.ops._xpu_C.gdn_attention(**kw)
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"S3 ERR op: {type(e).__name__}: {str(e)[:400]}", flush=True)
        return False
    state("S3 after real prefill op")
    try:
        torch.ops._xpu_C.gdn_attention(**kw)
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"S4 ERR op2: {type(e).__name__}: {str(e)[:400]}", flush=True)
        return False
    state("S4 after second op")
    return True


def main():
    capdir = sys.argv[1] if len(sys.argv) > 1 else "/root/r16cap/out"
    with_graphs = "--nograph" not in sys.argv
    print("schema:", torch.ops._xpu_C.gdn_attention.default._schema, flush=True)
    merged = load_caps(capdir)
    # real-request caps = post blob whose metadata actually materialized
    # (dummy-profile caps carry meta._amd_type NoneType and would only
    # schema-error the op call)
    real = {k: v for k, v in merged.items()
            if isinstance(v.get("meta"), dict)
            and v["meta"].get("_amd_type") == "dict"}
    if not real:
        real = {k: v for k, v in merged.items()
                if any(f.endswith("p.pt") for f in v.get("_files", []))}
    if not real:
        real = merged
    if not real:
        print(f"no caps in {capdir}", flush=True)
        return 2
    ok = True
    for idx in sorted(real)[:2]:
        ok = replay(idx, real[idx], with_graphs) and ok
    print(f"RESULT flips={FLIPS if FLIPS else 'NONE'}", flush=True)
    return 0 if not FLIPS and ok else 1


if __name__ == "__main__":
    sys.exit(main())
