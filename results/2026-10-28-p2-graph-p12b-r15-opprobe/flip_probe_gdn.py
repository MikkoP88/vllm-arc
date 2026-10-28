# r15 — op-level flip probe (defect-b): does the GDN SYCL prefill kernel
# (torch.ops._xpu_C.gdn_attention) flip the device-default stream into a
# recording state when XPU graph machinery is live? Standalone container
# probe, NO serve, NO monkeypatching (r12b law: never touch torch.ops).
# Legs cited: r12 (onset in compiled GDN prefill), r13 (eager onset in
# in_proj_ba mm, same layer), r14 (W14 posture still flips) — this probe
# isolates the one op all three rounds point at.
import os
import sys

import torch

import vllm  # noqa: F401  (loads the _xpu_C op registration)

try:
    import vllm._xpu_ops  # noqa: F401  (explicit: registers gdn_attention)
except Exception as e:
    print(f"xpu_ops import note: {e}", flush=True)

DEV = "xpu"
DT = torch.float16
HID = 5120
NK = 16         # num_k_heads FULL count (kernel divides by tp_size)
NV = 48         # num_v_heads FULL count (kernel divides by tp_size)
HK = 128
HV = 128
TP = 2
NTOK = 1024     # one prefill seq, 1024 tokens

flips = []


def state(tag):
    rec = torch.xpu.is_current_stream_capturing()
    if rec:
        flips.append(tag)
        print(f"FLIP <<< at {tag}: stream RECORDING", flush=True)
    else:
        print(f"ok    {tag}: clean", flush=True)
    return rec


def main():
    print("schema:", torch.ops._xpu_C.gdn_attention.default._schema, flush=True)
    state("S0 baseline")

    # ---- graph machinery live: capture + replay a small graph (r9 posture)
    # PROBE_NOGRAPH=1 skips this — the discriminator: if the kernel still
    # crashes L0 without any graph machinery, the probe inputs are bad;
    # if it runs clean eager, the graph-pool coexistence is the trigger.
    g = None
    if not os.environ.get("PROBE_NOGRAPH"):
        side = torch.xpu.Stream()
        torch.xpu.graph.default_capture_stream = side
        dummy = torch.zeros(64, 64, device=DEV, dtype=DT)
        g = torch.xpu.graphs.XPUGraph()
        with torch.xpu.graph(g):
            y = dummy @ dummy
        torch.xpu.synchronize()
        state("S1 after capture")
        for i in range(3):
            g.replay()
        torch.xpu.synchronize()
        state("S1b after replays")
    else:
        print("S1 skipped (PROBE_NOGRAPH)", flush=True)

    # ---- projections (the gemms the onset stacks catch the thread at)
    w_qkvz = torch.randn(8192, HID, device=DEV, dtype=DT) * 0.02
    w_ba = torch.randn(2 * NV // TP, HID, device=DEV, dtype=DT) * 0.02
    x = torch.randn(NTOK, HID, device=DEV, dtype=DT) * 0.1
    qkvz = torch.nn.functional.linear(x, w_qkvz)
    state("S2a in_proj_qkvz mm")
    ba = torch.nn.functional.linear(x, w_ba)
    state("S2b in_proj_ba mm")

    # ---- GDN core inputs (prefill, non-spec; v-tensors per-rank = NV//TP)
    NV_R = NV // TP
    core_out = torch.zeros(NTOK, NV_R, HV, device=DEV, dtype=DT)
    z = torch.empty_like(core_out)
    conv_dim = (2048 * 2 // TP) + (6144 // TP)  # q+k per rank + v per rank
    conv_state = torch.zeros(3, conv_dim, device=DEV, dtype=DT)  # SD layout (state_len, dim) — vllm default
    ssm_state = torch.zeros(NV_R, HK, HV, device=DEV, dtype=DT)
    conv_w = torch.randn(conv_dim, 4, device=DEV, dtype=DT) * 0.05
    conv_b = torch.randn(conv_dim, device=DEV, dtype=DT) * 0.05
    qsl = torch.tensor([0, NTOK], device=DEV, dtype=torch.int32)

    kwargs = dict(
        core_attn_out=core_out,
        z=z,
        projected_states_qkvz=qkvz,
        projected_states_ba=ba,
        num_k_heads=NK,
        num_v_heads=NV,
        head_k_dim=HK,
        head_v_dim=HV,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_w,
        conv_bias=conv_b,
        activation="silu",
        A_log=torch.randn(NK // TP, device=DEV, dtype=torch.float32) * 0.1,
        dt_bias=torch.randn(NK // TP, device=DEV, dtype=DT) * 0.1,
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        has_initial_state=torch.tensor([False], device=DEV, dtype=torch.bool),
        non_spec_query_start_loc=qsl,
        non_spec_token_indx=torch.tensor([0], device=DEV, dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([0], device=DEV, dtype=torch.int32),
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=NTOK,
        tp_size=TP,
        reorder_input=True,
    )

    def run_core(tag, **over):
        kw = dict(kwargs)
        kw.update(over)
        try:
            torch.ops._xpu_C.gdn_attention(**kw)
            torch.xpu.synchronize()
        except Exception as e:
            print(f"ERR   {tag}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            return False
        state(tag)
        return True

    stage = os.environ.get("PROBE_STAGE", "both")
    if stage in ("both", "decode"):
        # decode-shaped call FIRST (small; if it runs clean standalone the
        # prefill path is the specific poison)
        dec_out = torch.zeros(1, NV_R, HV, device=DEV, dtype=DT)
        dec_z = torch.empty_like(dec_out)
        run_core(
            "S5 gdn_attention DECODE c1",
            core_attn_out=dec_out,
            z=dec_z,
            projected_states_qkvz=qkvz[:1].contiguous(),
            projected_states_ba=ba[:1].contiguous(),
            num_prefills=0,
            num_decodes=1,
            non_spec_query_start_loc=torch.tensor([0, 1], device=DEV, dtype=torch.int32),
            non_spec_token_indx=torch.tensor([0], device=DEV, dtype=torch.int32),
            num_actual_tokens=1,
        )

    if stage in ("both", "prefill"):
        # prefill-shape core call — THE test (graph machinery live unless
        # PROBE_NOGRAPH)
        run_core("S4 gdn_attention PREFILL 1024")
        run_core("S4b gdn_attention PREFILL again")

    print(f"RESULT flips={flips if flips else 'NONE'}", flush=True)
    return 0 if flips else 1


if __name__ == "__main__":
    sys.exit(main())
