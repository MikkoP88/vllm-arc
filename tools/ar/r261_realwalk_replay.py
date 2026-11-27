#!/usr/bin/env python3
"""r261_realwalk_replay.py -- the REAL-WALK table replay (P6.47 r261,
runs INSIDE an arc-image container with /dev/dri).

THE r258b VACUITY (this harness's reason to exist): r258b's
``make_table(tail=True)`` placed the 18 windowed ids at columns
622..639 of the 640-wide table (``col = width - len(ids)``), but the
kernel's window walk reads columns 36..53: the per-row window is
[sl[i]-1023, sl[i]] -> block columns floor((sl-1023)/64)..floor((sl-1)/64)
= 36..53 for sl in 3329..3446. Every r258b cell therefore read BLOCK 0
(zeros at the walked columns) eighteen times; the real crash ids were
never in the walk. The 36/36 CLEAN verdict only exonerates the
mis-placed form -- the real-walk geometry (18 DISTINCT blocks) was
never replayed offline. The r258b docstring even says "36 head pads +
the 18 ids at 36..53"; the code disagreed with its own docstring.

The engine's bt rows are the DUMP rows verbatim at columns 0..53:
36 zero-pads (the dropped out-of-window region pointing at block 0)
then the 18 real ids at columns 36..53. This harness places them
exactly so:

  windowed: bt[:, 0:36] = 0, bt[:, 36:54] = the 18 ids (width 640;
            columns 54..639 = 0, never walked per the r258a bounds
            audit)
  full-attn: bt[:, 0:27] = the 27 ids (width 320; unchanged, that
             placement was already faithful)

Tables (the crash rows; 18 real ids per windowed group, 27 per
full-attn):
  m8g  -- the r255 m8g crash leg (the first banked crash)
  r257 -- the r257 faithful replay (the second banked crash)
  v3   -- the r260 V3 crash (multi->multi; all five windowed groups
          are clean DESCENDING LIFO runs -- the least-fragmented
          crash geometry)
  v4   -- the r260 V4 crash (simple->multi; the minimal-history
          repro; g0 scattered, g1..g4 descending runs)
  ctl  -- the pristine-state control: CONSECUTIVE fresh ids per
          group (the r256/r257-minimal clean class) -- under the
          REAL walk, this discriminates scatter-vs-consecutive

Crossed with four pool fills (zero / recycled / random / nanband) x
3 seeds -- the r258b axis unchanged. Per cell: the faithful step
order (the five windowed native calls then the full-attn native
call), 45 s SIGALRM watchdog per call, GuC-reset settle sleeps, a
canary matmul abort. If the kernel+real-walk is the crash surface,
some table x fill cell prints HANG/DEVLOST and the offline repro is
in hand; if all 60 cells are clean, the remaining delta is the
engine's same-step store/interleave context (the r262 e2e lane
ablation: ARC_MXKV_NATIVE=0 on the V4 repro).

Geometry is the engine's exact form (native_decode.py + the banked
capture telemetry): q [118,16,256] bf16, pool [1600,64,8,528] uint8,
max_seqlen_k=40960 (r212f walk-capacity law), softmax_scale=256**-0.5,
window_size=(1023,0) -- the 256/64-via-_16 masked twin at qgroup 2;
and q [118,16,512], pool [1600,128,2,1056], scale=512**-0.5,
windowless -- the 512/128-via-_16 entry at qgroup 8. fmt="mxfp8".

Usage: r261_realwalk_replay.py <out.json>
"""

import json
import signal
import sys
import time

import torch

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

# ---- the crash geometry (byte-exact from the r255/r257/r260 banks) ---
ROWS = 118
CACHED = 3328          # num_computed_tokens
PROMPT = 3446          # prompt_token_ids_len (== the t0 rehit row's sl)
W = 1024               # gemma sliding_window
SCALE_W = 0.0625       # head_size 256 ** -0.5
SCALE_F = 512 ** -0.5
FMT = "mxfp8"          # the m8g posture (ARC_KVDTYPE=mxfp8)
MAX_SK = 40960         # bt_width * block_size both classes (r212f)

# The banked crash tables: per-group REAL ids (the windowed dump rows
# are 36 zero-pads + these 18; the full-attn rows are these 27 at
# 0..26). Sources: r255 m8g + r257 banks (the r258b literals), and
# the r260 V3/V4 req_lines dumps (results/2026-11-27-p647-r260-face-
# bisect/r260/v{3,4}_*/req_lines.txt -- extracted programmatically,
# verified n=18/27 and unique-per-group against the banked rows).
M8G = {
    "g0": [805, 818, 717, 758, 846, 785, 44, 760, 821, 767, 719, 796,
           412, 762, 13, 815, 84, 1206],
    "g1": [40, 41, 42, 43, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
           72, 1410, 132],
    "g2": [1198, 1197, 1196, 1195, 1194, 1193, 1192, 1191, 215, 216,
           217, 218, 219, 220, 221, 222, 1202, 319],
    "g3": [1061, 1060, 1059, 548, 409, 329, 603, 575, 540, 402, 545,
           362, 438, 600, 393, 537, 1069, 1203],
    "g4": [419, 318, 1381, 338, 516, 752, 325, 446, 314, 144, 1450,
           568, 608, 1395, 136, 359, 528, 96],
    "g5": [742, 1427, 514, 935, 554, 557, 1399, 1408, 1382, 343, 428,
           138, 1406, 932, 1407, 1429, 137, 142, 1432, 930, 470, 749,
           334, 143, 461, 421, 89],
}
R257 = {
    "g0": [62, 49, 134, 109, 21, 82, 184, 107, 46, 100, 132, 71, 19,
           105, 153, 52, 177, 231],
    "g1": [180, 181, 182, 183, 201, 202, 203, 204, 205, 206, 207, 208,
           209, 210, 211, 212, 590, 624],
    "g2": [267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278,
           279, 280, 281, 282, 235, 415],
    "g3": [338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
           350, 351, 352, 353, 366, 234],
    "g4": [619, 531, 709, 532, 618, 473, 707, 426, 551, 564, 623, 527,
           472, 583, 556, 528, 259, 224],
    "g5": [483, 710, 403, 416, 469, 402, 587, 596, 620, 431, 523, 558,
           594, 419, 595, 621, 557, 562, 617, 421, 524, 476, 450, 563,
           529, 397, 217],
}
# r260 V3 (multi->multi, the class-2 rehit): all five windowed groups
# are descending LIFO runs (the cold's own blocks) + a 2-id fresh tail.
V3 = {
    "g0": [594, 593, 592, 591, 590, 589, 588, 587, 586, 585, 584, 583,
           582, 581, 580, 579, 485, 396],
    "g1": [540, 539, 538, 537, 536, 535, 534, 533, 532, 531, 530, 529,
           528, 527, 526, 525, 391, 385],
    "g2": [470, 469, 468, 467, 466, 465, 464, 463, 462, 461, 460, 459,
           458, 457, 456, 455, 364, 358],
    "g3": [416, 415, 414, 413, 412, 411, 410, 409, 408, 407, 406, 405,
           404, 403, 402, 401, 353, 347],
    "g4": [346, 345, 344, 343, 342, 341, 340, 339, 338, 337, 336, 335,
           334, 333, 332, 331, 577, 578],
    "g5": [328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317,
           316, 315, 314, 313, 312, 311, 310, 309, 308, 307, 306, 305,
           304, 303, 480],
}
# r260 V4 (simple->multi, the minimal-history repro): g0 is the
# scattered reissue, g1..g4 descending runs + 2-id fresh tails.
V4 = {
    "g0": [577, 590, 538, 547, 618, 557, 707, 549, 593, 556, 540, 568,
           620, 551, 519, 587, 437, 332],
    "g1": [476, 475, 474, 473, 472, 471, 470, 469, 468, 467, 466, 465,
           464, 463, 462, 461, 327, 321],
    "g2": [422, 421, 420, 419, 418, 417, 416, 415, 414, 413, 412, 411,
           410, 409, 408, 407, 316, 310],
    "g3": [352, 351, 350, 349, 348, 347, 346, 345, 344, 343, 342, 341,
           340, 339, 338, 337, 305, 299],
    "g4": [298, 297, 296, 295, 294, 293, 292, 291, 290, 289, 288, 287,
           286, 285, 284, 283, 527, 561],
    "g5": [280, 279, 278, 277, 276, 275, 274, 273, 272, 271, 270, 269,
           268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257,
           240, 239, 432],
}
FILLS = ("zero", "recycled", "random", "nanband")
SEEDS = (0, 1, 2)

# Pool geometry: [num_blocks, block, Hk, slot_bytes]. num_blocks 1600
# covers every banked id (max 1450); max_seqlen_k stays at the engine's
# 40960 capacity form. Slot offsets mirror mx_slot_layout(fmt 0, d):
# kb [0, d), ks [d, d + d/32), vb [d + d/32, 2d + d/32), vs at
# 2d + d/32, slot = 2d + 2*(d/32).
NBLOCKS = 1600
HK_W = 8     # sliding-class kv heads per GPU (16/TP2; stride 4224)
HK_F = 2     # global-class kv heads per GPU (4/TP2; stride 2112)
BTW_W = 640  # windowed table width (40960/64; ids walked at 36..53)
BTW_F = 320  # full-attn table width (40960/128; ids at the head 0..26)
HQ = 16      # query heads per GPU both classes (32/TP2)
PADS_W = 36  # the dropped out-of-window region's zero-pad columns


def slot_bytes(d: int) -> int:
    n = d // 32
    return 2 * d + 2 * n


def fill_pool(pool: torch.Tensor, fill: str, seed: int, d: int) -> None:
    n = d // 32
    ks_off, vb_off, vs_off = d, d + n, 2 * d + n
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    if fill == "zero":
        return
    if fill == "nanband":
        pool[:, :, :, :ks_off] = 0x7F          # e4m3 NaN payload
        pool[:, :, :, ks_off:vb_off] = 0xFF    # e8m0 NaN scale codes
        pool[:, :, :, vb_off:vs_off] = 0x7F
        pool[:, :, :, vs_off:] = 0xFF
        return
    if fill == "random":
        pool.copy_(
            torch.randint(0, 256, pool.shape, dtype=torch.uint8,
                          generator=g).to(pool.device))
        return
    # recycled: e4m3 encodings of plausible K/V values + legit-band
    # e8m0 codes 119..125 (the r225k deep-layer band). The payload
    # bytes ride the float8 view so every byte is a VALID e4m3 encoding
    # (no NaN patterns); the scale bytes are the codes a real store
    # writes for O(1) activations.
    vals = torch.randn(pool.shape[0], pool.shape[1], pool.shape[2], d,
                       generator=g).mul(0.5)
    f8 = vals.to(torch.float32).to(torch.float8_e4m3fn)
    payload = f8.view(torch.uint8).reshape(pool.shape[0], pool.shape[1],
                                           pool.shape[2], d)
    codes = torch.randint(119, 126, (pool.shape[0], pool.shape[1],
                                      pool.shape[2], n),
                          generator=g, dtype=torch.uint8)
    pool[:, :, :, :ks_off].copy_(payload.to(pool.device))
    pool[:, :, :, ks_off:vb_off].copy_(codes.to(pool.device))
    pool[:, :, :, vb_off:vs_off].copy_(payload.to(pool.device))
    pool[:, :, :, vs_off:].copy_(codes.to(pool.device))


def make_table(ids, width: int, dev, padded_head: bool = False):
    """[ROWS, width] int32, the ENGINE's placement:
    padded_head=False (windowed): the dump row verbatim at columns
      0..53 -- PADS_W zero-pads at 0..35 (the dropped out-of-window
      region), the 18 real ids at 36..53, columns 54+ zero (outside
      the window walk per the r258a bounds audit).
    padded_head=True (full-attn): the 27 ids at the head 0..26."""
    bt = torch.zeros(ROWS, width, dtype=torch.int32, device=dev)
    if padded_head:
        bt[:, :len(ids)] = torch.tensor(
            ids, dtype=torch.int32, device=dev)
    else:
        bt[:, PADS_W:PADS_W + len(ids)] = torch.tensor(
            ids, dtype=torch.int32, device=dev)
    return bt


def ctl_ids() -> dict:
    """The pristine-state control: consecutive fresh ids (the
    r256/r257-minimal free-list shape), now under the REAL walk."""
    out = {}
    base = 40
    for gi in range(5):
        out[f"g{gi}"] = list(range(base + 18 * gi, base + 18 * (gi + 1)))
    out["g5"] = list(range(200, 227))
    return out


class HangError(Exception):
    pass


def arm_watchdog(seconds: int):
    def fire(signum, frame):  # noqa: ARG001 -- signal protocol
        raise HangError(f"watchdog {seconds}s")
    signal.signal(signal.SIGALRM, fire)
    signal.alarm(seconds)


def disarm_watchdog():
    signal.alarm(0)


def canary(dev) -> bool:
    try:
        a = torch.randn(256, 256, device=dev)
        b = a @ a
        return bool(torch.isfinite(b).all().item())
    except Exception:  # noqa: BLE001 -- the wedged-device probe
        return False


def cell(ids: dict, fill: str, seed: int, dev, log) -> str:
    """The faithful step: 5 windowed twins (256/64-via-_16 masked, the
    new P-C2 surface) then the full-attn twin (512/128-via-_16
    windowless). Returns per-call verdicts w0..w4,f0."""
    t0 = time.time()
    g = torch.Generator(device="cpu").manual_seed(seed)
    sl = torch.arange(CACHED + 1, CACHED + ROWS + 1, dtype=torch.int32)
    cu_q = torch.arange(ROWS + 1, dtype=torch.int32, device=dev)
    verdicts = {}

    def one(label, run):
        """Run one native call under its own 45 s watchdog; a
        hang/devlost aborts the rest of the cell (contaminated)."""
        arm_watchdog(45)
        try:
            run()
            verdicts[label] = "OK"
        except HangError:
            verdicts[label] = "HANG"
            return False
        except Exception as exc:  # noqa: BLE001 -- crash outcomes are data
            up = str(exc).upper()
            verdicts[label] = ("DEVLOST" if "DEVICE_LOST" in up
                               or "UR_RESULT_ERROR" in up
                               else f"ERR:{type(exc).__name__}")
            log(f"   [{label}] exc: {str(exc)[:200]}")
            return False
        finally:
            disarm_watchdog()
        return True

    # all allocation + fill OUTSIDE the watchdogs (CPU-side, ~10 s
    # class per pool; both pools live simultaneously as in the engine)
    pool_w = torch.zeros(
        NBLOCKS, 64, HK_W, slot_bytes(256), dtype=torch.uint8,
        device=dev)
    fill_pool(pool_w, fill, seed, 256)
    q_w = torch.randn(ROWS, HQ, 256, generator=g).to(
        torch.bfloat16).to(dev)
    bts_w = [make_table(ids[f"g{gi}"], BTW_W, dev) for gi in range(5)]

    pool_f = torch.zeros(
        NBLOCKS, 128, HK_F, slot_bytes(512), dtype=torch.uint8,
        device=dev)
    fill_pool(pool_f, fill, seed, 512)
    q_f = torch.randn(ROWS, HQ, 512, generator=g).to(
        torch.bfloat16).to(dev)
    bt_f = make_table(ids["g5"], BTW_F, dev, padded_head=True)

    for gi in range(5):
        def run_w(pool=pool_w, q=q_w, bt=bts_w[gi]):
            out = mx_paged_decode(
                q, pool, FMT, cu_q, sl.to(dev), bt,
                max_seqlen_k=MAX_SK, softmax_scale=SCALE_W,
                num_splits_kv=0, window_size=(W - 1, 0))
            torch.xpu.synchronize()
            del out

        if not one(f"w{gi}", run_w):
            break
    else:
        def run_f(pool=pool_f, q=q_f, bt=bt_f):
            out = mx_paged_decode(
                q, pool, FMT, cu_q, sl.to(dev), bt,
                max_seqlen_k=MAX_SK, softmax_scale=SCALE_F,
                num_splits_kv=0, window_size=None)
            torch.xpu.synchronize()
            del out

        one("f0", run_f)

    del pool_w, q_w, bts_w, pool_f, q_f, bt_f
    torch.xpu.empty_cache()
    calls = ",".join(f"{k}={v}" for k, v in sorted(verdicts.items()))
    return f"{calls}|{int((time.time() - t0) * 1000)}"


def log(msg: str):
    print(msg, flush=True)


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "r261_results.json"
    dev = torch.device("xpu")
    tables = {"m8g": M8G, "r257": R257, "v3": V3, "v4": V4,
              "ctl": ctl_ids()}
    results = []
    hangs = devlosts = 0
    ncells = 0
    log(f"R261_START device={torch.xpu.get_device_name(0)} "
        f"rows={ROWS} cached={CACHED} w={W} max_sk={MAX_SK} "
        f"geom=256/64/hk8+512/128/hk2 walk=36..53")
    for tname, ids in tables.items():
        for fill in FILLS:
            for seed in SEEDS:
                ncells += 1
                calls, dt = cell(ids, fill, seed, dev, log).split("|")
                cell_hangs = sum(1 for c in calls.split(",")
                                 if c.endswith("=HANG"))
                cell_devlosts = sum(1 for c in calls.split(",")
                                    if c.endswith("=DEVLOST"))
                hangs += cell_hangs
                devlosts += cell_devlosts
                results.append(
                    {"table": tname, "fill": fill, "seed": seed,
                     "calls": dict(c.split("=") for c in calls.split(",")),
                     "ms": int(dt)})
                log(f"R261_CELL table={tname} fill={fill} seed={seed} "
                    f"[{calls}] dt={dt}ms")
                if cell_hangs or cell_devlosts:
                    # let the GuC reset + driver settle before the next
                    # cell's allocations (the reset precedes the python
                    # surface by 13-18 s)
                    time.sleep(30)
                if not canary(dev):
                    log("R261_DEVICE_WEDGED -- aborting the matrix")
                    break
            if not canary(dev):
                break
        if not canary(dev):
            break
    with open(out_path, "w", newline="\n") as f:
        json.dump(results, f, indent=1)
    log(f"R261_DONE hangs={hangs} devlosts={devlosts} cells={ncells} "
        f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
