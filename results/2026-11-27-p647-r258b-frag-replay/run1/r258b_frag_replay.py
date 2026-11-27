#!/usr/bin/env python3
"""r258b_frag_replay.py -- the synthetic fragmented-table hang repro
(P6.47 r258, runs INSIDE an arc-image container with /dev/dri).

r257 isolated the state delta: the byte-faithful OFF-branch suite
crashed with the exact r255 geometry (computed=3328 scheduled=118) on
the suite-fragmented allocator, while the same request bytes ran CLEAN
on pristine state. The r258a static audit then cleared every address /
plan / reduce derivation in the 118-tail native path (launcher ->
kernel -> mainloop -> reduce): the per-row walk [k_block0, k_blocks)
covers ONLY the real-id region of the block table (36..53), all ids are
allocator-issued and in-pool, and the r208 zero-fill guards keep the
reduce on written slots. The walk is NOT the hang surface -- the
remaining content-dependent differentiators are (a) the fragmented
block-id ORDER (free-list scatter, vs pristine consecutive) and (b) the
BYTES in the never-zeroed fresh windowed blocks (recycled garbage from
the suite's freed allocations vs driver-fresh zeros).

This harness replays the crash step's native calls DIRECTLY (no engine)
on the cert29 wheel, crossing the exact banked crash tables with
controlled pool fills:

  tables: m8g   -- the r255 m8g crash leg's five windowed page-64 rows
                  + the page-128 full-attn row (the first banked crash)
          r257  -- the r257 faithful replay's rows (the second banked
                  crash, different id generation)
          ctl   -- the pristine-state control: 36 zero-pads + 18
                  CONSECUTIVE fresh ids (the r256/r257-minimal class)
  fills:  zero     -- torch.zeros (driver-fresh pages; the minimal-pair
                     analog)
          recycled -- plausible old-KV bytes: e4m3 encodings of N(0,1)
                     payloads + e8m0 scale codes 119..125 (the legit
                     store band, r225k deep-layer codes)
          random  -- whole-pool randint(0,256) (ancient-garbage
                     extreme: covers e8m0 0xFF and e4m3 NaN bytes)
          nanband -- adversarial corner: every scale byte 0xFF (e8m0
                     NaN per the MX spec), every payload byte 0x7F
                     (e4m3 NaN)
  seeds:  0, 1, 2 (q + fill vary; the tables are fixed literals)

Per cell: the FAITHFUL STEP ORDER -- the five windowed mx_paged_decode
calls (D=256 page-64 masked twin, W=1024 -> window (1023, 0), 118 synth
rows, sl 3329..3446, auto splits) then the full-attn call (D=512
page-128 windowless twin, 27-wide table). A SIGALRM watchdog (45 s)
declares HANG if a cell never completes; a UR_RESULT_ERROR_DEVICE_LOST
raised by a sync declares DEVLOST (the GuC-watchdog reset precedes the
python surface by 13-18 s -- a repro of the crash class). A canary
matmul after every cell aborts the matrix if the device wedges.

Verdict protocol (crash outcomes are DATA): every cell prints
R258B_CELL table=.. fill=.. seed=.. verdict=OK|HANG|DEVLOST|ERR dt=ms
and the run ends R258B_DONE hangs=<n> devlosts=<n> cells=<n>.

Usage: r258b_frag_replay.py <out.json>
"""

import json
import signal
import sys
import time

import torch

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

try:  # the full-attn twin's lane (the engine's synth else calls the
    # TRITON mx_decode for the windowless group, not the wheel)
    from vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode import mx_decode
    HAVE_TRITON = True
except Exception as exc:  # noqa: BLE001 -- the matrix survives without
    print(f"R258B_TRITON_IMPORT_FAILED {exc!r}", flush=True)
    HAVE_TRITON = False

# ---- the crash geometry (byte-exact from the r255/r257 banks) --------
ROWS = 118
CACHED = 3328          # num_computed_tokens
PROMPT = 3446          # prompt_token_ids_len (== the t0 rehit row's sl)
W = 1024               # gemma sliding_window
SCALE_W = 0.0625       # head_size 256 ** -0.5
SCALE_F = 512 ** -0.5

# The banked crash tables (tools/p647_r258_crashtables.py over
# results/2026-11-26-p647-cert29-gate/{m8g}/serve.log and
# results/2026-11-27-p647-r257-faithful-replay/r257/replay/
# devicelost_context.txt). Rows are the per-group block_ids lists with
# their zero-pads: 54 entries = 36 pads + 18 real (windowed, page 64);
# 27 real (full-attn, page 128).
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
FILLS = ("zero", "recycled", "random", "nanband")
SEEDS = (0, 1, 2)

# Pool geometry: [num_blocks, block, Hk, slot_bytes]. num_blocks 1600
# covers every banked id (max 1450); total_seqlen_k stays far under the
# 2^24 surface split. Slot offsets mirror mx_slot_layout(fmt 0, d):
# kb [0, d), ks [d, d + d/32), vb [d + d/32, 2d + d/32), vs at
# 2d + d/32, slot = 2d + 2*(d/32).
NBLOCKS = 1600
HK = 2


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


def make_table(ids, width: int, dev) -> torch.Tensor:
    """[ROWS, width] with the ids at the TAIL (the crash rows are
    zero-padded at the head: 36 pads then 18 real for page-64)."""
    bt = torch.zeros(ROWS, width, dtype=torch.int32, device=dev)
    bt[:, width - len(ids):] = torch.tensor(
        ids, dtype=torch.int32, device=dev)
    return bt


def ctl_ids() -> dict:
    """The pristine-state control: consecutive fresh ids (the
    r256/r257-minimal free-list shape)."""
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
    """The faithful step: 5 windowed twins (native wheel, the new P-C2
    surface) then the full-attn twin (the engine's synth else -> the
    TRITON mx_decode lane). Returns per-call verdicts w0..w4,f0."""
    t0 = time.time()
    g = torch.Generator(device="cpu").manual_seed(seed)
    sl = torch.arange(CACHED + 1, CACHED + ROWS + 1, dtype=torch.int32)
    cu_q = torch.arange(ROWS + 1, dtype=torch.int32, device=dev)
    verdicts = {}

    def one(label, run):
        """Run one native/triton call under the shared cell watchdog;
        a hang/devlost aborts the rest of the cell (contaminated)."""
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
        return True

    arm_watchdog(45)
    try:
        # one shared windowed pool per cell: the five groups' tables
        # differ, the fill distribution does not (fresh pools per group
        # cost 5x the CPU fill for no content-class difference)
        pool_w = torch.zeros(
            NBLOCKS, 64, HK, slot_bytes(256), dtype=torch.uint8,
            device=dev)
        fill_pool(pool_w, fill, seed, 256)
        q_w = None
        for gi in range(5):
            if q_w is None:
                q_w = torch.randn(ROWS, 16, 256, generator=g).to(
                    torch.bfloat16).to(dev)
            bt = make_table(ids[f"g{gi}"], 54, dev)

            def run_w(gi=gi, pool=pool_w, q=q_w, bt=bt):
                out = mx_paged_decode(
                    q, pool, 0, cu_q, sl.to(dev), bt,
                    max_seqlen_k=54 * 64, softmax_scale=SCALE_W,
                    num_splits_kv=0, window_size=(W - 1, 0))
                torch.xpu.synchronize()
                del out

            if not one(f"w{gi}", run_w):
                break
            del bt
        else:
            if HAVE_TRITON:
                pool = torch.zeros(
                    NBLOCKS, 128, HK, slot_bytes(512), dtype=torch.uint8,
                    device=dev)
                fill_pool(pool, fill, seed, 512)
                q = torch.randn(ROWS, 4, 512, generator=g).to(
                    torch.bfloat16).to(dev)
                bt = make_table(ids["g5"], 27, dev)

                def run_f(pool=pool, q=q, bt=bt):
                    out = mx_decode(q, pool, bt, sl.to(dev), SCALE_F)
                    torch.xpu.synchronize()
                    del out

                one("f0", run_f)
            else:
                verdicts["f0"] = "SKIP_NO_TRITON"
    finally:
        disarm_watchdog()
    torch.xpu.empty_cache()
    calls = ",".join(f"{k}={v}" for k, v in sorted(verdicts.items()))
    return f"{calls}|{int((time.time() - t0) * 1000)}"


def log(msg: str):
    print(msg, flush=True)


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "r258b_results.json"
    dev = torch.device("xpu")
    tables = {"m8g": M8G, "r257": R257, "ctl": ctl_ids()}
    results = []
    hangs = devlosts = 0
    ncells = 0
    log(f"R258B_START device={torch.xpu.get_device_name(0)} "
        f"rows={ROWS} cached={CACHED} w={W}")
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
                log(f"R258B_CELL table={tname} fill={fill} seed={seed} "
                    f"[{calls}] dt={dt}ms")
                if cell_hangs or cell_devlosts:
                    # let the GuC reset + driver settle before the next
                    # cell's allocations (the reset precedes the python
                    # surface by 13-18 s)
                    time.sleep(30)
                if not canary(dev):
                    log("R258B_DEVICE_WEDGED -- aborting the matrix")
                    break
            if not canary(dev):
                break
        if not canary(dev):
            break
    with open(out_path, "w", newline="\n") as f:
        json.dump(results, f, indent=1)
    log(f"R258B_DONE hangs={hangs} devlosts={devlosts} cells={ncells} "
        f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
