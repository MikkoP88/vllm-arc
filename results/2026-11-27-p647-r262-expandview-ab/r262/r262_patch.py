#!/usr/bin/env python3
"""r262_patch.py -- build an r262 mx_kv overlay from the pristine image tree.

Usage (ON THE RIG, host python3):
    python3 r262_patch.py <overlay_dir> --mode {on,off,probe}

<overlay_dir> must hold an unmodified copy of the cert29 image's
vllm_gaudi/xpu/patches/mx_kv/ tree (docker cp out of the image). The
pristine sha256 of the two touched files is checked first, then each
edit is applied by exact-once anchored replacement -- a second run on
an already-patched tree refuses instead of double-applying. The mode
is baked into the bytes (no env forwarding: the serve wrapper exports
only a whitelist of ARC_* variables into the engine process), so each
overlay's sha256s are the positive proof of what a phase ran:

  on     native_decode.py: the FIX (the landing candidate) + a boot
         marker; engine.py untouched
  off    native_decode.py: the boot marker ONLY (the control -- the
         certified bytes' behavior on a mounted overlay)
  probe  the fix + engine.py's leg-only diagnostic (always on in this
         overlay; never lands)

THE DEFECT (P6.47 r262, static audit 2026-11-27). The windowed synth
path (engine.py, continuation prefill 8 < q_len <= 128) hands
mx_decode_native the vendor-shaped stride-0 expand of ONE persistent
block-table row: bt_row.expand(q_len, -1). The Triton lanes multiply
the row index by block_tables.stride(0) -- right for an expand view.
The native wheel does not: mx_paged_decode_xe2.cpp passes
block_table.data_ptr() and size(1), and the kernel addresses row r at
idx_b * max_pages_per_seq -- a CONTIGUOUS pitch. So rows 1..q-1 read
the persistent table's FOLLOWING rows (zeros under single-request
traffic -> block 0) and, past its max_num_reqs (64) rows, whatever the
allocator placed after it, as block ids. Every crash in the rung chain
is a q_len > 64 synth ride (the multi rehit: 118 rows); the 56/57-row
essay/code rehits stay inside the zeroed table.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MXKV_DIR = "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv"

# The cert29 image bytes (== the tree at 16effb49): the patch is
# authored against these and refuses any drift.
PRISTINE_SHA256 = {
    "engine.py": "479d5b8621355eb2db23080142961095212ebc7491688ec909b8195b80edc499",
    "native_decode.py": "5f0db2d83aa79f7b2da7a7c6b28fccf79b10124d05380dd3ea5a921ee966dbdd",
}

# ---- native_decode.py --------------------------------------------------

ND_OLD_A = "_telemetry_logged: set[tuple] = set()\n"

ND_NEW_A = ND_OLD_A + '''
# r262 STRIDE LAW (2026-11-27, P6.47 gemma DEVICE_LOST). The wheel
# addresses a block-table row as base + row * bt_width -- a CONTIGUOUS
# pitch (mx_paged_decode_mainloop.hpp: b_offset = idx_b *
# max_pages_per_seq; mx_paged_decode_xe2.cpp passes
# block_table.data_ptr() and size(1), never a stride). The windowed
# synth path hands this bridge the vendor-shaped stride-0 expand of
# ONE persistent-table row: the Triton lanes multiply by stride(0)
# (right for an expand view), the wheel does not -- rows 1..q-1 read
# the persistent table's FOLLOWING rows (zeros -> block 0) and, past
# its max_num_reqs rows, whatever the allocator placed after it, as
# block ids. Every caller's contract here is per-row tables, so this
# boundary is the one place that can enforce it. is_contiguous() is
# host-side metadata; contiguous() is a device copy (capture-safe).
_bt_law_logged: set = set()
'''

ND_OLD_B = """    mx_paged_decode = _load()
    n_rows = q_rows.shape[0]
"""

ND_NEW_B = """    mx_paged_decode = _load()
    n_rows = q_rows.shape[0]
    if ("boot",) not in _bt_law_logged:
        _bt_law_logged.add(("boot",))
        logging.getLogger("vllm.mx_native").info(
            "MXKV native bridge r262 overlay: mode=%s", "MODETOKEN"
        )
FIXBLOCK"""

ND_FIXBLOCK = """    if not bt_rows.is_contiguous():
        _bt0 = int(bt_rows.stride(0))
        bt_rows = bt_rows.contiguous()
        _mkey = (_bt0, int(kv_cache.data_ptr()))
        if _mkey not in _bt_law_logged:
            _bt_law_logged.add(_mkey)
            logging.getLogger("vllm.mx_native").info(
                "MXKV native bt materialized: rows=%d bt_width=%d stride0=%d "
                "pool=0x%x (r262 stride law: the wheel addresses rows at a "
                "contiguous pitch)",
                n_rows,
                int(bt_rows.shape[1]),
                _bt0,
                int(kv_cache.data_ptr()),
            )
"""

# ---- engine.py (probe mode only) --------------------------------------

EN_OLD_A = """_trace_calls = 0

# v4 verify-row sharing (A11 lever 2, 2026-11-12): ONE walk per request
"""

EN_NEW_A = """_trace_calls = 0

# r262 leg-only diagnostic (P6.47 gemma DEVICE_LOST, 2026-11-27). At
# the windowed synth path, log once per (block table, pool) class the
# block-table rows the native wheel dereferences for that call: the
# wheel addresses row r at base + r * bt_width, so the stride-0 expand
# of ONE persistent-table row shows it that row for r = 0, the table's
# following rows for 1 <= r < max_num_reqs, and adjacent device memory
# beyond. Rows past the table are read back through as_strided -- OUT
# OF BOUNDS BY DESIGN on a sacrificial boot; a fault there is itself
# the finding. Debug instrument only; never lands.
_BTPROBE = True
_bt_probe_seen: set = set()


def _bt_probe(
    bt_row: torch.Tensor,  # (1, max_blocks) int32 -- the request's table row
    q_len: int,
    seq_len: int,
    kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_bytes)
    sliding_window: int,
    smap: torch.Tensor,  # the metadata's slot mapping (int64)
) -> None:
    \"\"\"Log the block-table rows the wheel dereferences for one synth call.\"\"\"
    try:
        width = int(bt_row.shape[1])
        block_size = int(kv_cache.shape[1])
        pool_blocks = int(kv_cache.shape[0])
        item = bt_row.element_size()
        off = int(bt_row.storage_offset())
        base = int(bt_row.data_ptr()) - item * off
        st_bytes = int(bt_row.untyped_storage().nbytes())
        key = (base, int(kv_cache.data_ptr()), q_len)
        if key in _bt_probe_seen:
            return
        _bt_probe_seen.add(key)
        smap_base = int(smap.data_ptr()) if smap is not None else 0
        smap_bytes = int(smap.untyped_storage().nbytes()) if smap is not None else 0
        table_end = base + st_bytes
        logger.info(
            "MXKVBTPROBE call q=%d width=%d row_stride=%d table_rows=%d "
            "storage_bytes=%d row0_storage_off=%d pool_blocks=%d block_size=%d "
            "window=%d seq_len=%d table_base=0x%x table_end=0x%x pool=0x%x "
            "smap_base=0x%x smap_bytes=%d smap_adjacent=%s",
            q_len,
            width,
            int(bt_row.stride(0)),
            st_bytes // (item * width),
            st_bytes,
            off,
            pool_blocks,
            block_size,
            int(sliding_window),
            seq_len,
            base,
            table_end,
            int(kv_cache.data_ptr()),
            smap_base,
            smap_bytes,
            str(smap_base == table_end),
        )
        for r in range(q_len):
            sl_r = (seq_len - q_len) + r + 1
            c1 = (sl_r - 1) // block_size
            c0 = max(sl_r - 1 - sliding_window, 0) // block_size
            past = item * (off + r * width) - st_bytes  # bytes past the table end
            tag = "table" if past < 0 else "BEYOND"
            try:
                row = torch.as_strided(bt_row, (1, width), (width, 1), off + r * width)
                ids = row[0, c0 : c1 + 1].to(torch.int64).cpu().tolist()
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "MXKVBTPROBE row=%d sl=%d %s past_end_bytes=%d READ_FAILED %r",
                    r,
                    sl_r,
                    tag,
                    past,
                    exc,
                )
                return
            logger.info(
                "MXKVBTPROBE row=%d sl=%d cols=%d..%d %s past_end_bytes=%d "
                "smap_rel=%d ids=%s min=%d max=%d nonzero=%d oob=%d",
                r,
                sl_r,
                c0,
                c1,
                tag,
                past,
                past - (smap_base - table_end) if smap_bytes else -1,
                ids,
                min(ids),
                max(ids),
                sum(1 for v in ids if v != 0),
                sum(1 for v in ids if v < 0 or v >= pool_blocks),
            )
    except Exception as exc:  # noqa: BLE001 -- debug instrument, never kill serving
        logger.error("MXKVBTPROBE failed: %r", exc)


# v4 verify-row sharing (A11 lever 2, 2026-11-12): ONE walk per request
"""

EN_OLD_B = """                synth_bt = bt_row.expand(q_len, -1)
                if self.sliding_window:
"""

EN_NEW_B = """                synth_bt = bt_row.expand(q_len, -1)
                if self.sliding_window:
                    _bt_probe(
                        bt_row,
                        q_len,
                        seq_len,
                        kv_cache,
                        self.sliding_window,
                        getattr(attn_metadata, "slot_mapping", None),
                    )
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> str:
    # bytes round-trip: no newline translation on any host (memory law:
    # p139 write-site newline -- every open() pins the byte form).
    return path.read_bytes().decode("utf-8")


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _apply(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"R262_PATCH_ANCHOR_FAIL {what}: expected exactly 1 match, found {n}")
    return text.replace(old, new, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("overlay_dir")
    ap.add_argument("--mode", choices=("on", "off", "probe"), required=True)
    args = ap.parse_args()
    ov = Path(args.overlay_dir)
    mode = args.mode

    for name, want in PRISTINE_SHA256.items():
        path = ov / name
        if not path.is_file():
            raise SystemExit(f"R262_PATCH_NO_FILE {path}")
        got = _sha256(path)
        if got != want:
            raise SystemExit(f"R262_PATCH_PRISTINE_MISMATCH {name} sha256={got} want={want}")

    nd_path = ov / "native_decode.py"
    nd = _read(nd_path)
    if "\r" in nd:
        raise SystemExit("R262_PATCH_CRLF native_decode.py")
    nd = _apply(nd, ND_OLD_A, ND_NEW_A if mode != "off" else ND_OLD_A + "_bt_law_logged: set = set()\n", "native A")
    marker = ND_NEW_B.replace("MODETOKEN", mode).replace("FIXBLOCK", ND_FIXBLOCK if mode != "off" else "")
    nd = _apply(nd, ND_OLD_B, marker, "native B")
    _write(nd_path, nd)

    if mode == "probe":
        en_path = ov / "engine.py"
        en = _read(en_path)
        if "\r" in en:
            raise SystemExit("R262_PATCH_CRLF engine.py")
        en = _apply(en, EN_OLD_A, EN_NEW_A, "engine A")
        en = _apply(en, EN_OLD_B, EN_NEW_B, "engine B")
        _write(en_path, en)
        print(f"R262_PATCHED engine.py sha256={_sha256(en_path)}")
    print(f"R262_PATCHED native_decode.py sha256={_sha256(nd_path)} mode={mode}")
    py = ov / "__pycache__"
    if py.exists():
        print(f"R262_PATCH_WARN pycache_present {py}")
    print(f"R262_PATCH_OK mode={mode} dir={ov}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
