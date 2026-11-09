# r128p_cut8_report.py -- P6.34 cut-8 stage-B analyzer (pure json, no torch).
# Reads R128B_OUT rows (r128b_jm_gate.py) and renders the per-cell
# verdict table for the P6.34 stop condition: EVERY cell jm < stock
# (or tie at a banked ceiling). Best-of-ARC_JM_SGF is the lawful jm
# number (the knob is runtime, read per call); stock/scalar measured on
# the default pass. Floors: 589 GB/s DRAM, 992 GB/s L2 (r128a bars).
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/r128b_rows.json"
rows = json.load(open(path))["rows"]

# shape -> bytes class (weights MB; <8 MB ~ L2-resident on B70 per r128a)
cells = {}
for r in rows:
    key = (r["tag"], r["lane"], r["M"])
    cells.setdefault(key, {})[r["sgf"]] = r

# shapes: (K, N) from the gate; weights MB = K*N*(0.5 if g16 else 1)
SH = {
    "gemma.up": (5376, 10752), "gemma.down": (10752, 5376),
    "gemma.qkv": (5376, 8192), "gemma.o": (5376, 5376),
    "qwen.up": (5120, 8704), "qwen.down": (8704, 5120),
    "qwen.o": (5120, 5120),
}

wins = ties = losses = 0
print(f"{'cell':34s} {'st(ms)':>8s} {'sk(ms)':>8s} {'jm best':>8s} "
      f"{'sgf':>5s} {'jm/st':>6s} {'GB/s':>6s} {'floor%':>7s} verdict")
for (tag, lane, m), per_sgf in sorted(cells.items()):
    st = sk = None
    for sgf, r in per_sgf.items():
        if sgf == "2048":
            st, sk = r["st"], r["sk"]
    cand = {s: r["jm"] for s, r in per_sgf.items()}
    if not cand:
        print(f"{tag}/{lane}/M{m}: NO JM ROWS (all failed?)")
        losses += 1
        continue
    best_sgf = min(cand, key=cand.get)
    jm = cand[best_sgf]
    K, N = SH[tag]
    mb = K * N * (0.5 if lane == "g16" else 1.0) / 1e6
    gbs = mb / jm / 1e-3 if jm > 0 else 0.0
    floor = 589.0 if mb >= 8 else 992.0
    fpct = 100.0 * gbs / floor
    if st is None or st != st:  # NaN guard (stock only on default pass)
        verdict = "no-stock-row"
    elif jm < st * 0.98:
        verdict = "WIN"
        wins += 1
    elif jm <= st * 1.02:
        verdict = "TIE(2%)"
        ties += 1
    else:
        verdict = "LOSS"
        losses += 1
    st_s = f"{st:.4f}" if st is not None and st == st else "--"
    sk_s = f"{sk:.4f}" if sk is not None and sk == sk else "--"
    print(f"{tag + '/' + lane + '/M' + str(m):34s} {st_s:>8s} {sk_s:>8s} "
          f"{jm:8.4f} {best_sgf:>5s} "
          f"{(st / jm if st is not None and st == st else 0):6.2f} "
          f"{gbs:6.0f} {fpct:6.1f}% {verdict}")

print(f"\ncells: {wins} WIN / {ties} TIE / {losses} LOSS  "
      f"(stop condition: 0 LOSS or LOSS-with-banked-math)")
allrows = [r for r in rows if r["sgf"] == "2048"]
if allrows:
    import statistics
    for lane in ("w16pt", "w16pc", "m816", "g16"):
        rs = [r for r in allrows if r["lane"] == lane]
        if rs:
            med = statistics.median(r["jm_vs_st"] for r in rs)
            print(f"  lane {lane}: median jm/st (default SGF) = {med:.2f}x "
                  f"over {len(rs)} cells")
