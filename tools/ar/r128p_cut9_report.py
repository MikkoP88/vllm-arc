# r128p_cut9_report.py -- P6.34 cut-9 stage-B analyzer (pure json, no torch).
# Reads R128B_OUT rows (r128b_jm_gate.py, dual-variant build8 wheel: rows
# carry var in {"5","14"}) and renders the per-cell verdict table for the
# P6.34 stop condition: EVERY cell jm < stock (or tie at a banked ceiling).
# Best-of-(ARC_JM_SGF, ARC_JM_VAR) is the lawful jm number (both knobs are
# runtime, read per call); stock/scalar measured on the default pass.
# Verdict per variant column (jm5 = lottery control, jm14 = candidate) and
# for the best variant overall. Floors: 589 GB/s DRAM, 992 GB/s L2 (r128a).
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/r128b_rows.json"
rows = json.load(open(path))["rows"]

# shape -> bytes class (weights MB; <8 MB ~ L2-resident on B70 per r128a)
cells = {}
for r in rows:
    key = (r["tag"], r["lane"], r["M"])
    cells.setdefault(key, {})[(r.get("var", "5"), r["sgf"])] = r

# shapes: (K, N) from the gate; weights MB = K*N*(0.5 if g16 else 1)
SH = {
    "gemma.up": (5376, 10752), "gemma.down": (10752, 5376),
    "gemma.qkv": (5376, 8192), "gemma.o": (5376, 5376),
    "qwen.up": (5120, 8704), "qwen.down": (8704, 5120),
    "qwen.o": (5120, 5120),
}


def verdicts(jm, st):
    if st is None or st != st:  # NaN guard (stock only on default pass)
        return "no-stock-row"
    if jm < st * 0.98:
        return "WIN"
    if jm <= st * 1.02:
        return "TIE(2%)"
    return "LOSS"


wins = {v: 0 for v in ("5", "14")}
ties = {v: 0 for v in ("5", "14")}
losses = {v: 0 for v in ("5", "14")}
bwins = bties = blosses = 0
print(f"{'cell':32s} {'st(ms)':>8s} {'jm5':>8s} {'v/s':>5s} "
      f"{'jm14':>8s} {'v/s':>5s} {'best':>8s} {'var/sgf':>9s} "
      f"{'GB/s':>6s} {'floor%':>7s} best-verdict")
for (tag, lane, m), per in sorted(cells.items()):
    st = sk = None
    for (var, sgf), r in per.items():
        if sgf == "2048":
            st, sk = r["st"], r["sk"]
    cols = {}
    for v in ("5", "14"):
        cand = {s: r["jm"] for (vr, s), r in per.items() if vr == v}
        cols[v] = min(cand, key=cand.get) if cand else None
    if not any(cols.values()):
        print(f"{tag}/{lane}/M{m}: NO JM ROWS (all failed?)")
        blosses += 1
        continue
    best_v = min((v for v in ("5", "14") if cols[v]),
                 key=lambda v: per[(v, cols[v])]["jm"])
    bs = cols[best_v]
    jm_best = per[(best_v, bs)]["jm"]
    K, N = SH[tag]
    mb = K * N * (0.5 if lane == "g16" else 1.0) / 1e6
    gbs = mb / jm_best if jm_best > 0 else 0.0  # MB / ms = GB/s
    floor = 589.0 if mb >= 8 else 992.0
    fpct = 100.0 * gbs / floor
    line = f"{tag + '/' + lane + '/M' + str(m):32s} "
    line += f"{st:.4f}" if st is not None and st == st else f"{'--':>8s}"
    for v in ("5", "14"):
        if cols[v] is None:
            line += f" {'FAIL':>8s} {'--':>5s}"
            losses[v] += 1
            continue
        t = per[(v, cols[v])]["jm"]
        vd = verdicts(t, st)
        line += f" {t:8.4f} {cols[v]:>5s}"
        wins[v] += vd == "WIN"
        ties[v] += vd == "TIE(2%)"
        losses[v] += vd == "LOSS"
    vd_best = verdicts(jm_best, st)
    bwins += vd_best == "WIN"
    bties += vd_best == "TIE(2%)"
    blosses += vd_best in ("LOSS", "no-stock-row")
    print(line + f" {jm_best:8.4f} {best_v + '/' + bs:>9s} "
          f"{gbs:6.0f} {fpct:6.1f}% {vd_best}")

print(f"\nper-variant cells (best-of-SGF):")
for v in ("5", "14"):
    print(f"  jm{v}: {wins[v]} WIN / {ties[v]} TIE / {losses[v]} LOSS")
print(f"best-variant cells: {bwins} WIN / {bties} TIE / {blosses} LOSS  "
      f"(stop condition: 0 LOSS or LOSS-with-banked-math)")
allrows = [r for r in rows if r["sgf"] == "2048"]
if allrows:
    import statistics
    for v in ("5", "14"):
        for lane in ("w16pt", "w16pc", "m816", "g16"):
            rs = [r for r in allrows
                  if r["lane"] == lane and r.get("var", "5") == v]
            if rs:
                med = statistics.median(r["jm_vs_st"] for r in rs)
                print(f"  var {v} lane {lane}: median jm/st (SGF 2048) = "
                      f"{med:.2f}x over {len(rs)} cells")
