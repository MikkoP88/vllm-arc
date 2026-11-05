#!/usr/bin/env node
// trace_ar_fold.js — P6.22 r94b: fold a kineto XPU rank trace into the
// AR-family device-time table, external-id-linking c10d::allreduce_
// CPU ops to their kernels and merging call ORDER with the ARC_AR_TRACE
// census (per-pid) to attribute device ms per message size.
//
// Usage: node trace_ar_fold.js rank0.trace.json.gz census.jsonl [--json out]
// Method provenance: r67b (op-class capture, external-id verification).
'use strict';
const fs = require('fs');
const zlib = require('zlib');

const [, , traceGz, censusPath, ...rest] = process.argv;
if (!traceGz || !censusPath) {
  console.error('usage: node trace_ar_fold.js TRACE.json.gz CENSUS.jsonl');
  process.exit(1);
}
const jsonOut = rest.includes('--json')
  ? rest[rest.indexOf('--json') + 1] : null;

// r67b kernel classes (regex over kineto kernel names)
const CLASSES = [
  ['AR_ring', /allreduce_large_su_ring_write_multi_kernel/],
  ['AR_reducecopy', /ReduceCopyKernel/],
  ['AR_barrier', /oneccl_invoke_barrier/],
  ['AR_misc_ccl', /allreduce|allgather|oneccl|ccl/i],
  ['GEMM', /gemm_kernel/],
  ['FA', /compat::experimental::detail::KernelFunctor/],
  ['GDN', /gdn::/],
  ['norm', /fused_add_rms_norm|layer_norm|multirow_norm/],
  ['act', /act_and_mul/],
  ['elementwise', /VectorizedElementwise|elementwise|CatArrayBatched|copy_|direct_copy|_to_copy|clone/i],
  ['quant', /quant|fp8/i],
  ['memcpy_memset', /memcpy|memset/],
];

const buf = zlib.gunzipSync(fs.readFileSync(traceGz));
const trace = JSON.parse(buf.toString('utf8'));
const ev = trace.traceEvents;
console.error(`events=${ev.length}`);

// 1) per-class device-kernel totals
const clsTot = new Map([...CLASSES.map(([n]) => [n, {t: 0, n: 0}]),
  ['other', {t: 0, n: 0}]]);
let kernelTotal = 0;
const kernels = []; // {ts, dur, name, eid, pid}
for (const e of ev) {
  if (e.ph !== 'X' || e.cat !== 'kernel') continue;
  kernelTotal += e.dur;
  let hit = 'other';
  for (const [name, re] of CLASSES) {
    if (re.test(e.name)) { hit = name; break; }
  }
  const c = clsTot.get(hit);
  c.t += e.dur; c.n += 1;
  if (hit.startsWith('AR_')) kernels.push(
    { ts: e.ts, dur: e.dur, name: e.name, eid: e.args?.['External id'], pid: e.pid });
}
console.log(`kernel total = ${(kernelTotal / 1e6).toFixed(3)} s`);
for (const [name, c] of clsTot) {
  if (c.n) console.log(
    `  ${name.padEnd(14)} ${(c.t / 1e6).toFixed(3)} s  x${c.n}`);
}

// 2) c10d::allreduce_ cpu ops, per pid, in ts order; sum linked kernels
const opsByPid = new Map(); // pid -> [{ts, dur, eid}]
for (const e of ev) {
  if (e.ph !== 'X' || e.cat !== 'cpu_op') continue;
  if (!/c10d::allreduce_?$/.test(e.name)) continue;
  if (!opsByPid.has(e.pid)) opsByPid.set(e.pid, []);
  opsByPid.get(e.pid).push(
    { ts: e.ts, dur: e.dur, eid: e.args?.['External id'] });
}
const kByEid = new Map(); // eid -> {t, n}
for (const k of kernels) {
  if (k.eid === undefined) continue;
  const s = kByEid.get(k.eid) || { t: 0, n: 0 };
  s.t += k.dur; s.n += 1;
  kByEid.set(k.eid, s);
}

// 3) census per pid, all_reduce in ts order; merge by ORDER within pid
const census = fs.readFileSync(censusPath, 'utf8').split('\n')
  .filter((l) => l.trim())
  .map((l) => { try { return JSON.parse(l); } catch { return null; } })
  .filter((r) => r && r.op === 'all_reduce' && r.numel);
const censusByPid = new Map();
for (const r of census) {
  if (!censusByPid.has(r.pid)) censusByPid.set(r.pid, []);
  censusByPid.get(r.pid).push(r);
}
const perSize = new Map(); // bytes -> {n, devTot, devRing, devRed}
let linked = 0, unlinked = 0;
for (const [pid, ops] of opsByPid) {
  const cens = censusByPid.get(pid) || [];
  ops.sort((a, b) => a.ts - b.ts);
  cens.sort((a, b) => a.ts - b.ts);
  for (let i = 0; i < ops.length; i++) {
    const o = ops[i];
    const k = o.eid !== undefined ? kByEid.get(o.eid) : undefined;
    const c = cens[i];
    if (!k) { unlinked++; continue; }
    linked++;
    const key = c ? (c.bytes || 0) : -1;
    const s = perSize.get(key) || { n: 0, devTot: 0 };
    s.n += 1; s.devTot += k.t;
    perSize.set(key, s);
  }
}
console.log(`allreduce ops linked=${linked} unlinked=${unlinked} ` +
  `(order-merge assumes equal call sequences per pid)`);
const rows = [...perSize.entries()].sort((a, b) => b[1].devTot - a[1].devTot);
console.log('per-size device time (top 15 by total):');
console.log('  bytes        MiB      n   devTot_ms  dev_mean_ms');
for (const [b, s] of rows.slice(0, 15)) {
  console.log(`  ${String(b).padStart(10)} ${(b / 1048576).toFixed(2).padStart(8)} ` +
    `${String(s.n).padStart(4)} ${(s.devTot / 1e3).toFixed(1).padStart(9)}` +
    `${(s.devTot / s.n / 1e3).toFixed(3).padStart(11)}`);
}
const arTot = rows.reduce((a, [, s]) => a + s.devTot, 0);
console.log(`AR-family device total (linked) = ${(arTot / 1e6).toFixed(3)} s`);
if (jsonOut) {
  fs.writeFileSync(jsonOut, JSON.stringify({
    kernel_total_ms: kernelTotal / 1e3,
    classes: Object.fromEntries([...clsTot].map(
      ([n, c]) => [n, { ms: c.t / 1e3, n: c.n }])),
    per_size: rows.map(([bytes, s]) => ({
      bytes, n: s.n, dev_tot_ms: s.devTot / 1e3,
      dev_mean_ms: s.devTot / s.n / 1e3 })),
    linked, unlinked,
  }, null, 1));
  console.log(`wrote ${jsonOut}`);
}
