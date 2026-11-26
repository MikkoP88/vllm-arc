#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p246_lmc_gate_probe.py -- P-D LMC x MX boot-gate offline probe
(P6.47, r246).

Runs INSIDE the cert image with the three P-D files overlaid
(platform.py, envs.py, patches/lmc_mx_boot_gate.py -- r246_lmc_gate_
offline.sh mounts them over site-packages). No engine boot: the gate
fires at connector construction, so the probe drives the REAL wiring
(platform _install_wheel_hooks -> meta-path finder -> factory import
-> exec-wrap -> KVConnectorFactory.create_connector wrap).

RUN 1 (rev 1, 2026-11-26 13:10:59Z, preserved in the r246 rig logs):
the gate's core PROVEN on the real factory -- hook fired, refusal
green over all three MX dtypes, stock None-config intact, idempotence
green. 4 FAILs were probe-stub gaps, not gate defects: the stock
create_connector reads config.scheduler_config (the HMA check) and
the real get_connector_class reads kv_transfer_config.
kv_connector_module_path -- both missing from the rev-1 stubs. Rev 2
adds the two stub attrs (disable_hybrid_kv_cache_manager=True keeps
the stock HMA refuse off the pass-through path; module_path=None
routes the real registry loader).

  A_fired        the hook fired on the real factory (marker on the
                 wrapped create_connector __func__).
  B_refuse       (a) LMC-named connector over mxfp4 -> ValueError
                 carrying ARC_LMC_MX_REFUSED (the r241 posture law
                 made structural).
  B_mxfp8/_mxk8v4 the refusal holds for the other two MX dtypes.
  D_resolve      the REAL registry entry "LMCacheConnectorV1"
                 resolves to an LMC-named class (the detection basis)
                 -- resolve-only, never constructed.
  C_scope        (c1) a NON-LMC connector over MX passes through (no
                 evidence basis to refuse), posture line logged, the
                 stock constructor runs.
  D_lawful       (c2) LMC over a NON-MX dtype (checkpoint_fp8) passes
                 through and constructs -- the certified LMC lane is
                 untouched.
  E_bypass       knob VLLM_ARC_LMC_MX_GATE=0 bypasses the refusal
                 (A/B evidence only) with the gate=bypassed posture
                 line.
  F_stock        stock None-config behavior preserved (the factory's
                 own kv_transfer_config-must-be-set ValueError).
  G_idem         a second gate _install() is a no-op.

Pass-through faces (C/D/E) drive stub connector classes through a
temporarily replaced get_connector_class (restored in finally):
constructing a REAL LMCacheConnectorV1 inside a probe would initialize
the lmcache engine -- not a probe's business.

Exit 0 = all faces green; exit 1 = any face fails (each face prints
[face ok|FAIL] lines; VERDICT last).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import types

MX_DTYPES = ("mxfp4", "mxfp8", "mxk8v4")
FACES = (
    "A_fired",
    "B_refuse",
    "B_mxfp8",
    "B_mxk8v4",
    "D_resolve",
    "C_scope",
    "D_lawful",
    "E_bypass",
    "F_stock",
    "G_idem",
)


class _Stub:
    """Light stand-in connector class; records construction."""

    instances: list = []

    def __init__(self, config, role, kv_cache_config) -> None:  # noqa: D107
        self.config = config
        type(self).instances.append(self)


class _StubLMC(_Stub):
    __name__ = "LMCacheProbeStub"


class _StubNixl(_Stub):
    __name__ = "NixlProbeStub"


class _KTC:  # kv_transfer_config stub
    def __init__(self, name: str) -> None:
        self.kv_connector = name
        # the real get_connector_class reads this attr unconditionally
        self.kv_connector_module_path = None


class _CC:  # cache_config stub
    def __init__(self, dt: str | None) -> None:
        self.kv_cache_dtype = dt


class _SC:  # scheduler_config stub (the stock HMA check reads it)
    disable_hybrid_kv_cache_manager = True


class _Cfg:  # VllmConfig stub (the gate + stock path attrs)
    def __init__(self, dt: str | None, name: str) -> None:
        self.cache_config = _CC(dt)
        self.scheduler_config = _SC()
        self.kv_transfer_config = _KTC(name)


class _PostureCapture(logging.Handler):
    """Records INFO records emitted on the factory module's logger."""

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(level=logging.INFO)
        self.logger = logger
        self.records: list[str] = []

    def __enter__(self) -> _PostureCapture:
        self.records = []
        self.logger.addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        self.logger.removeHandler(self)

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            self.records.append(record.getMessage())


def main() -> int:  # noqa: C901 -- probe faces are one flat census
    failed: list[str] = []

    def _fail(face: str, msg: str) -> None:
        print(f"[face {face}] FAIL {msg}")
        failed.append(face)

    def _ok(face: str, msg: str) -> None:
        print(f"[face {face}] ok {msg}")

    print("== p246 LMC x MX boot-gate probe (rev 2)")

    # ---- A: platform import + hook install + factory wrap ----------
    try:
        import vllm_gaudi.xpu.platform as plat
        plat._install_wheel_hooks()
        fac = __import__(
            "vllm.distributed.kv_transfer.kv_connector.factory",
            fromlist=["KVConnectorFactory"],
        )
    except Exception as e:
        _fail("A_fired", f"platform/factory import died: {e!r}")
        print(f"== p246 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
        print("VERDICT: FAIL")
        return 1
    factory = fac.KVConnectorFactory
    wrap = factory.__dict__["create_connector"]
    if not getattr(getattr(wrap, "__func__", None), "_arc_mx_gate", False):
        _fail("A_fired", "create_connector carries no _arc_mx_gate marker")
        print(f"== p246 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
        print("VERDICT: FAIL")
        return 1
    _ok("A_fired", "hook fired on the real factory create_connector")

    # ---- B: refusal over all three MX dtypes -----------------------
    for face, dt in (("B_refuse", "mxfp4"), ("B_mxfp8", "mxfp8"),
                     ("B_mxk8v4", "mxk8v4")):
        try:
            factory.create_connector(_Cfg(dt, "LMCacheConnectorV1"), "r", None)
            _fail(face, f"LMCache over {dt} was NOT refused")
        except ValueError as e:
            if "ARC_LMC_MX_REFUSED" in str(e):
                _ok(face, f"{dt} refused: {str(e)[:52]}...")
            else:
                _fail(face, f"wrong ValueError over {dt}: {str(e)[:70]}")

    # ---- D_resolve: the REAL registry entry (detection basis) -------
    real_name = ""
    with contextlib.suppress(Exception):
        real_name = factory.get_connector_class(
            _KTC("LMCacheConnectorV1")).__name__
    if "lmcache" in real_name.lower():
        _ok("D_resolve",
            f"real registry resolves LMCacheConnectorV1 -> {real_name}")
    else:
        _fail("D_resolve", f"real resolution not LMC-named: {real_name!r}")

    # ---- pass-through helper (stub swap + restore) ------------------
    orig_get_cls = factory.__dict__["get_connector_class"].__func__

    def _pass_through(face: str, dt: str, stub_cls: type,
                      conn_name: str, expect: str) -> None:
        _Stub.instances = []
        factory.get_connector_class = classmethod(lambda cls, ktc: stub_cls)
        try:
            with _PostureCapture(fac.logger) as cap:
                ret = factory.create_connector(_Cfg(dt, conn_name), "r", None)
            lines = " | ".join(cap.records)
            if "ARC_LMC_MX_REFUSED" in lines:
                _fail(face, f"refused on the lawful path: {lines[:70]}")
            elif expect not in lines:
                _fail(face, f"posture line missing ({expect}): {lines[:70]}")
            elif ret not in _Stub.instances:
                _fail(face, "stock constructor did not run")
            else:
                _ok(face, f"pass-through + posture ({expect})")
        except Exception as e:
            _fail(face, f"unexpected {e!r}")
        finally:
            factory.get_connector_class = classmethod(orig_get_cls)

    # ---- C: (c1) non-LMC connector over MX passes through -----------
    _pass_through("C_scope", "mxfp8", _StubNixl, "NixlConnector",
                  "gate=on connector=NixlConnector")

    # ---- D: (c2) LMC over NON-MX dtype passes through ---------------
    _pass_through("D_lawful", "checkpoint_fp8", _StubLMC, "LMCacheConnectorV1",
                  "gate=on connector=LMCacheConnectorV1")

    # ---- E: knob bypass ----------------------------------------------
    os.environ["VLLM_ARC_LMC_MX_GATE"] = "0"
    try:
        _pass_through("E_bypass", "mxfp4", _StubLMC, "LMCacheConnectorV1",
                      "gate=bypassed")
    finally:
        del os.environ["VLLM_ARC_LMC_MX_GATE"]

    # ---- F: stock None-config behavior preserved ---------------------
    try:
        factory.create_connector(
            types.SimpleNamespace(kv_transfer_config=None), "r", None)
        _fail("F_stock", "None-config unexpectedly constructed")
    except ValueError as e:
        if "kv_transfer_config must be set" in str(e):
            _ok("F_stock", "stock None-config ValueError intact")
        else:
            _fail("F_stock", f"wrong None-config error: {str(e)[:70]}")

    # ---- G: idempotence ----------------------------------------------
    import vllm_gaudi.xpu.patches.lmc_mx_boot_gate as gate_mod
    gate_mod._install(fac)
    if getattr(factory.__dict__["create_connector"].__func__,
               "_arc_mx_gate", False):
        _ok("G_idem", "second _install kept exactly one wrapped constructor")
    else:
        _fail("G_idem", "second _install dropped the wrap")

    print(f"== p246 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
    print("VERDICT: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
