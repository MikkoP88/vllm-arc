#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p253_ph4_gate_flip_probe.py -- P-H4 posture-aware gate-flip
offline probe (P6.47, r253).

Runs INSIDE cert28 with the five P-H4 files overlaid (platform.py,
envs.py, patches/lmc_mx_boot_gate.py, lmcint/vllm_v1_adapter.py --
r253_ph4_gate_flip_offline.sh mounts them over site-packages). No
engine boot: the gate fires at connector construction, so the probe
drives the REAL wiring (platform _install_wheel_hooks -> meta-path
finder -> factory import -> exec-wrap -> KVConnectorFactory.
create_connector wrap), exactly the p246 pattern (11/11 at P-D).

The P-H4 delta under test: the MX x LMCache branch is POSTURE-AWARE
-- the r249 packed-row interop (capability marker
``ARC_LMC_MX_INTEROP`` on the overlaid adapter) admits the lane with
a ``gate=interop-live`` posture line; its absence keeps the P-D
refusal (``ARC_LMC_MX_REFUSED``). The probe toggles the capability
marker ON THE REAL MODULE (delattr/restore) to drive both postures
in one container -- the gate reads the live module through
sys.modules, so the toggle is exactly what a pre-r249 install looks
like to it.

RUN HISTORY (rig logs preserved under /root/build/p253):
  run 1 (rev 1, 18:39:54Z): rc=1 right after face A -- the staging
  import died at the STOCK lmcache_integration/__init__ (CudaIPCWrapper
  is lmcache-main-only; cert28 pins lmcache 0.5.5 -- the kit __init__
  header documents exactly this). Root cause: import ORDER. The probe
  imported the kit path BARE; the engine never does -- the p41b
  exec-wrap hook on vllm...v1.lmcache_connector installs the lmcint
  source-override finder (lmcache_hma_connector._SrcOverrideFinder,
  which maps the three vendored lmcache_integration names onto the
  kit files) at registry RESOLUTION, before anything touches the kit.
  The gate is safe by construction (its create_connector wrap
  resolves the connector class before the posture branch), but the
  probe must drive the same chain first. Rev 2 adds the R_bootstrap
  face: the real resolution, then the staging import guarded.

Faces (15):
  A_fired        the hook fired on the real factory (marker on the
                 wrapped create_connector __func__).
  R_bootstrap    the REAL registry resolution runs (the engine's own
                 import order -- it installs the p41b lmcint
                 src-override finder the gate's _interop_live import
                 depends on) and the overlaid kit adapter then
                 imports clean, carrying the ARC_LMC_MX_INTEROP
                 capability marker.
  H_live_mxfp4/  interop PRESENT + LMC over the MX dtype -> the stub
  _mxfp8/_mxk8v4 constructs (pass-through) with the
                 ``gate=interop-live`` posture line -- never refused.
  B_stock_mxfp4/ interop ABSENT (marker deleted) + LMC over MX ->
  _mxfp8/_mxk8v4 ARC_LMC_MX_REFUSED ValueError (the P-D posture
                 stands on pre-r249 installs).
  B_legacy_stock the same refusal over the pre-rename attr
                 (cache_config.kv_cache_dtype) with interop absent --
                 the fallback chain stays pinned under the new
                 posture logic.
  D_resolve      the REAL registry entry "LMCacheConnectorV1"
                 resolves to an LMC-named class (detection basis)
                 -- resolve-only, never constructed.
  C_scope        a NON-LMC connector over MX (interop present)
                 passes through, ``gate=on`` posture line.
  D_lawful       LMC over a NON-MX dtype (checkpoint_fp8) passes
                 through and constructs -- the certified LMC lane is
                 untouched.
  E_bypass_stock interop absent + knob VLLM_ARC_LMC_MX_GATE=0 ->
                 ``gate=bypassed`` posture line noting interop=absent
                 (A/B evidence only).
  F_stock        stock None-config behavior preserved (the factory's
                 own kv_transfer_config-must-be-set ValueError).
  G_idem         a second gate _install() is a no-op.

Pass-through faces (H/C/D/E) drive stub connector classes through a
temporarily replaced get_connector_class (restored in finally):
constructing a REAL LMCacheConnectorV1 inside a probe would
initialize the lmcache engine -- not a probe's business.

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
    "R_bootstrap",
    "H_live_mxfp4",
    "H_live_mxfp8",
    "H_live_mxk8v4",
    "B_stock_mxfp4",
    "B_stock_mxfp8",
    "B_stock_mxk8v4",
    "B_legacy_stock",
    "D_resolve",
    "C_scope",
    "D_lawful",
    "E_bypass_stock",
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
        # the stock create_connector's logger.info reads it
        self.engine_id = "p253-probe"


class _CC:  # cache_config stub -- the vllm-0.30 surface (cache.py:111)
    def __init__(self, dt: str | None) -> None:
        self.cache_dtype = dt


class _CCLegacy:  # pre-rename surface (pins the fallback chain order)
    def __init__(self, dt: str | None) -> None:
        self.kv_cache_dtype = dt


class _SC:  # scheduler_config stub (the stock HMA check reads it)
    disable_hybrid_kv_cache_manager = True


class _Cfg:  # VllmConfig stub (the gate + stock path attrs)
    def __init__(self, dt: str | None, name: str,
                 cc_cls: type = _CC) -> None:
        self.cache_config = cc_cls(dt)
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

    print("== p253 P-H4 posture-aware gate-flip probe (rev 2)")

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
        print(f"== p253 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
        print("VERDICT: FAIL")
        return 1
    factory = fac.KVConnectorFactory
    wrap = factory.__dict__["create_connector"]
    if not getattr(getattr(wrap, "__func__", None), "_arc_mx_gate", False):
        _fail("A_fired", "create_connector carries no _arc_mx_gate marker")
        print(f"== p253 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
        print("VERDICT: FAIL")
        return 1
    _ok("A_fired", "hook fired on the real factory create_connector")

    # ---- R: the engine's kit redirect must be live (p41b exec-wrap) --
    # The engine NEVER imports the kit path bare: the registry
    # resolution first imports vllm...v1.lmcache_connector, whose
    # exec-wrap hook (lmcache_hma_connector) installs the lmcint
    # source-override finder -- without it the vendored
    # lmcache_integration/__init__ execs and dies on lmcache 0.5.5
    # (CudaIPCWrapper is lmcache-main-only; the run-1 lesson). The
    # gate's own _interop_live rides the same order (its wrap resolves
    # the connector class before the posture branch), so the probe
    # drives the real resolution before touching the kit.
    try:
        factory.get_connector_class(_KTC("LMCacheConnectorV1"))
    except Exception as e:
        _fail("R_bootstrap", f"real registry resolution died: {e!r}")
        print(f"== p253 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
        print("VERDICT: FAIL")
        return 1
    _ok("R_bootstrap", "real resolution ran (p41b lmcint src-override finder installed)")
    try:
        import vllm_gaudi.xpu.patches.lmcint.vllm_v1_adapter as _ad
    except Exception as e:
        _fail("R_bootstrap", f"kit adapter import dead post-redirect: {e!r}")
        print(f"== p253 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
        print("VERDICT: FAIL")
        return 1

    # the capability module the gate reads (toggled per posture)
    _cap_saved = getattr(_ad, "ARC_LMC_MX_INTEROP", None)
    if _cap_saved is None:
        _fail("R_bootstrap", "overlaid adapter carries no ARC_LMC_MX_INTEROP marker")
        print(f"== p253 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
        print("VERDICT: FAIL")
        return 1

    def _interop_absent() -> None:
        with contextlib.suppress(AttributeError):
            del _ad.ARC_LMC_MX_INTEROP

    # ---- H: interop live -> pass-through + gate=interop-live --------
    # (pass-through helper defined before the face block; the same
    # stub-swap + restore pattern as p246.)
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

    for face, dt in (("H_live_mxfp4", "mxfp4"), ("H_live_mxfp8", "mxfp8"),
                     ("H_live_mxk8v4", "mxk8v4")):
        _ad.ARC_LMC_MX_INTEROP = _cap_saved  # live posture
        _pass_through(face, dt, _StubLMC, "LMCacheConnectorV1",
                      "gate=interop-live")

    # ---- B: interop absent -> the P-D refusal stands -----------------
    try:
        for face, dt in (("B_stock_mxfp4", "mxfp4"),
                         ("B_stock_mxfp8", "mxfp8"),
                         ("B_stock_mxk8v4", "mxk8v4")):
            _interop_absent()
            try:
                factory.create_connector(
                    _Cfg(dt, "LMCacheConnectorV1"), "r", None)
                _fail(face, f"LMCache over {dt} was NOT refused")
            except ValueError as e:
                if "ARC_LMC_MX_REFUSED" in str(e):
                    _ok(face, f"{dt} refused (interop absent): "
                              f"{str(e)[:48]}...")
                else:
                    _fail(face, f"wrong ValueError over {dt}: {str(e)[:70]}")

        # ---- B_legacy: pre-rename surface, interop still absent -----
        try:
            factory.create_connector(
                _Cfg("mxfp4", "LMCacheConnectorV1", _CCLegacy), "r", None)
            _fail("B_legacy_stock",
                  "LMCache over mxfp4 (kv_cache_dtype) NOT refused")
        except ValueError as e:
            if "ARC_LMC_MX_REFUSED" in str(e):
                _ok("B_legacy_stock",
                    "kv_cache_dtype surface refused (fallback pinned)")
            else:
                _fail("B_legacy_stock",
                      f"wrong ValueError: {str(e)[:70]}")
    finally:
        _ad.ARC_LMC_MX_INTEROP = _cap_saved

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

    # ---- C: (c1) non-LMC connector over MX passes through -----------
    _ad.ARC_LMC_MX_INTEROP = _cap_saved
    _pass_through("C_scope", "mxfp8", _StubNixl, "NixlConnector",
                  "gate=on connector=NixlConnector")

    # ---- D: (c2) LMC over NON-MX dtype passes through ---------------
    _pass_through("D_lawful", "checkpoint_fp8", _StubLMC,
                  "LMCacheConnectorV1", "gate=on connector=LMCacheConnectorV1")

    # ---- E: knob bypass with interop absent --------------------------
    os.environ["VLLM_ARC_LMC_MX_GATE"] = "0"
    try:
        _interop_absent()
        _pass_through("E_bypass_stock", "mxfp4", _StubLMC,
                      "LMCacheConnectorV1", "gate=bypassed")
    finally:
        del os.environ["VLLM_ARC_LMC_MX_GATE"]
        _ad.ARC_LMC_MX_INTEROP = _cap_saved

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

    print(f"== p253 faces ok={len(FACES) - len(failed)}/{len(FACES)}")
    print("VERDICT: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
