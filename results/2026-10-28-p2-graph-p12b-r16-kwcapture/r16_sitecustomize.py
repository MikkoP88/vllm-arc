# r16 capture hook v6 — PYTHONPATH-injected sitecustomize.
# Wraps QwenGatedDeltaNetAttention.forward_xpu (class attr only, r12b law)
# and stages the REAL first-request layer-0 call to CPU for standalone
# replay. v5 lesson: by save time the default stream's queue is in
# command-graph capture state at the SYCL level, so torch.save's blocking
# D2H (an event wait on the current stream) raises "wait method cannot be
# used for an event associated with a command graph". v6 therefore stages
# EVERYTHING on a private side stream (stream synchronize = queue wait, no
# event object) and writes two files per call:
#   cap_<pid>_<idx>e.pt  entry blob  (BEFORE orig: hidden, module weights,
#                                     proj weights, pool geometry) — the
#                                     stream is clean here (flip window is
#                                     inside orig per r16v5 onset stack)
#   cap_<pid>_<idx>p.pt  post blob   (AFTER orig: step GDN metadata, pool
#                                     rows at referenced ids, recomputed
#                                     qkvz/ba projections)
# v5 evidence baked into design: onset stack = thread inside in_proj_ba
# (qwen_gdn_linear_attn.py:992 plain F.linear) BEFORE the SYCL op of this
# step; retry on a fresh stream flipped again (runtime-wide state).
import os
import sys
import threading
import time

_CAP_DIR = os.environ.get("R16_CAP_DIR", "")
_MAX = int(os.environ.get("R16_CAP_MAX", "8"))
_state = {"count": 0, "installed": False}
_TARGET = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"


def _stage(fn):
    """Run staging on a private stream; host-wait via stream sync (never an
    event on the caller's — possibly capture-poisoned — current stream)."""
    import torch
    side = torch.xpu.Stream()
    with torch.xpu.stream(side):
        out = fn()
    side.synchronize()
    return out


def _cpu(t):
    return t.detach().to("cpu", copy=True) if t is not None else None


def _mod_state_cpu(self, torch):
    d = {}
    for name in ("num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim",
                 "activation", "tp_size", "gqa_interleaved_layout"):
        d[name] = getattr(self, name, None)
    tensors = {}
    c = getattr(self, "conv1d", None)
    if c is not None:
        tensors["conv_weight"] = c.weight
        if getattr(c, "bias", None) is not None:
            tensors["conv_bias"] = c.bias
    for name in ("A_log", "dt_bias"):
        t = getattr(self, name, None)
        if torch.is_tensor(t):
            tensors[name] = t
    for pname in ("in_proj_qkvz", "in_proj_ba"):
        m = getattr(self, pname, None)
        if m is not None:
            for wname in ("weight", "bias"):
                w = getattr(m, wname, None)
                if torch.is_tensor(w):
                    tensors[f"{pname}.{wname}"] = w
    kv = getattr(self, "kv_cache", None)
    pools = None
    if isinstance(kv, (list, tuple)) and len(kv) >= 2 and torch.is_tensor(kv[0]):
        pools = (kv[0], kv[1])

    def grab():
        out = {k: _cpu(v) for k, v in tensors.items()}
        if pools is not None:
            for key, p in (("conv_pool", pools[0]), ("ssm_pool", pools[1])):
                out[f"{key}_shape"] = tuple(p.shape)
                out[f"{key}_stride"] = tuple(p.stride())
        return out

    d.update(_stage(grab))
    d["pools_bound"] = pools is not None
    return d


def _pool_rows_cpu(self, torch, ids_dev, nrows):
    kv = getattr(self, "kv_cache", None)
    if not (isinstance(kv, (list, tuple)) and len(kv) >= 2
            and torch.is_tensor(kv[0])):
        return None
    if ids_dev is None:
        ids_dev = torch.arange(min(64, kv[0].shape[0]), device=kv[0].device)

    def grab():
        return {"conv_pool_rows": _cpu(kv[0][ids_dev]),
                "ssm_pool_rows": _cpu(kv[1][ids_dev]),
                "row_ids": _cpu(ids_dev)}

    try:
        return _stage(grab)
    except Exception as e:  # noqa: BLE001
        return {"err": repr(e)[:200]}


def _meta_dev(torch):
    """Read the step's GDN metadata tensors (materialized by op time, not at
    forward_xpu entry — v1 lesson). Returns (scalars, device tensors)."""
    out = {}
    try:
        from vllm.forward_context import get_forward_context
        amd_raw = getattr(get_forward_context(), "attn_metadata", None)
        out["_amd_type"] = type(amd_raw).__name__
        if isinstance(amd_raw, dict):
            out["_amd_len"] = len(amd_raw)
            for _k, amd in amd_raw.items():
                for f in ("num_prefills", "num_decodes", "num_spec_decodes",
                          "num_actual_tokens", "num_accepted_tokens",
                          "has_initial_state", "non_spec_query_start_loc",
                          "non_spec_token_indx",
                          "non_spec_state_indices_tensor",
                          "spec_query_start_loc", "spec_token_indx",
                          "spec_state_indices_tensor", "spec_sequence_masks"):
                    v = getattr(amd, f, None)
                    out[f] = v if not torch.is_tensor(v) else ("DEV", v)
                break  # one layer's row: same step for every GDN layer
    except Exception as e:  # noqa: BLE001
        out["err"] = repr(e)[:200]
    return out


def _meta_cpu(meta_dev):
    scal = {k: v for k, v in meta_dev.items() if not isinstance(v, tuple)}
    tens = {k: v[1] for k, v in meta_dev.items() if isinstance(v, tuple)}

    def grab():
        return {k: (_cpu(v) if torch_like(v) else v) for k, v in tens.items()}

    try:
        staged = _stage(grab)
    except Exception as e:  # noqa: BLE001
        staged = {"stage_err": repr(e)[:200]}
    out = dict(scal)
    out.update(staged)
    return out


def torch_like(v):
    import torch
    return torch.is_tensor(v)


def _proj_cpu(self, torch, hidden_states):
    """Recompute the op's projection args (same input + weights,
    deterministic) staged to CPU — exact qkvz/ba values for replay."""

    def grab():
        qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        return {"qkvz": _cpu(qkvz), "ba": _cpu(ba)}

    try:
        return _stage(grab)
    except Exception as e:  # noqa: BLE001
        return {"err": repr(e)[:200]}


def _write(blob, pid, idx, tag):
    try:
        os.makedirs(_CAP_DIR, exist_ok=True)
        path = os.path.join(_CAP_DIR, f"cap_{pid}_{idx:02d}{tag}.pt")
        torch_save_cpu(blob, path)
        print(f"[r16cap] saved {path} info={blob.get('info')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[r16cap] SAVE-FAIL{tag} {e!r}", flush=True)


def torch_save_cpu(blob, path):
    import torch
    torch.save(blob, path)


def _try_wrap():
    if _state["installed"] or not _CAP_DIR:
        return True
    mod = sys.modules.get(_TARGET)
    if mod is None:
        return False
    cls = getattr(mod, "QwenGatedDeltaNetAttention", None)
    if cls is None or getattr(cls, "_r16_wrapped", False):
        return True

    # v7 preamble probes: bracket the flip op. pre_rec at embedding
    # entry/exit and around every TP allreduce call (the preamble before
    # layer-0's GDN forward: embed gemm + AR + layernorm).
    def _rec(torch):
        try:
            return torch.xpu.is_current_stream_capturing()
        except Exception:  # noqa: BLE001
            return None

    try:
        import vllm.distributed.communication_op as _co
        _ar = _co.tensor_model_parallel_all_reduce

        def ar_probe(input_):
            import torch
            n = input_.numel() * input_.element_size()
            pre = _rec(torch)
            if pre is None or n < 512 * 5120 * 2:
                return _ar(input_)
            print(f"[r16pre] AR entry numel={tuple(input_.shape)} "
                  f"bytes={n} pre_rec={pre}", flush=True)
            out = _ar(input_)
            print(f"[r16pre] AR exit  pre_rec={_rec(torch)}", flush=True)
            return out

        if not getattr(_co.tensor_model_parallel_all_reduce, "_r16", False):
            ar_probe._r16 = True
            _co.tensor_model_parallel_all_reduce = ar_probe
            # layers bind the name at import time into their own module
            # namespaces — sweep them so the probe actually fires
            n = 0
            for _name, _m in list(sys.modules.items()):
                try:
                    if getattr(_m, "tensor_model_parallel_all_reduce",
                               None) is _ar:
                        setattr(_m, "tensor_model_parallel_all_reduce",
                                ar_probe)
                        n += 1
                except Exception:  # noqa: BLE001
                    pass
            print(f"[r16pre] allreduce probe installed "
                  f"(+{n} importer namespaces)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[r16pre] AR probe fail: {e!r}", flush=True)

    try:
        from vllm.model_executor.layers import vocab_parallel_embedding as _ve
        _V = _ve.VocabParallelEmbedding
        _orig_emb = _V.forward

        def emb_probe(self, input_):
            import torch
            n = int(input_.numel())
            if n < 512:
                return _orig_emb(self, input_)
            print(f"[r16pre] EMB entry tok={n} pre_rec={_rec(torch)}",
                  flush=True)
            out = _orig_emb(self, input_)
            print(f"[r16pre] EMB exit  pre_rec={_rec(torch)}", flush=True)
            return out

        if not getattr(_V.forward, "_r16", False):
            emb_probe._r16 = True
            _V.forward = emb_probe
            print("[r16pre] embedding probe installed", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[r16pre] EMB probe fail: {e!r}", flush=True)

    orig = cls.forward_xpu

    def wrapped(self, hidden_states):
        import torch
        idx = _state["count"]
        tok = hidden_states.size(0)
        info = {"num_tokens": int(tok)}
        pre_rec = None
        try:
            pre_rec = torch.xpu.is_current_stream_capturing()
        except Exception:  # noqa: BLE001
            pass
        is_l0 = self.prefix.endswith("layers.0.linear_attn")
        save = is_l0 and tok > 512 and idx < _MAX
        if save:
            _state["count"] = idx + 1
            try:
                amd_entry = getattr(__import__(
                    "vllm.forward_context", fromlist=["get_forward_context"]
                ).get_forward_context(), "attn_metadata", None)
                info["meta_entry"] = (type(amd_entry).__name__
                                      if not isinstance(amd_entry, dict)
                                      else f"dict[{len(amd_entry)}]")
            except Exception as e:  # noqa: BLE001
                info["meta_entry"] = repr(e)[:120]
            kv = getattr(self, "kv_cache", None)
            info["pools"] = isinstance(kv, (list, tuple)) and len(kv) >= 2 \
                and torch.is_tensor(kv[0])
            # ENTRY print: survives even if every later stage burns (v5
            # lesson — the save itself died on the poisoned stream).
            print(f"[r16cap] CALL pid={os.getpid()} idx={idx} tok={tok} "
                  f"pre_rec={pre_rec} {info}", flush=True)
            entry = None
            try:
                entry = {"info": dict(info),
                         "hidden_states": _stage(
                             lambda: _cpu(hidden_states)),
                         "module": _mod_state_cpu(self, torch)}
                _write(entry, os.getpid(), idx, "e")
            except Exception as e:  # noqa: BLE001
                print(f"[r16cap] ENTRY-FAIL {e!r}", flush=True)
        try:
            out = orig(self, hidden_states)
        except Exception as e:  # noqa: BLE001
            info["fwd_err"] = repr(e)[:300]
            if save:
                _post(self, torch, hidden_states, info, idx)
            raise
        if save:
            _post(self, torch, hidden_states, info, idx)
        return out

    def _post(self, torch, hidden_states, info, idx):
        try:
            post_rec = None
            try:
                post_rec = torch.xpu.is_current_stream_capturing()
            except Exception:  # noqa: BLE001
                pass
            info["post_recording"] = post_rec
            info["t"] = time.time()
            info["prefix"] = self.prefix
            meta_dev = _meta_dev(torch)
            ids = None
            for k in ("non_spec_state_indices_tensor",
                      "spec_state_indices_tensor"):
                v = meta_dev.get(k)
                if isinstance(v, tuple):
                    ids = v[1]
                    break
            blob = {"info": dict(info), "meta": _meta_cpu(meta_dev)}
            blob.update(_pool_rows_cpu(self, torch, ids, None) or {})
            blob.update(_proj_cpu(self, torch, hidden_states))
            _write(blob, os.getpid(), idx, "p")
        except Exception as e:  # noqa: BLE001
            print(f"[r16cap] POST-FAIL {e!r}", flush=True)

    wrapped._r16_wrapped = True
    cls.forward_xpu = wrapped
    _state["installed"] = True
    print(f"[r16cap] forward_xpu wrapped v6 (dir={_CAP_DIR} max={_MAX} "
          f"pid={os.getpid()})", flush=True)
    return True


def _poll():
    for _ in range(3600):
        if _try_wrap():
            return
        time.sleep(1.0)


class _WrapOnImport:
    """Fork-safe injection: vllm v1 workers are FORKED from EngineCore
    BEFORE the model import — poll threads die at fork, so the hook must be
    plain process state that fires at import time in any process."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            import importlib.util
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _try_wrap()
            except Exception as e:  # noqa: BLE001
                print(f"[r16cap] wrap-after-import failed: {e!r}", flush=True)

        loader.exec_module = exec_module
        return spec


if _CAP_DIR:
    sys.meta_path.insert(0, _WrapOnImport())
    threading.Thread(target=_poll, daemon=True).start()
