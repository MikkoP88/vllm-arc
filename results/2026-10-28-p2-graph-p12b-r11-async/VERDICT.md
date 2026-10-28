# VERDICT — r11: --async-scheduling exonerated — identical flip at first-completion; serve-level champion deltas exhausted, r12 tracer next
image:  vllm-arc:v0.30.0-b70-p12b f790fb4adee0 (fork b8bf1431)
env:    ARC_LEG=graph-async GRAPH=1 SPEC=0 ARC_ASYNC=1 (flag confirmed in
        config dump — AsyncScheduler active)
result: FLIP AGAIN, identical signature: first completions 200 OK,
        onset 05:49:53 in the completion window, then 500s + death.
        ASYNC-SCHEDULING EXONERATED.
state:  serve-level champion deltas all eliminated: capture stream (r8/r9),
        ICL=0 (r10), async-scheduling (r11), prefix (earlier). Remaining
        deltas are structural: vllm 0.30 step/completion machinery vs
        0.26.1, torch 2.13 vs 2.12 binary, oneAPI 2026 vs 2025.3.
next:   r12 tracer build — env-gated ARC_TRACE_FLIP: 10ms watchdog + all-
        thread stack dump at onset + capture-state logging at every
        hygiene boundary; names the python frame executing when the
        default first reads recording.
