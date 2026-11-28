# r266 gemma issue-analysis fact table

## b3_g4_29
```
leg=b3_g4_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g4 chat_cells=all g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/r266/chat_template_gemma4_official.jinja ARC_RPARSER=gemma4 ARC_TPARSER=gemma4]
skin: g4(official-it) rp='gemma4' tp='gemma4'
weights: fp8  kv: fp8_e4m3
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0
probe: served=g4 oracle_match=True ptok_mismatch=0
chat t0: ok=60/81 reploop=18 empty=0 echo=33 lp1=-1.576 lp_gen=-0.295
chat card: ok=41/81 reploop=5 empty=0 echo=12 lp1=-2.969 lp_gen=-0.979
raw served-fmt t0: ok=15/27 ptok_ok=27/27
gate battery: http=200 'What is the capital of France? Answer with just the city name.\nWhat is'
gate battery: http=200 'Write a Python function that returns the n-th Fibonacci number using i'
gate battery: http=200 'hmannthought\nhmannthought\nhmannthought\nhmannthought\nhmannthought\nhmann'
gate sanity0: http=200 'Reply with exactly: HEALTH CHECK OK\nReply with exactly: HEALTH CHECK O'
gate sanity1: http=200 'Reply with exactly: HEALTH CHECK OK\nReply with exactly: HEALTH CHECK O'
```
## b5_g3_19
```
leg=b5_g3_19 image=vllm-arc:v0.30.0-b70-cert19 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER=]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: fp8_e4m3
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
chat t0: ok=48/54 reploop=0 empty=0 echo=36 lp1=-0.678 lp_gen=-0.102
chat card: ok=52/54 reploop=0 empty=0 echo=6 lp1=-1.672 lp_gen=-0.423
raw served-fmt t0: ok=24/27 ptok_ok=27/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 'def fibonacci(n): if n <= 0: return "Input should be a positive intege'
gate battery: http=200 'Three chemical elements that are gases at room temperature are:\n1. Hyd'
gate sanity0: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
```
## b4_g3_20
```
leg=b4_g3_20 image=vllm-arc:v0.30.0-b70-cert20 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER=]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: fp8_e4m3
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
chat t0: ok=48/54 reploop=0 empty=0 echo=36 lp1=-0.678 lp_gen=-0.102
chat card: ok=52/54 reploop=0 empty=0 echo=6 lp1=-1.671 lp_gen=-0.422
raw served-fmt t0: ok=24/27 ptok_ok=27/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 'def fibonacci(n): if n <= 0: return "Input should be a positive intege'
gate battery: http=200 'Three chemical elements that are gases at room temperature are:\n1. Hyd'
gate sanity0: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
```
## c1_bf16_29
```
leg=c1_bf16_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=0 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_DTYPE=bfloat16]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: fp8_e4m3
engine: err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
chat t0: ok=48/54 reploop=0 empty=0 echo=36 lp1=-0.679 lp_gen=-0.102
chat card: ok=49/54 reploop=0 empty=0 echo=7 lp1=-1.755 lp_gen=-0.424
raw served-fmt t0: ok=24/27 ptok_ok=27/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 'def fibonacci(n): if n <= 0: return "Input should be a positive intege'
gate battery: http=200 'Three chemical elements that are gases at room temperature are:\n1. Hyd'
gate sanity0: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
```
## c2_fa2off_29
```
leg=c2_fa2off_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=0 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_ATTN_FA2=0]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: fp8_e4m3
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
chat t0: ok=48/54 reploop=0 empty=0 echo=36 lp1=-0.692 lp_gen=-0.103
chat card: ok=52/54 reploop=0 empty=0 echo=6 lp1=-1.668 lp_gen=-0.442
raw served-fmt t0: ok=24/27 ptok_ok=27/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 'def fibonacci(n): if n <= 0: return "Input should be a positive intege'
gate battery: http=200 'Three chemical elements that are gases at room temperature are:\n1. Hyd'
gate sanity0: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
```
## e2_m8g_eager_29
```
leg=e2_m8g_eager_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 ARC_GRAPH=0]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: mxfp8+native
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 mx_native=True post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=1 post_native_lines=18 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
chat card: ok=1/1 reploop=0 empty=0 echo=0 lp1=-2.606 lp_gen=-0.448
raw served-fmt t0: ok=0/27 ptok_ok=0/27
gate battery: http=200 'Paris our our our our our our our our our our our our our our our our '
gate battery: http=200 'def our our our our our our our our our our our our our our our our ou'
gate battery: http=200 'Three our our our our our our our our our our our our our our our our '
gate sanity0: http=200 'HEALTH our our our our our our our our our our our our our our our our'
gate sanity1: http=200 'HEALTH our our our our our our our our our our our our our our our our'
```
## e1_g3_eager_29
```
leg=e1_g3_eager_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_GRAPH=0]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: fp8_e4m3
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
chat t0: ok=48/54 reploop=0 empty=0 echo=36 lp1=-0.7 lp_gen=-0.103
chat card: ok=52/54 reploop=0 empty=0 echo=6 lp1=-1.677 lp_gen=-0.453
raw served-fmt t0: ok=24/27 ptok_ok=27/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 'def fibonacci(n): if n <= 0: return "Input should be a positive intege'
gate battery: http=200 'Three chemical elements that are gases at room temperature are:\n1. Hyd'
gate sanity0: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
```

## cross-leg discriminators

| leg | image | skin | weights | kv | oracle/ptok | chat_t0 ok | t0 flags | lp1 | nll |
|---|---|---|---|---|---|---|---|---|---|
| b3_g4_29 | v0.30.0-b70-cert29 | g4(official-it) rp='gemma4' tp='gemma4' | fp8 | fp8_e4m3 | True/ok | 60/81 | rp18 e0 | -1.576 | None |
| b5_g3_19 | v0.30.0-b70-cert19 | g3(P6.28) rp='' tp='' | fp8 | fp8_e4m3 | True/ok | 48/54 | rp0 e0 | -0.678 | None |
| b4_g3_20 | v0.30.0-b70-cert20 | g3(P6.28) rp='' tp='' | fp8 | fp8_e4m3 | True/ok | 48/54 | rp0 e0 | -0.678 | None |
| c1_bf16_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | fp8 | fp8_e4m3 | True/ok | 48/54 | rp0 e0 | -0.679 | None |
| c2_fa2off_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | fp8 | fp8_e4m3 | True/ok | 48/54 | rp0 e0 | -0.692 | None |
| e2_m8g_eager_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | fp8 | mxfp8+native | True/ok | 0/0 | rp0 e0 | None | None |
| e1_g3_eager_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | fp8 | fp8_e4m3 | True/ok | 48/54 | rp0 e0 | -0.7 | None |
