# r266 gemma issue-analysis fact table

## b2_g3_29
```
leg=b2_g3_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=all g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER=]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: fp8_e4m3
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
chat t0: ok=48/54 reploop=0 empty=0 echo=36 lp1=-0.678 lp_gen=-0.102
chat card: ok=52/54 reploop=0 empty=0 echo=6 lp1=-1.669 lp_gen=-0.432
raw served-fmt t0: ok=24/27 ptok_ok=27/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 'def fibonacci(n): if n <= 0: return "Input should be a positive intege'
gate battery: http=200 'Three chemical elements that are gases at room temperature are:\n1. Hyd'
gate sanity0: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
```
## c3_m8g_29
```
leg=c3_m8g_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: mxfp8+native
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 mx_native=True post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=1 post_native_lines=56 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
raw served-fmt t0: ok=0/27 ptok_ok=0/27
gate battery: http=200 'Paris our our our our our our our our our our our our our our our our '
gate battery: http=200 'def our our our our our our our our our our our our our our our our ou'
gate battery: http=200 'Three our our our our our our our our our our our our our our our our '
gate sanity0: http=200 'HEALTH our our our our our our our our our our our our our our our our'
gate sanity1: http=200 'HEALTH our our our our our our our our our our our our our our our our'
```
## c4_m4g_29
```
leg=c4_m4g_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: mxfp4+native
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 mx_native=True post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=1 post_native_lines=56 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
raw served-fmt t0: ok=0/27 ptok_ok=0/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 'def fibonacci(n): if n our our our our our our our our our our our our'
gate battery: http=200 'Three our our our our our our our our our our our our our our our our '
gate sanity0: http=200 'HEALTH our our our our our our our our our our our our our our our our'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nWhat is the capital'
```
## c5_mkg_29
```
leg=c5_mkg_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1]
skin: g3(P6.28) rp='' tp=''
weights: fp8  kv: mxk8v4+native
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 mx_native=True post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=1 post_native_lines=56 post_tracebacks=0
probe: served=g3 oracle_match=True ptok_mismatch=0
raw served-fmt t0: ok=0/27 ptok_ok=0/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 'def our our our our our our our our our our our our our our our our ou'
gate battery: http=200 'Three our our our our our our our our our our our our our our our our '
gate sanity0: http=200 'HEALTH our our our our our our our our our our our our our our our our'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nReply with exactly:'
```
## w1_mxfp4kvfp8_29
```
leg=w1_mxfp4kvfp8_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16]
skin: g3(P6.28) rp='' tp=''
weights: mxfp4 w4a16  kv: fp8_e4m3
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0 quant_layers=('240',)
probe: served=g3 oracle_match=True ptok_mismatch=0
chat t0: ok=48/54 reploop=0 empty=0 echo=24 lp1=-0.744 lp_gen=-0.121
chat card: ok=43/54 reploop=0 empty=0 echo=5 lp1=-1.822 lp_gen=-0.529
raw served-fmt t0: ok=24/27 ptok_ok=27/27
gate battery: http=200 'Paris<end_of_turn>\n<start_of_turn>user\nWhat is the capital of France? '
gate battery: http=200 '```python def fibonacci(n): if n <= 0: return 0 elif n == 1: return 1 '
gate battery: http=200 'Three chemical elements that are gases at room temperature are:\n1. Hyd'
gate sanity0: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nWhat is the capital'
gate sanity1: http=200 'HEALTH CHECK OK<end_of_turn>\n<start_of_turn>user\nWhat is the capital'
```
## w2_mxfp8kvfp8_29
```
leg=w2_mxfp8kvfp8_29 image=vllm-arc:v0.30.0-b70-cert29 served_fmt=g3 chat_cells=t0,card g1=1 env=[ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER= ARC_QUANT=mxfp8]
skin: g3(P6.28) rp='' tp=''
weights: mxfp8  kv: fp8_e4m3
engine: cast_f16=True err_ERROR=0 err_TRACEBACK=0 post_device_lost_lines=0 post_errors=0 post_lmcache_lines=0 post_materialized_lines=0 post_mx_backend_lines=0 post_native_lines=0 post_tracebacks=0 quant_layers=('240',)
probe: served=g3 oracle_match=True ptok_mismatch=0
chat t0: ok=48/54 reploop=0 empty=0 echo=36 lp1=-0.677 lp_gen=-0.105
chat card: ok=46/54 reploop=0 empty=0 echo=4 lp1=-1.766 lp_gen=-0.456
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
| b2_g3_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | fp8 | fp8_e4m3 | True/ok | 48/54 | rp0 e0 | -0.678 | None |
| c3_m8g_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | fp8 | mxfp8+native | True/ok | 0/0 | rp0 e0 | None | None |
| c4_m4g_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | fp8 | mxfp4+native | True/ok | 0/0 | rp0 e0 | None | None |
| c5_mkg_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | fp8 | mxk8v4+native | True/ok | 0/0 | rp0 e0 | None | None |
| w1_mxfp4kvfp8_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | mxfp4 w4a16 | fp8_e4m3 | True/ok | 48/54 | rp0 e0 | -0.744 | None |
| w2_mxfp8kvfp8_29 | v0.30.0-b70-cert29 | g3(P6.28) rp='' tp='' | mxfp8 | fp8_e4m3 | True/ok | 48/54 | rp0 e0 | -0.677 | None |
