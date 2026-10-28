import json, time, urllib.request, hashlib, random
URL="http://127.0.0.1:8000/v1/completions"; MODEL="qwen3.8-27b-fp8"
def send(ids, max_tokens):
    body=json.dumps({"model":MODEL,"prompt":ids,"max_tokens":max_tokens,"temperature":0,"ignore_eos":True}).encode()
    t0=time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"}),timeout=900) as r:
        obj=json.load(r)
    return (time.perf_counter()-t0)*1000, hashlib.sha256(obj["choices"][0]["text"].encode()).hexdigest()[:16]
rng=random.Random(20261028); ids=[rng.randint(1000,50000) for _ in range(8192)]
ts=[]
for i in range(5):
    dt,sha=send(ids,1); ts.append(dt); print("warm[%d] %.1f ms sha=%s"%(i,dt,sha),flush=True)
s=sorted(ts); print("WARM5 min=%.1f med=%.1f max=%.1f gate_med<=750:%s"%(s[0],s[2],s[-1],s[2]<=750))
