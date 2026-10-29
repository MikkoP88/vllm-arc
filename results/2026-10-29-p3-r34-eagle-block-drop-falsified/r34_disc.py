import json, time, urllib.request, hashlib, random
URL="http://127.0.0.1:8000/v1/completions"; MODEL="qwen3.8-27b-fp8"
def send(ids, mt):
    body=json.dumps({"model":MODEL,"prompt":ids,"max_tokens":mt,"temperature":0,"ignore_eos":True}).encode()
    t0=time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"}),timeout=900) as r:
        obj=json.load(r)
    return (time.perf_counter()-t0)*1000, hashlib.sha256(obj["choices"][0]["text"].encode()).hexdigest()[:16]
def ids_for(seed, n):
    rng=random.Random(seed); return [rng.randint(1000,50000) for _ in range(n)]
g2 = ids_for(30303002, 8192)
n4 = ids_for(202610294, 8064)
dt,a1 = send(g2,1);  print("A1 g2 mt=1  %.0fms %s"%(dt,a1),flush=True); time.sleep(1)
dt,b1 = send(g2,64); print("B1 g2 mt=64 %.0fms %s"%(dt,b1),flush=True); time.sleep(1)
for i in range(4):
    dt,s = send(n4,1); print("N4[%d] %.0fms %s"%(i,dt,s),flush=True); time.sleep(1)
dt,a2 = send(g2,1);  print("A2 g2 mt=1  %.0fms %s"%(dt,a2),flush=True); time.sleep(1)
dt,b2 = send(g2,64); print("B2 g2 mt=64 %.0fms %s"%(dt,b2),flush=True)
print("PREFILL_%s DECODE_%s"%("FLIP" if a2!=a1 else "STABLE", "FLIP" if b2!=b1 else "STABLE"))
