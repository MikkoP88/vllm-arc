#!/usr/bin/env python3
"""P619d norms2 — full disambiguation of the t8f dump: real norms of
every output, the data location inside cache_pages, gather-vs-slice
equality, and where tokens 0..8 actually live."""
import torch

d = torch.load("/root/p619f_dump.pt", map_location="cpu", weights_only=False)
mq, ref = d["mq_out"].float(), d["ref0"].float()
print("eng mq rows :", [round(float(r.norm()), 4) for r in mq])
print("eng ref0    :", [round(float(r.norm()), 4) for r in ref])

cp = d["cache_pages"]
bt = d["bt_row"][0]
print("bt_row[:10] :", bt[:10].tolist())
# where is the data?
per_page = (cp != 0).sum(dim=(1, 2, 3))
nz = torch.nonzero(per_page)
print("pages with data:", [(int(p), int(per_page[p])) for p in nz[:, 0][:8]])
p0 = int(bt[0])
g = cp[p0]
print(f"gathered page bt[0]={p0}: nonzero={int((g != 0).sum())}")
print("  slot0 head0 bytes[:16]:", g[0, 0, :16].tolist())
print("  slot1 head0 bytes[:8] :", g[1, 0, :8].tolist())
# tokens 0..8 are slots 0..8 of the FIRST page only if bt[0] holds them;
# if bt[0] is a shared/prefix page the request tokens may sit at a later
# page: report slots-with-data in the first 3 gathered pages
for k in range(min(3, cp.shape[0])):
    pg = cp[k]
    slots = torch.nonzero((pg != 0).sum(dim=(1, 2)))
    print(f"gathered[{k}] (bt={int(bt[k])}): slots-with-data="
          f"{slots[:16].flatten().tolist()}")
print("seq_lens[i-related]:", d["seq_lens"][:10].tolist(), "q0:", d["q0"].tolist())
