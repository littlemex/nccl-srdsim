# SPDX-License-Identifier: Apache-2.0
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
from svgkit import Fig, C
f = Fig(1060, 560, "Release orders of the modes for two operations to the same peer")
f.text(530, 56, "A = A0 A1 A2 + signal As (submitted first),  B = B0 B1 (submitted second)", size=14)
modes = [
 ("srd",         ["B1","A1","A0","B0","A2","As"], "one random legal order; As after A0..A2"),
 ("inorder",     ["A0","A1","A2","As","B0","B1"], "submission order; the control"),
 ("peer-seq",    ["A1","A0","A2","As","B1","B0"], "A before B; chunks within an op reordered"),
 ("reverse-ops", ["B0","B1","A0","A1","A2","As"], "newest operation first"),
 ("signal-last", ["A0","A1","A2","B0","B1","As"], "every payload before any signal"),
 ("defer-one",   ["B0","B1","A0","A1","A2","As"], "A held until B released; same as reverse-ops here"),
 ("violate-i1",  ["A0","As","B0","A1","B1","A2"], "As before A1, A2: breaks I1 on purpose"),
]
y = 80
for name, order, note in modes:
    f.text(30, y+28, name, size=14, weight="bold", anchor="start")
    for k, u in enumerate(order):
        sig = u.endswith("s")
        col = C["sig"] if sig else (C["nccl"] if u.startswith("A") else C["wire"])
        f.cell(160 + k*78, y+6, 70, 36, u, col, hatch=sig)
    f.text(160 + 6*78 + 14, y+29, note, size=13, anchor="start", color=(C["sig"] if name=="violate-i1" else "#333333"))
    y += 62
f.text(530, 538, "Colour and pattern: A units / B units / signal units (hatched). srd draws a new legal order per seed.", size=12, italic=True)
f.save(os.path.join(OUT, "modes.svg"))
