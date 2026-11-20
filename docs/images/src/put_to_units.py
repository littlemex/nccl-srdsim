# SPDX-License-Identifier: Apache-2.0
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
from svgkit import Fig, C
f = Fig(1000, 420, "A signalled put becomes placement units and one signal unit")
f.box(30, 60, 250, 70, "iputSignal", C["nccl"], "32 KiB put + signal", size=14)
f.arrow(285, 95, 385, 95)
f.text(335, 82, "8 KiB chunks", size=12, color="#333333")
x = 390
for i in range(4):
    f.cell(x + i*95, 70, 85, 50, f"P{i}", C["sim"])
f.cell(x + 4*95, 70, 85, 50, "S", C["sig"], hatch=True)
f.text(x + 2*95 + 40, 150, "P0..P3 are mutually unordered; in the modes that keep I1, S follows all of P0..P3", size=13)
rows = [("srd", ["P2", "P0", "P3", "P1", "S"], None, "legal: S after its last placement"),
        ("violate-i1", ["P2", "S", "P0", "P3", "P1"], 1, "breaks I1: S visible while P0, P1, P3 are missing")]
y = 190
for name, order, bad, note in rows:
    f.text(60, y+32, name, size=14, weight="bold", anchor="start")
    for k, u in enumerate(order):
        col = C["sig"] if u == "S" else C["sim"]
        f.cell(200 + k*95, y+8, 85, 40, u, col, hatch=(u == "S"))
    f.text(200 + 5*95 + 10, y+33, note, size=13, anchor="start", color=(C["sig"] if bad else "#333333"))
    y += 70
f.arrow(200, 350, 660, 350, "release order (time)", lx=430, ly=342)
f.text(500, 395, "The receiver applies units in exactly this order: each peer's stream preserves it.", size=13, italic=True)
f.save(os.path.join(OUT, "put-to-units.svg"))
