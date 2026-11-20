# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "modes"


def draw(L):
    f = Fig(1100, 600, L, L.t("Release orders of the modes for two operations to the same peer",
                               "同じ宛先ランクへの 2 つの操作を、各モードがどの順にリリースするか"))
    f.text(550, 58, L.t("A = A0 A1, no signal (submitted first); B = B0 B1 B2 + signal Bs (submitted second)",
                        "A = A0 A1、シグナルなし (先に投入)。B = B0 B1 B2 + シグナル Bs (後に投入)"), size=14)
    modes = [
        ("srd", ["B2", "A1", "B0", "B1", "Bs", "A0"], L.t("random per seed; Bs before A0 is legal", "シードごとにランダム。Bs が A0 より先でもよい")),
        ("inorder", ["A0", "A1", "B0", "B1", "B2", "Bs"], L.t("submission order: the control", "投入した順: 対照")),
        ("peer-seq", ["A1", "A0", "B2", "B0", "B1", "Bs"], L.t("A before B; chunks reordered", "A を B より前に、チャンクは入れ替え")),
        ("reverse-ops", ["B0", "B1", "B2", "Bs", "A0", "A1"], L.t("newest first, lowest chunk first", "新しい操作から、チャンクは小さい順")),
        ("signal-last", ["A1", "B0", "A0", "B2", "B1", "Bs"], L.t("all payloads, then signals", "ペイロードをすべて、次にシグナル")),
        ("defer-one", ["B0", "B2", "B1", "Bs", "A1", "A0"], L.t("oldest held to the end; random chunks", "最も古い操作を最後まで保留、チャンクはランダム")),
        ("violate-i1", ["Bs", "B1", "A0", "B0", "A1", "B2"], L.t("a pending signal goes first: breaks I1", "未リリースのシグナルを最初に: I1 を破る")),
    ]
    y = 82
    for name, order, note in modes:
        f.text(30, y + 28, name, size=15, weight="bold", anchor="start")
        for k, u in enumerate(order):
            role = "sig" if u.endswith("s") else ("nccl" if u.startswith("A") else "wire")
            f.cell(170 + k * 80, y + 6, 72, 36, u, role)
        f.mark(670, y + 25, name != "violate-i1")
        f.text(690, y + 30, note, size=13, anchor="start")
        y += 64
    for i, (role, lab) in enumerate([("nccl", L.t("units of A", "A のユニット")), ("wire", L.t("units of B", "B のユニット")),
                                     ("sig", L.t("signal", "シグナル"))]):
        f.cell(170 + i * 170, 535, 30, 18, "", role)
        f.text(208 + i * 170, 549, lab, size=12, anchor="start", color="#333333")
    f.text(1070, 590, L.t("one example order per mode; the randomised modes draw another per seed",
                          "モードごとの順番の一例。ランダムなモードはシードごとに別の順番になる"), size=12, anchor="end", color="#555555")
    return f
