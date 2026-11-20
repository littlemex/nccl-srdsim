# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "modes"


def draw(L):
    f = Fig(1100, 600, L, L.t("Release orders of the modes for two operations to the same peer",
                               "同じ相手への 2 つの操作を、各モードがどの順に送り出すか"))
    f.text(550, 58, L.t("A = A0 A1 A2 + signal As (submitted first), B = B0 B1 (submitted second)",
                        "A = A0 A1 A2 + 合図 As (先に受け付け)、B = B0 B1 (後に受け付け)"), size=14)
    modes = [
        ("srd", ["B1", "A1", "A0", "B0", "A2", "As"], L.t("one random legal order per seed", "種ごとに正しい順番を 1 つ選ぶ")),
        ("inorder", ["A0", "A1", "A2", "As", "B0", "B1"], L.t("submission order: the control", "受け付けた順: 対照")),
        ("peer-seq", ["A1", "A0", "A2", "As", "B1", "B0"], L.t("A before B; chunks reordered", "A は B より前、チャンクは入れ替え")),
        ("reverse-ops", ["B0", "B1", "A0", "A1", "A2", "As"], L.t("newest operation first", "新しい操作から")),
        ("signal-last", ["A0", "A1", "A2", "B0", "B1", "As"], L.t("all payloads, then signals", "ペイロードをすべて、次に合図")),
        ("defer-one", ["B0", "B1", "A0", "A1", "A2", "As"], L.t("oldest held back to the end", "最も古い操作を最後まで保留")),
        ("violate-i1", ["A0", "As", "B0", "A1", "B1", "A2"], L.t("As before A1, A2: breaks I1", "As が A1、A2 より前: I1 を破る")),
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
                                     ("sig", L.t("signal", "合図"))]):
        f.cell(170 + i * 170, 535, 30, 18, "", role)
        f.text(208 + i * 170, 549, lab, size=12, anchor="start", color="#333333")
    f.text(1070, 590, L.t("defer-one equals reverse-ops when there are only two operations",
                          "操作が 2 つだけのとき、defer-one は reverse-ops と同じになる"), size=12, anchor="end", color="#555555")
    return f
