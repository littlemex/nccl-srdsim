# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "put-to-units"


def draw(L):
    f = Fig(1100, 430, L, L.t("A signalled put becomes placement units and one signal unit",
                               "シグナル付きの put は、配置のユニットと 1 つのシグナルのユニットになる"))
    f.box(30, 62, 250, 70, "iputSignal", "nccl", L.t("32 KiB + a signal", "32 KiB + シグナル"))
    f.arrow(285, 97, 395, 97, L.t("8 KiB chunks", "8 KiB ずつ"))
    for i in range(4):
        f.cell(400 + i * 100, 72, 88, 50, f"P{i}", "sim")
    f.cell(800, 72, 88, 50, "S", "sig")
    f.text(645, 152, L.t("P0..P3 have no order among them; I1 puts S after all of them",
                         "P0〜P3 の間に順番は無い。I1 は S をそのすべての後に置く"), size=13)
    rows = [("srd", ["P2", "P0", "P3", "P1", "S"], True, L.t("legal: S after its last placement", "正しい: S は最後の配置の後")),
            ("violate-i1", ["S", "P2", "P0", "P3", "P1"], False,
             L.t("breaks I1: S goes before every placement", "I1 を破る: S がすべての配置より先"))]
    y = 190
    for name, order, ok, note in rows:
        f.text(40, y + 32, name, size=15, weight="bold", anchor="start")
        for k, u in enumerate(order):
            f.cell(190 + k * 95, y + 8, 85, 42, u, "sig" if u == "S" else "sim")
        f.mark(700, y + 29, ok)
        f.text(720, y + 34, note, size=13, anchor="start")
        y += 72
    f.arrow(190, 352, 660, 352, L.t("release order (time)", "リリースする順番 (時間)"), ly=344)
    f.text(550, 400, L.t("The target applies units in exactly this order: each peer's stream keeps it.",
                         "相手はユニットをまさにこの順に反映する。相手ごとの接続がこの順番を保つ。"), size=13, italic=True)
    return f
