# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "why"


def draw(L):
    f = Fig(1100, 470, L, L.t("Code that relies on more order than the contract passes in-order tests and fails on SRD",
                               "契約より強い順番に頼るコードは、順番どおりのテストを通り、SRD で壊れる"))
    f.text(550, 58, L.t("Rank 0 puts A (no signal), then B with a signal. Rank 1 waits for the signal, then reads A and B.",
                        "ランク 0 が A (合図なし)、次に合図付きの B を put する。ランク 1 は合図を待ち、A と B を読む。"),
           size=13, color="#555555")
    cols = [(40, L.t("An in-order transport (a test setup)", "順番どおりに届く転送 (テストの環境)"),
             ["A", "B", "S"], L.t("A has landed: right value, bug hidden",
                                  "A は届いている。読む値は正しく、不具合は隠れたまま"), True),
            (580, L.t("SRD, or this simulator", "SRD、またはこのシミュレータ"),
             ["B", "S", "A"], L.t("S may come before A: A reads wrong",
                                  "S が A より先に届きうる。A を読むと誤った値"), False)]
    for x, title, order, note, ok in cols:
        f.group(x, 85, 480, 340, title, "sim" if not ok else "neutral")
        f.text(x + 240, 140, L.t("arrival order at rank 1", "ランク 1 に届く順番"), size=13, color="#555555")
        for k, u in enumerate(order):
            role = "sig" if u == "S" else ("nccl" if u == "A" else "wire")
            f.cell(x + 80 + k * 115, 160, 90, 50, u, role, size=18)
            if k:
                f.arrow(x + 80 + k * 115 - 25, 185, x + 80 + k * 115 - 2, 185)
        f.box(x + 60, 260, 360, 64, L.t("wait for S, then read A and B", "S を待ってから A と B を読む"), "target")
        f.mark(x + 90, 375, ok)
        f.text(x + 112, 381, note, size=14, anchor="start", limit=(x, x + 480))
    f.text(550, 455, L.t("The contract (I1) orders S only after its own put B; nothing orders A before S.",
                         "契約 (I1) が S より前を保証するのは、自分の put である B だけ。A が S より前に届く保証は無い。"),
           size=13, italic=True)
    return f
