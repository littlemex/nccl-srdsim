# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "gpu-check-order"


def draw(L):
    f = Fig(1100, 655, L, L.t("One round at the receiver: each chunk is checked once its own signal is visible",
                               "受け手の 1 ラウンド: 各チャンクは、自分のシグナルが見えたときに確かめる"))
    X0, DX, CW = 230, 95, 80          # time slot positions
    rows = [L.t("data lands", "データが届く"), L.t("signal visible", "シグナルが見える"), L.t("block checks", "ブロックが確かめる")]

    def lane(top, title, events):
        f.group(30, top, 1040, 235, title, "neutral")
        for i, r in enumerate(rows):
            f.text(60, top + 70 + i * 60, r, size=13, anchor="start", weight="bold")
            f.line(220, top + 74 + i * 60 + 18, 1050, top + 74 + i * 60 + 18, dash=True)
        for row, slot, label, ok in events:
            x, y = X0 + slot * DX, top + 50 + row * 60
            role = ("sim", "sig", "target")[row]
            f.cell(x, y, CW, 36, label, role, size=13)
            if ok is not None:
                f.mark(x + CW - 12, y + 18, ok)

    keep = [(0, 0, "C2", None), (1, 1, "s2", None), (2, 2, L.t("C2", "C2"), True),
            (0, 1, "C0", None), (1, 3, "s0", None), (2, 4, "C0", True),
            (0, 4, "C1", None), (1, 5, "s1", None), (2, 6, "C1", True)]
    lane(55, L.t("A mode that keeps I1 (srd, inorder, ...): every signal after its own data",
                 "I1 を守るモード (srd、inorder など): シグナルはいつも自分のデータの後"), keep)
    bad = [(0, 0, "C2", None), (1, 1, "s2", None), (2, 2, "C2", True),
           (1, 3, "s1", None), (2, 4, "C1", False), (0, 6, "C1", None),
           (0, 2, "C0", None), (1, 5, "s0", None), (2, 7, "C0", True)]
    lane(310, L.t("violate-i1: a signal may become visible before its own data",
                  "violate-i1: シグナルが自分のデータより先に見えることがある"), bad)
    f.text(550, 645, L.t("Three of the 16 chunks are shown. Chunks may land in any order; only a signal ahead of its own data is forbidden.",
                         "16 個のうち 3 個のチャンクを示す。チャンクはどの順に届いてもよく、禁じられるのは自分のデータより先のシグナルだけ。"),
           size=12, color="#555555")
    f.text(X0 + 4 * DX + CW / 2, 615, L.t("C1 is read before its data lands: its elements count as mismatched",
                                         "C1 はデータが届く前に読まれ、その要素は値の違うものとして数えられる"), size=13,
           color="#D55E00", weight="bold")
    f.arrow(X0, 568, X0 + 8 * DX - 10, 568, L.t("time", "時間"), ly=586, width=1.5)
    return f
