# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "gpu-encoding"


def draw(L):
    f = Fig(1100, 470, L, L.t("Every element says which round, sender, block and index it belongs to",
                               "各要素は、どのラウンド、送り手、ブロック、位置のものかを表す"))
    parts = [("nccl", L.t("round + 1", "ラウンド + 1"), "x 0x9E3779B1"), ("nccl", L.t("sender rank", "送り手のランク"), "x 0x85EBCA6B"),
             ("nccl", L.t("block", "ブロック"), "x 0xC2B2AE35"), ("nccl", L.t("index", "位置"), L.t("as is", "そのまま"))]
    for i, (role, name, how) in enumerate(parts):
        x = 40 + i * 255
        f.box(x, 70, 200, 70, name, role, how.replace("x ", "× "), sub_size=13)
        if i:
            f.text(x - 27, 112, "XOR", size=14, weight="bold")
    f.arrow(550, 150, 550, 188)
    f.box(380, 190, 340, 56, L.t("expected value (32 bits)", "期待する値 (32 ビット)"), "target")
    rows = [
        (L.t("data not yet landed (previous round's value, or initial contents)", "データがまだ届いていない (前のラウンドの値か、最初の中身)"), L.t("round differs", "ラウンドが違う")),
        (L.t("chunk from the wrong rank", "別のランクから来たチャンク"), L.t("sender differs", "送り手が違う")),
        (L.t("chunk written into another block's slot", "別のブロックのスロットに書かれた"), L.t("block differs", "ブロックが違う")),
        (L.t("chunk placed at the wrong offset", "違う位置に置かれたチャンク"), L.t("index differs", "位置が違う")),
    ]
    f.text(60, 290, L.t("Misdelivery", "誤配送"), size=14, weight="bold", anchor="start")
    f.text(760, 290, L.t("Why it is caught", "見つかる理由"), size=14, weight="bold", anchor="start")
    for i, (what, why) in enumerate(rows):
        y = 315 + i * 36
        f.line(50, y - 10, 1050, y - 10, colour="#DDDDDD")
        f.text(60, y + 12, what, size=13, anchor="start")
        f.mark(735, y + 6, True)
        f.text(760, y + 12, why, size=13, anchor="start")
    return f
