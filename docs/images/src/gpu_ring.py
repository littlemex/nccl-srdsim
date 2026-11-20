# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "gpu-ring"


def draw(L):
    f = Fig(1100, 650, L, L.t("Three ranks in a ring on one GPU", "1 枚の GPU の上で輪になる 3 つのランク"))
    # Top: the processes.
    f.group(30, 55, 1040, 225, L.t("GPU 0, shared through one MPS server", "GPU 0。1 つの MPS サーバーで共有する"), "nccl")
    xs = [80, 430, 780]
    for r, x in enumerate(xs):
        f.box(x, 100, 240, 90, L.t(f"rank {r}", f"ランク {r}"), "nccl",
              [L.t("own process", "別のプロセス"), f"NCCL_HOSTID=srdsim-rank-{r}"], sub_size=12)
    f.arrow(322, 130, 428, 130, L.t("puts", "put"), width=2.5)
    f.arrow(672, 130, 778, 130, L.t("puts", "put"), width=2.5)
    f.path("M 900 192 C 900 240, 200 240, 200 194", width=2.5)
    f.text(550, 266, L.t("rank 2 puts to rank 0", "ランク 2 はランク 0 へ put する"), size=12, color="#333333")
    # Bottom: one block between rank r and rank r + 1.
    f.text(550, 312, L.t("One block, in one round (block b of rank r sends to block b of rank r + 1)",
                         "1 ラウンドの 1 つのブロック (ランク r のブロック b が、ランク r + 1 のブロック b へ送る)"),
           size=15, weight="bold")
    f.group(30, 330, 330, 290, L.t("rank r, block b", "ランク r、ブロック b"), "nccl")
    f.group(740, 330, 330, 290, L.t("rank r + 1, block b", "ランク r + 1、ブロック b"), "target")
    f.box(55, 370, 280, 60, L.t("send slot", "送るスロット"), "nccl", L.t("65,536 elements", "65,536 要素"))
    f.box(765, 370, 280, 60, L.t("receive slot", "受け取るスロット"), "target", L.t("16 chunks", "16 チャンク"))
    for c in range(4):
        f.cell(400 + c * 75, 360, 62, 34, f"C{c}" if c < 3 else "C15", "sim", size=13)
        f.cell(400 + c * 75, 402, 62, 26, f"s{c}" if c < 3 else "s15", "sig", size=12)
    f.arrow(338, 400, 396, 400)
    f.arrow(700, 400, 762, 400)
    f.text(550, 452, L.t("16 chunks of 4,096 elements, each with a weak signal of its own id", "4,096 要素のチャンク 16 個。それぞれに別の ID の弱い合図が付く"),
           size=13)
    f.step(380, 345, 1)
    f.box(765, 490, 280, 60, L.t("check each chunk", "チャンクを確かめる"), "target",
          L.t("once its signal is visible", "合図が見えたら"))
    f.step(745, 520, 2)
    f.arrow(765, 570, 340, 570, L.t("weak signal: the slot is free again", "弱い合図: スロットが空いた"), dash=True,
            width=2.5, ly=562)
    f.step(560, 595, 3)
    f.box(55, 490, 280, 60, L.t("flush; next round: wait, refill", "flush。次のラウンドで待って詰め直す"), "nccl", L.t("then puts again", "それから再び put"))
    f.step(40, 520, 4)
    return f
