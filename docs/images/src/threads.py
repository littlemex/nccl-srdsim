# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "threads"


def draw(L):
    f = Fig(1100, 560, L, L.t("Threads, RMA contexts and streams of one communicator on rank 0", "ランク 0 の 1 つのコミュニケータのスレッド、RMA コンテキスト、接続"))
    gx = 190
    f.group(gx, 55, 600, 480, L.t("Rank 0", "ランク 0"), "sim")
    bx, bw = gx + 25, 550
    f.box(bx, 90, bw, 60, L.t("NCCL's GIN proxy thread", "NCCL の GIN プロキシのスレッド"), "nccl",
          L.t("calls iput, iputSignal, iget, rmaProgress, test", "iput、iputSignal、iget、rmaProgress、test を呼ぶ"))
    for i in range(2):
        x = bx + i * 285
        f.box(x, 200, 265, 80, L.t(f"RMA context {i}", f"RMA コンテキスト {i}"), "sim",
              [L.t("ordered per target,", "相手ごとに順番を持ち、"), L.t("in one shared scheduler", "1 つのスケジューラを共有")])
        f.arrow(x + 132, 150, x + 132, 198)
        f.arrow(x + 132, 280, x + 132, 333)
    f.text(bx + bw / 2, 305, L.t("nothing is ordered across contexts", "コンテキストをまたぐ順番は無い"), size=13, italic=True)
    f.box(bx, 335, bw, 56, L.t("Send streams: one per peer", "送信接続: 相手ごとに 1 本"), "wire")
    f.box(bx, 445, bw, 64, L.t("Receiving agent: one thread per inbound stream", "受信エージェント: 受信接続ごとに 1 スレッド"),
          "target", L.t("a responder thread sends the replies to gets", "get への応答は応答スレッドが送る"))
    f.box(870, 335, 210, 174, L.t("Ranks 1, 2", "ランク 1、2"), "neutral", L.t("same structure", "同じ作り"))
    f.arrow(bx + bw + 2, 363, 868, 363, L.t("to each peer", "各相手へ"))
    f.arrow(868, 477, bx + bw + 2, 477, L.t("from each peer", "各相手から"))
    f.path(f"M {bx-2},363 C {bx-120},363 {bx-120},477 {bx-2},477", colour="#555555", dash=True)
    f.text(bx - 95, 410, L.t("to itself:", "自分あて:"), size=12, anchor="end", color="#555555")
    f.text(bx - 95, 426, L.t("a socket pair", "ソケットペア"), size=12, anchor="end", color="#555555")
    f.text(935, 540, L.t("every pair of ranks has a stream in each direction", "どの 2 ランクの間にも向きごとに 1 本"),
           size=12, color="#555555")
    return f
