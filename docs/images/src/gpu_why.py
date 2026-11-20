# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "gpu-why"


def draw(L):
    f = Fig(1100, 470, L, L.t("What the host tests show, and what only a GPU run shows",
                               "ホストのテストで分かること、GPU で動かして初めて分かること"))
    # Left: host tests.
    f.group(30, 60, 470, 380, L.t("Host tests (CI, no GPU, no NCCL)", "ホストのテスト (CI、GPU なし、NCCL なし)"),
            "neutral")
    f.box(60, 110, 410, 60, L.t("Scheduler", "スケジューラ"), "sim", L.t("the order operations are delivered in", "操作を届ける順番"))
    f.box(60, 190, 410, 60, L.t("Transport", "転送"), "wire", L.t("carrying data between processes", "プロセスの間でデータを運ぶ"))
    f.box(60, 270, 410, 60, L.t("Send queue", "送信キュー"), "sim", L.t("data copied out of the GPU", "GPU から写したデータ"))
    f.text(265, 370, L.t("Never run here: the plugin under NCCL,", "ここでは動かないもの: NCCL の下の"), size=13)
    f.text(265, 390, L.t("and its writes into GPU memory", "プラグインと、GPU のメモリへの書き込み"), size=13)
    # Right: GPU run.
    f.group(600, 60, 470, 380, L.t("GPU test (one GPU, real NCCL)", "GPU のテスト (GPU 1 枚、本物の NCCL)"), "nccl")
    rows = [
        ("nccl", L.t("GPU kernel", "GPU のカーネル"), L.t("GIN puts with weak signals", "弱い合図付きの GIN の put")),
        ("nccl", L.t("NCCL GIN proxy backend", "NCCL の GIN プロキシバックエンド"), "NCCL_GIN_TYPE=2"),
        ("sim", L.t("This plugin", "このプラグイン"), L.t("loaded through NCCL_RMA_PLUGIN", "NCCL_RMA_PLUGIN で読み込む")),
        ("target", L.t("Device memory", "デバイスのメモリ"), L.t("data and signal counters", "データと合図のカウンタ")),
    ]
    for i, (role, name, sub) in enumerate(rows):
        y = 100 + i * 82
        f.box(640, y, 390, 60, name, role, sub)
        if i:
            f.arrow(835, y - 22, 835, y - 2)
    f.arrow(505, 250, 595, 250, L.t("adds", "加わる"), width=2.5)
    return f
