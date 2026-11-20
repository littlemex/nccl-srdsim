# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "architecture-overview"


def draw(L):
    f = Fig(1100, 640, L, L.t("Where the plugin sits", "プラグインの位置"))
    f.group(20, 55, 520, 545, L.t("Rank A, sending", "ランク A (送る側)"), "nccl")
    f.box(50, 95, 460, 64, L.t("GPU kernel", "GPU のカーネル"), "nccl",
          L.t("GIN device API: put, signal, get", "GIN のデバイス API: put、シグナル、get"))
    f.box(50, 200, 460, 64, L.t("NCCL GIN proxy backend", "NCCL の GIN プロキシバックエンド"), "nccl",
          L.t("a host thread, NCCL_GIN_TYPE=2", "ホストのスレッド、NCCL_GIN_TYPE=2"))
    f.box(50, 305, 460, 64, L.t("This plugin's op table", "このプラグインの関数テーブル"), "sim",
          "iput, iputSignal, iget, iflush, test, rmaProgress")
    f.box(50, 410, 215, 64, L.t("Scheduler", "スケジューラ"), "sim", L.t("chooses the order", "順番を決める"))
    f.box(295, 410, 215, 64, L.t("Send queue", "送信キュー"), "sim", L.t("stages device data", "デバイスのデータを移す"))
    f.box(50, 515, 460, 56, L.t("Transport: one TCP stream per peer", "トランスポート: 相手ごとに 1 本の TCP"), "wire")
    f.arrow(280, 159, 280, 198, L.t("descriptor", "記述子"), lx=292, anchor="start")
    f.arrow(280, 264, 280, 303, L.t("op call", "操作の呼び出し"), lx=292, anchor="start")
    f.arrow(157, 369, 157, 408, L.t("submit", "投入"), lx=169, anchor="start")
    f.arrow(265, 442, 293, 442)
    f.arrow(157, 474, 157, 513, L.t("release", "リリース"), lx=169, anchor="start")
    f.arrow(402, 474, 402, 513)
    f.group(560, 55, 520, 545, L.t("Rank B, the target", "ランク B (相手)"), "target")
    f.box(590, 95, 220, 110, L.t("Data window", "データのウィンドウ"), "target", L.t("payload bytes", "ペイロード"))
    f.box(835, 95, 215, 110, L.t("Signal window", "シグナルのウィンドウ"), "sig", L.t("signal words", "シグナルの値"))
    f.box(590, 300, 460, 70, L.t("Receiving agent", "受信エージェント"), "target",
          L.t("writes each unit into this rank's memory", "ユニットを自分のメモリに書く"))
    f.box(590, 515, 460, 56, L.t("Transport: the receiving end", "トランスポート: 受信側"), "wire")
    f.arrow(510, 543, 588, 543, L.t("frames", "フレーム"))
    f.arrow(820, 515, 820, 372, L.t("in release order", "リリースした順"), lx=832, anchor="start")
    f.arrow(700, 300, 700, 207, L.t("placement", "配置"), lx=712, anchor="start")
    f.arrow(942, 300, 942, 207, L.t("signal", "シグナル"), lx=954, anchor="start")
    f.text(550, 630, L.t("Every rank runs both halves: it sends its own operations and applies its peers'.",
                         "どのランクも両方を受け持つ。自分の操作を送り、相手の操作を自分のメモリに反映する。"),
           size=13, italic=True)
    return f
