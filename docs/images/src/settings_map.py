# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "settings-map"


def draw(L):
    f = Fig(1100, 580, L, L.t("Where each group of settings acts on an operation's path",
                               "設定のまとまりが、操作の通り道のどこに効くか"))
    f.text(550, 54, L.t("Names are shown without the SRDSIM_ prefix", "名前は先頭の SRDSIM_ を省いて書いている"),
           size=13, color="#555555")
    # The path, left to right.
    stages = [
        ("nccl", L.t("NCCL calls", "NCCL の呼び出し"), L.t("iput, iputSignal, iget", "iput、iputSignal、iget")),
        ("sim", L.t("Scheduler", "スケジューラ"), L.t("splits and orders", "分割と順番")),
        ("sim", L.t("Send side", "送る側"), L.t("staging, send queue", "staging、送信キュー")),
        ("wire", L.t("Transport", "トランスポート"), L.t("TCP per peer", "相手ごとの TCP")),
        ("target", L.t("Receive side", "受信側"), L.t("writes the target", "宛先ランクのメモリに書く")),
    ]
    x0, w, gap, y = 30, 190, 22, 262
    for i, (role, name, sub) in enumerate(stages):
        x = x0 + i * (w + gap)
        f.box(x, y, w, 70, name, role, sub)
        if i:
            f.arrow(x - gap, y + 35, x - 2, y + 35)
    cx = [x0 + i * (w + gap) + w / 2 for i in range(5)]

    def note(i, top, title, names, role, dash=False, wide=1):
        """A group of settings above (top=True) or below the stage it acts on."""
        bw = w * wide + gap * (wide - 1)
        bx = cx[i] - w / 2
        by = 72 if top else 370
        f.box(bx, by, bw, 150 if top else 170, title, role, names, dash=dash, size=14, sub_size=12, pattern=False)
        f.line(cx[i], by + (150 if top else 0), cx[i], y if top else y + 70, dash=True)

    note(1, True, L.t("Schedule", "スケジュール"),
         ["MODE", "SEED", "CHUNK_BYTES", "DEFER_KIND"], "sim")
    note(2, True, L.t("Delivery", "配送"),
         ["ASYNC_DELIVERY", "STAGING_RING", "PAGEABLE_STAGING", "PROGRESS_ONE_CONTEXT"],
         "sim")
    note(3, True, L.t("Connection", "接続"), ["LOCAL_IP"], "wire")
    note(4, True, L.t("Delivery", "配送"),
         ["DELIVERY_CONTEXT", "NO_GDRCOPY", "SIGNAL_KERNEL"], "target")
    note(0, False, L.t("Change the data", "データを変える"),
         ["DRY_RUN", "SNAPSHOT_SOURCE", L.t("(at submit)", "(投入時)")], "sig")
    note(4, False, L.t("Negative controls", "ネガティブコントロール"),
         ["CORRUPT_EVERY", "DROP_EVERY", L.t("(device writes)", "(デバイスへの書き込み)")], "sig")
    # Impairments: read at connect, applied on the send and receive sides.
    f.box(cx[1] - w / 2, 370, w * 2 + gap, 170, L.t("Impairments", "インペアメント"), "neutral",
          ["DELAY_US_MIN, DELAY_US_MAX", "BURST, BURST_PAUSE_US", "APPLY_INTERVAL_US, SEND_BUFFER_BYTES",
           "FREEZE_PEER, FREEZE_AFTER_UNITS, FREEZE_MS", "SIGNAL_LAG_US",
           L.t("(read at connect, slow the fabric on purpose)", "(接続時に読み、わざと遅くする)")], size=14, pattern=False)
    f.box(cx[3] - w / 2, 370, w, 170, L.t("Diagnostics", "診断"), "neutral",
          [L.t("logs, probes,", "ログ、プローブ、"), L.t("stall monitor,", "停止の監視、"),
           L.t("crash handler", "クラッシュハンドラ"), L.t("(whole path)", "(通り道全体)")], size=14, pattern=False)
    return f
