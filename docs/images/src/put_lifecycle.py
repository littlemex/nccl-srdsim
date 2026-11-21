# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "put-lifecycle"


def draw(L):
    f = Fig(1100, 430, L, L.t("Life of a signalled put, in the order of the numbered steps in the text",
                               "シグナル付きの put の一生 (番号は本文の手順と同じ)"))
    steps = [
        ("nccl", L.t("Submit", "投入"), [L.t("iputSignal returns", "iputSignal がすぐ"), L.t("at once", "戻る")]),
        ("sim", L.t("Split", "分割"), [L.t("8 KiB placements", "8 KiB の配置"), L.t("+ 1 signal unit", "+ シグナル 1 つ")]),
        ("sim", L.t("Release", "リリース"), [L.t("rmaProgress pulls", "rmaProgress が"), L.t("64 per plugin", "プラグインの"), L.t("context", "コンテキストごと 64 個")]),
        ("wire", L.t("Send", "送信"), [L.t("in release order", "リリースした順に"), L.t("to the target", "宛先ランクの接続へ")]),
        ("target", L.t("Apply", "反映"), [L.t("the target writes", "宛先ランクが自分の"), L.t("its own memory", "メモリに書く")]),
        ("nccl", L.t("Complete", "完了"), [L.t("test: source reusable", "test: 送り元を再利用可"),
                                           L.t("iflush: all released", "iflush: 全リリース済み"),
                                           L.t("and gets landed", "で get も到着")]),
    ]
    w, gap, y = 155, 22, 120
    for i, (role, name, sub) in enumerate(steps):
        x = 30 + i * (w + gap)
        f.box(x, y, w, 110, name, role, sub, size=15)
        f.step(x + w / 2, y - 22, i + 1)
        if i:
            f.arrow(x - gap, y + 55, x - 2, y + 55)
    f.group(30, 290, 1040, 110, L.t("Who acts", "誰が動くか"), "neutral")
    f.text(30 + 2.5 * (w + gap) + w / 2 - 280, 345, L.t("sender: NCCL's proxy thread calls steps 1 to 4 and 6",
                                                         "送る側: NCCL のプロキシのスレッドが 1〜4 と 6 を呼ぶ"),
           size=14, anchor="start")
    f.text(30 + 2.5 * (w + gap) + w / 2 - 280, 372, L.t("target: the receiving agent's thread does step 5",
                                                         "宛先ランク: 受信エージェントのスレッドが 5 を行う"),
           size=14, anchor="start")
    return f
