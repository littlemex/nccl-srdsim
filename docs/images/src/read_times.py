# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "read-times"


def draw(L):
    f = Fig(1100, 400, L, L.t("When the plugin reads each setting", "プラグインが各設定を読むとき"))
    y = 150
    f.arrow(40, y, 1070, y, width=2.5)
    f.text(1065, y + 26, L.t("process time", "プロセスの時間"), size=12, anchor="end", color="#555555")
    events = [
        (150, L.t("library loaded", "ライブラリの読み込み"), "sig", L.t("once", "1 回"),
         ["CORRUPT_EVERY", "DROP_EVERY"]),
        (415, L.t("first use of each", "それぞれの最初の利用"), "sim", L.t("once, kept", "1 回、以後そのまま"),
         [L.t("delivery, diagnostics,", "配送、診断、"), L.t("DRY_RUN, SNAPSHOT_SOURCE", "DRY_RUN、SNAPSHOT_SOURCE"),
          L.t("CHUNK_BYTES as slot size", "スロットの大きさの CHUNK_BYTES")]),
        (680, L.t("each RMA context created", "RMA コンテキストを作るたび"), "nccl", L.t("per context", "コンテキストごと"),
         ["MODE, SEED, CHUNK_BYTES", "DEFER_KIND, TRACE", "DETECT_OVERLAP"]),
        (945, L.t("each address published", "アドレスを公開するたび"), "wire", L.t("every time", "毎回"),
         ["LOCAL_IP"]),
    ]
    for x, when, role, how, names in events:
        f.items.append(f'<circle cx="{x}" cy="{y}" r="9" fill="#FFFFFF" stroke="#222222" stroke-width="2.5"/>')
        f.text(x, y - 46, when, size=14, weight="bold")
        f.text(x, y - 24, how, size=12, color="#555555")
        f.box(x - 115, y + 50, 230, 150, L.t("reads", "読むもの"), role, names, size=14, pattern=False)
        f.line(x, y + 9, x, y + 50, dash=True)
    f.text(550, 385, L.t("Set every setting before NCCL initialises; none is read again on the data path.",
                         "設定はすべて NCCL の初期化より前に決めておく。データの通り道で読み直すものは無い。"),
           size=13, italic=True)
    return f
