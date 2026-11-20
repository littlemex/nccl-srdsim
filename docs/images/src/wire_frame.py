# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "wire-frame"


def draw(L):
    f = Fig(1100, 350, L, L.t("One frame on the wire: a 56-byte header, then the payload",
                               "トランスポートの 1 フレーム: 56 バイトのヘッダと、その後のペイロード"))
    fields = [("kind", 4, "wire"), ("window", 4, "target"), ("dest_window", 4, "neutral"), ("requester", 4, "neutral"),
              ("offset", 8, "target"), ("length", 8, "wire"), ("value", 8, "sig"), ("op_id", 8, "nccl"),
              ("dest_offset", 8, "neutral")]
    x, y = 30, 80
    for name, b, role in fields:
        w = 88 if b == 4 else 108
        f.cell(x, y, w - 4, 60, [name, f"{b} B"], role, size=12)
        x += w
    f.cell(x + 6, y, 1070 - x - 6, 60, [L.t("payload", "ペイロード"), L.t("length bytes", "length バイト")], "sim", size=12)
    rows = [
        ("Placement", L.t("window + offset, length bytes of payload", "window + offset に、length バイトを書く")),
        ("Signal", L.t("add value to the 8 bytes at window + offset", "window + offset の 8 バイトに value を足す")),
        ("GetRequest", L.t("read value bytes (length is 0); reply to requester at dest_window", "value バイトを読み (length は 0)、requester の dest_window に返す")),
        ("GetResponse", L.t("a placement, counted for the requester's get whose id is op_id", "配置として書き、op_id の番号を持つ要求側の get の分として数える")),
        ("Shutdown", L.t("the sender is closing the stream", "送り手が接続を閉じる")),
    ]
    yy = 180
    f.text(30, yy, L.t("kind", "kind"), size=13, weight="bold", anchor="start")
    f.text(220, yy, L.t("what the receiving agent does", "受信エージェントがすること"), size=13, weight="bold", anchor="start")
    for i, (k, what) in enumerate(rows):
        f.text(30, yy + 28 + i * 26, k, size=13, anchor="start", mono=True)
        f.text(220, yy + 28 + i * 26, what, size=13, anchor="start")
    f.text(1070, 340, L.t("grey fields are used by gets only", "灰色の欄は get だけが使う"), size=12, anchor="end", color="#555555")
    return f
