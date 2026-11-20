# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "get-flow"


def draw(L):
    f = Fig(1100, 470, L, L.t("A get: the owner's plugin answers it, one-sided",
                               "get: 持ち主のプラグインが片側通信として答える"))
    f.box(40, 60, 300, 56, L.t("Requester", "要求側"), "nccl")
    f.box(760, 60, 300, 56, L.t("Owner of the memory", "メモリの持ち主"), "target")
    f.line(190, 116, 190, 440, dash=True)
    f.line(910, 116, 910, 440, dash=True)
    rows = [
        (150, True, L.t("iget: a GetRequest unit, released like any unit", "iget: GetRequest のユニット。ほかと同じくリリースする")),
        (215, None, L.t("receive thread copies the bytes", "受信スレッドがバイトをコピーする")),
        (280, None, L.t("responder thread sends them in chunks", "応答スレッドがチャンクに分けて送る")),
        (345, False, L.t("GetResponse chunks, in order, naming the get", "GetResponse のチャンク。順番どおりで、get を名指す")),
        (410, None, L.t("bytes counted per get; at zero the get completes", "get ごとにバイトを数え、0 で get が完了する")),
    ]
    for i, (y, direction, label) in enumerate(rows):
        f.step(560 if direction is not None else (910 if i in (1, 2) else 190), y - 18 if direction is not None else y, i + 1)
        if direction is True:
            f.arrow(192, y, 908, y)
            f.text(550, y + 22, label, size=13)
        elif direction is False:
            f.arrow(908, y, 192, y)
            f.text(550, y + 22, label, size=13)
        else:
            anchor = "end" if i in (1, 2) else "start"
            x = 890 if i in (1, 2) else 210
            f.text(x, y + 5, label, size=13, anchor=anchor)
    f.text(550, 462, L.t("the owner's NCCL progress thread takes no part", "持ち主の NCCL の進行スレッドは関わらない"),
           size=12, color="#555555")
    return f
