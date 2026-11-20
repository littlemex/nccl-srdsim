# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "get-flow"


def draw(L):
    f = Fig(1100, 470, L, L.t("A get: the owner of the memory sends it back like a put",
                               "get: メモリの持ち主が、put と同じようにして送り返す"))
    f.box(40, 60, 300, 56, L.t("Requester", "要求側"), "nccl")
    f.box(760, 60, 300, 56, L.t("Owner of the memory", "メモリの持ち主"), "target")
    f.line(190, 116, 190, 440, dash=True)
    f.line(910, 116, 910, 440, dash=True)
    rows = [
        (150, True, L.t("iget: a GetRequest unit, released like any unit", "iget: GetRequest のユニット。ほかと同じくリリースする")),
        (215, None, L.t("submit the reply on its own first RMA context", "自分の最初の RMA コンテキストで応答を投入する")),
        (280, None, L.t("its rmaProgress releases the reply in chunks", "自分の rmaProgress が応答をチャンクでリリースする")),
        (345, False, L.t("GetResponse chunks, reordered like any put", "GetResponse のチャンク。put と同じく入れ替わる")),
        (410, None, L.t("each chunk applied is counted; iflush waits for all", "反映したチャンクを数え、iflush はすべてを待つ")),
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
    f.text(550, 462, L.t("a get is locally complete once its request is sent", "get は、要求を送った時点でローカル完了になる"),
           size=12, color="#555555")
    return f
