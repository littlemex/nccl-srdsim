# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "receive-side"


def draw(L):
    f = Fig(1100, 480, L, L.t("The receive side: the target writes into its own memory",
                               "受け取る側: 相手が自分のメモリに書く"))
    f.box(30, 70, 270, 64, L.t("Peer stream (TCP)", "相手からの接続 (TCP)"), "wire",
          L.t("frames in release order", "送り出した順のフレーム"))
    f.box(400, 70, 300, 64, L.t("Read the frame header", "フレームのヘッダを読む"), "wire",
          L.t("placement, signal, get", "配置、合図、get"))
    f.box(800, 70, 270, 64, L.t("Window table", "ウィンドウの表"), "target",
          L.t("window number to address", "番号から自分のアドレスへ"))
    f.box(400, 195, 300, 64, L.t("Check", "確かめる"), "target",
          L.t("known window, within bounds", "知っている番号、範囲の中"))
    f.box(30, 330, 300, 70, L.t("Host window", "ホストのウィンドウ"), "target", "memcpy")
    f.box(400, 330, 300, 70, L.t("Device window", "デバイスのウィンドウ"), "target",
          L.t("CUDA copy or GDRCopy", "CUDA のコピーか GDRCopy"))
    f.box(770, 330, 300, 70, L.t("Signal word", "合図の値"), "sig",
          [L.t("host: an add with release order", "ホスト: release の順序付きの加算"),
           L.t("device: a store of the running total", "デバイス: 合計値の書き込み")])
    f.arrow(300, 102, 398, 102)
    f.arrow(700, 102, 798, 102, L.t("number", "番号"))
    f.arrow(935, 134, 702, 215, L.t("address", "アドレス"), lx=860, ly=185)
    f.arrow(550, 134, 550, 193)
    f.arrow(500, 259, 180, 328)
    f.arrow(550, 259, 550, 328)
    f.arrow(600, 259, 920, 328)
    f.arrow(398, 227, 230, 227, colour="#D55E00", dash=True)
    f.text(225, 210, L.t("refused: the frame is skipped;", "断る: そのフレームを飛ばす。"), size=12, anchor="end", color="#D55E00")
    f.text(225, 226, L.t("if skipping would lose the next", "飛ばすと次のフレームの位置を"), size=12, anchor="end", color="#D55E00")
    f.text(225, 242, L.t("frame, the link is closed", "失うときは、接続を閉じる"), size=12, anchor="end", color="#D55E00")
    f.text(550, 455, L.t("A unit names a window by number, as a NIC's put names a remote key, never a pointer.",
                         "ユニットはウィンドウを番号で指す。NIC の put がリモートキーで指すのと同じで、ポインタは送らない。"),
           size=13, italic=True)
    return f
