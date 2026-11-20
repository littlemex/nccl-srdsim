# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "components"


def draw(L):
    f = Fig(1100, 450, L, L.t("Source files and what includes what", "ソースのファイルと、どれがどれを取り込むか"))
    f.box(250, 60, 400, 80, "nccl_rma_srdsim.cc", "sim",
          [L.t("op table, connections, windows,", "操作表、接続、ウィンドウ、"),
           L.t("device delivery, completion", "デバイスへの配送、完了")])
    heads = [("srdsim_schedule.h", L.t("order of release", "送り出す順番")),
             ("srdsim_transport.h", L.t("frames, receive agent", "フレーム、受け取り")),
             ("srdsim_async.h", L.t("send queue", "送信キュー")),
             ("srdsim_mapping.h", L.t("BAR arithmetic", "BAR の計算"))]
    for i, (name, sub) in enumerate(heads):
        x = 30 + i * 212
        f.box(x, 250, 196, 72, name, "sim", sub, size=14)
        f.arrow(450, 140, x + 98, 248, colour="#555555")
    f.text(450, 370, L.t("each header includes only the C++ standard library: tested on any host",
                         "どのヘッダも C++ の標準ライブラリしか取り込まない。どのホストでもテストできる"),
           size=13, italic=True)
    f.box(880, 250, 196, 72, "srdsim_impair.h", "neutral", L.t("impairments", "インペアメント"), size=14, dash=True)
    f.box(880, 380, 196, 50, L.t("liveness test", "liveness のテスト"), "neutral", size=13)
    f.arrow(978, 380, 978, 324, colour="#555555", dash=True)
    f.text(978, 232, L.t("not used by the plugin", "プラグインは使わない"), size=12, color="#555555")
    ext = [(L.t("NCCL plugin headers", "NCCL のプラグインのヘッダ"), False),
           (L.t("CUDA driver, NVRTC", "CUDA ドライバ、NVRTC"), True),
           (L.t("GDRCopy", "GDRCopy"), True)]
    for i, (name, opt) in enumerate(ext):
        y = 55 + i * 55
        f.box(780, y, 290, 44, name, "nccl", dash=opt, size=13)
        f.arrow(652, 100, 778, y + 22, colour="#555555", dash=opt)
    f.text(1070, 445, L.t("dashed: optional at build time, or not used by the plugin",
                          "破線: 組み立てで任意のもの、またはプラグインが使わないもの"), size=12, anchor="end", color="#555555")
    return f
