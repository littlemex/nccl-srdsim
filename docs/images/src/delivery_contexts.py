# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "delivery-contexts"


def draw(L):
    f = Fig(1100, 470, L, L.t("Four ways to write a device window, and what each waits behind",
                               "デバイスのウィンドウへの 4 つの書き方と、それぞれが何の後ろで待つか"))
    cols = [
        (L.t("own context", "専用のコンテキスト"), L.t("(default)", "(既定)"), "sim",
         [L.t("CUDA copy from a", "プラグインが作った"), L.t("context the plugin made", "コンテキストからのコピー")],
         L.t("VMM windows (ncclMemAlloc)", "VMM のウィンドウ (ncclMemAlloc)"),
         [(True, L.t("not the app's module loads", "アプリのモジュール読込を待たない"))]),
        ("primary", "SRDSIM_DELIVERY_CONTEXT", "neutral",
         [L.t("CUDA copy from the", "デバイスの primary"), L.t("primary context", "context からのコピー")],
         L.t("any allocation", "どんな確保でもよい"),
         [(False, L.t("waits behind module loads", "モジュールの読み込みと"), ), (False, L.t("and frees: can deadlock", "解放を待つ: 止まりうる"))]),
        ("green", "SRDSIM_DELIVERY_CONTEXT", "neutral",
         [L.t("CUDA copy on the", "最も小さい SM の区切りの"), L.t("smallest SM partition", "green context でのコピー")],
         L.t("VMM windows, CUDA 12.4+", "VMM のウィンドウ、CUDA 12.4 以降"),
         [(True, L.t("no time-slice wait", "割り当て時間を待たない")), (False, L.t("shares the primary lock", "primary とロックを共有"))]),
        ("GDRCopy", L.t("(build option)", "(組み立ての選択)"), "target",
         [L.t("CPU writes through", "CPU が GPU の BAR の"), L.t("the GPU's BAR mapping", "マッピング越しに書く")],
         L.t("/dev/gdrdrv", "/dev/gdrdrv"),
         [(True, L.t("never waits for the GPU", "GPU を一切待たない"))]),
    ]
    w, gap = 245, 20
    for i, (name, how, role, does, needs, notes) in enumerate(cols):
        x = 30 + i * (w + gap)
        f.box(x, 62, w, 66, name, role, how, size=16, sub_size=11)
        f.box(x, 150, w, 76, L.t("does", "すること"), "neutral", does, size=13, pattern=False)
        f.box(x, 245, w, 62, L.t("needs", "要るもの"), "neutral", needs, size=13, pattern=False)
        for k, (ok, note) in enumerate(notes):
            yy = 345 + k * 34
            f.mark(x + 18, yy, ok)
            f.text(x + 38, yy + 5, note, size=13, anchor="start", limit=(x, x + w + 10))
    f.text(550, 455, L.t("The hazard: the application's kernels are resident and spinning while they wait for the data a copy delivers.",
                         "危ないのは、アプリのカーネルが、コピーが届けるデータを待って居座り続けているとき。"),
           size=13, italic=True)
    return f
