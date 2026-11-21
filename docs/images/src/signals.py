# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "signals"


def draw(L):
    f = Fig(1100, 520, L, L.t("How a signal is applied", "シグナルの反映のしかた"))
    # Host window.
    f.group(20, 55, 340, 440, L.t("host window", "ホストのウィンドウ"), "target")
    f.box(40, 95, 300, 56, L.t("receiving agent", "受信エージェント"), "target")
    f.cell(60, 175, 260, 38, L.t("1  write the payload", "1  ペイロードを書く"), "sim", size=13)
    f.cell(60, 225, 260, 38, L.t("2  add: memory_order_release", "2  加算: memory_order_release"), "sig", size=13)
    f.arrow(190, 263, 190, 305, L.t("pairs with", "対になる"), lx=200, anchor="start")
    f.box(40, 310, 300, 56, L.t("reader: memory_order_acquire", "読み手: memory_order_acquire"), "nccl")
    f.mark(60, 410, True)
    f.text(80, 415, L.t("then sees every payload byte", "その後はペイロードがすべて見える"), size=13, anchor="start")
    f.text(190, 450, L.t("checked by T12 under ThreadSanitizer", "ThreadSanitizer の下で T12 が確かめる"), size=12,
           color="#555555")
    # Device window.
    f.group(380, 55, 340, 440, L.t("device window", "デバイスのウィンドウ"), "target")
    f.box(400, 95, 300, 76, L.t("window with SIGNAL_NEVER_RESET", "SIGNAL_NEVER_RESET 付きのウィンドウ"), "sim",
          [L.t("store the running total", "プラグインが持つ合計値を"), L.t("the plugin keeps", "8 バイトで書く")], size=13)
    f.box(400, 195, 300, 76, L.t("any other window", "それ以外のウィンドウ"), "neutral",
          [L.t("an atomic add in a kernel", "カーネルの atomic な加算"), L.t("compiled with NVRTC", "(NVRTC でコンパイル)")], size=13)
    f.text(550, 310, L.t("the store needs no SM, and is", "書き込みは SM を使わず、"), size=13)
    f.text(550, 330, L.t("valid only where the plugin is", "プラグインだけが書き手のとき"), size=13)
    f.text(550, 350, L.t("the only writer of the word", "にだけ正しい"), size=13)
    f.text(550, 390, L.t("NCCL registers its signal windows", "NCCL は自分のシグナルのウィンドウを"), size=12, color="#555555")
    f.text(550, 408, L.t("with the flag; an application may", "このフラグ付きで登録する。アプリは"), size=12, color="#555555")
    f.text(550, 426, L.t("reset a word in any other window", "ほかのウィンドウの値を消してよい"), size=12, color="#555555")
    # Strong signal.
    f.group(740, 55, 340, 440, L.t("strong signal", "強いシグナル"), "sig")
    for k, (u, role) in enumerate([("A", "nccl"), ("B", "wire"), ("S!", "sig")]):
        f.cell(775 + k * 95, 110, 80, 42, u, role, size=15)
    f.text(910, 185, L.t("S! is held on its RMA context and target", "S! は同じ RMA コンテキストと宛先ランクで"), size=13)
    f.text(910, 205, L.t("until A and B are released", "A と B をリリースするまで待たされる"), size=13)
    f.text(910, 260, L.t("SRD has no strong signal,", "SRD に強いシグナルは無いが、"), size=13)
    f.text(910, 280, L.t("but refusing one would stop", "断ると NCCL の GIN プロキシの"), size=13)
    f.text(910, 300, L.t("NCCL's GIN proxy thread", "スレッドが止まってしまう"), size=13)
    f.text(910, 350, L.t("the value added: the signal's value", "足す値: ADD 指定ならシグナルの値、"), size=12, color="#555555")
    f.text(910, 368, L.t("for NCCL_GIN_SIGNAL_OP_ADD, else 1", "それ以外は 1"), size=12, color="#555555")
    return f
