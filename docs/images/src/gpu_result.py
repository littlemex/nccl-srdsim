# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "gpu-result"


def draw(L):
    f = Fig(1100, 545, L, L.t("The record: four runs of run.sh, 5 seeds per mode, 3 ranks",
                               "記録: run.sh を 4 回、モードごとに 5 シード、3 ランク"))
    modes = ["srd", "inorder", "peer-seq", "reverse-ops", "signal-last", "defer-one", "violate-i1"]
    f.text(60, 75, L.t("Mode", "モード"), size=14, weight="bold", anchor="start")
    f.text(400, 75, L.t("Exit status per seed (every run)", "シードごとの終了ステータス (全回)"), size=14, weight="bold")
    f.text(820, 75, L.t("Outcome (as required)", "結果 (求めたとおり)"), size=14, weight="bold")
    for i, m in enumerate(modes):
        y = 92 + i * 44
        bad = m == "violate-i1"
        f.text(60, y + 24, m, size=14, anchor="start", mono=True)
        for k in range(5):
            f.cell(250 + k * 62, y + 4, 52, 30, "1" if bad else "0", "sig" if bad else "sim", size=14)
        f.mark(760, y + 19, True)
        f.text(780, y + 24, L.t("mismatched elements found", "値の違う要素が見つかった") if bad else
               L.t("no mismatched element", "値の違う要素なし"), size=13, anchor="start")
    # Range bar for violate-i1.
    y = 420
    f.text(60, y, L.t("violate-i1: mismatched elements rank 0 found, per seed, of 13,107,200 checked",
                      "violate-i1: ランク 0 が見つけた値の違う要素 (1 シードで確かめる 13,107,200 個のうち)"),
           size=13, anchor="start", weight="bold")
    x0, wbar = 60, 980
    f.items.append(f'<rect x="{x0}" y="{y+14}" width="{wbar}" height="22" fill="#FFFFFF" stroke="#888888"/>')
    lo, hi = 3600256 / 13107200, 11626592 / 13107200
    f.cell(x0 + wbar * lo, y + 14, wbar * (hi - lo), 22, "", "sig")
    f.text(x0 + wbar * lo, y + 58, "3,600,256 (27%)", size=12, anchor="middle")
    f.text(x0 + wbar * hi, y + 58, "11,626,592 (89%)", size=12, anchor="middle")
    f.text(x0, y + 82, "0", size=12, anchor="start", color="#555555")
    f.text(x0 + wbar, y + 82, "13,107,200", size=12, anchor="end", color="#555555")
    return f
