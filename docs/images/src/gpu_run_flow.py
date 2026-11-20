# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "gpu-run-flow"


def draw(L):
    f = Fig(1100, 620, L, L.t("What tests/gpu/run.sh does for each mode and seed",
                               "tests/gpu/run.sh がモードとシードごとにすること"))
    f.box(40, 60, 300, 70, L.t("start its own MPS server", "自分の MPS サーバーを起動"), "nccl",
          L.t("refuses to start next to another", "別のものが動いていれば始めない"))
    f.step(30, 95, 1)
    f.group(380, 55, 690, 395, L.t("for each mode, for each seed 1..SEEDS", "モードごと、シード 1..SEEDS ごと"), "neutral")
    f.arrow(342, 95, 408, 120)
    f.box(410, 95, 300, 70, L.t("launch every rank", "全ランクを起動"), "nccl",
          [L.t("its own NCCL_HOSTID, same mode and seed", "ランクごとの NCCL_HOSTID、同じモードとシード"),
           L.t("each under timeout RUN_TIMEOUT", "それぞれ timeout RUN_TIMEOUT の下で")], sub_size=11)
    f.step(400, 130, 2)
    f.box(410, 205, 300, 60, L.t("wait for each rank to exit", "各ランクが終わるのを待つ"), "nccl")
    f.arrow(560, 167, 560, 203)
    f.step(400, 235, 3)
    # Outcomes.
    outs = [(L.t("exit 0 or 1", "終了 0 か 1"), L.t("keep waiting", "待ち続ける"), "sim"),
            (L.t("exit 3: a wait timed out", "終了 3: 待ちが時間切れ"), L.t("others get 10 s, then stopped", "他は 10 秒後に止める"), "sig"),
            (L.t("any other exit", "それ以外の終了"), [L.t("setup error, crash, time limit:", "準備の失敗、クラッシュ、上限:"), L.t("others stopped at once", "他をすぐに止める")], "sig")]
    for i, (a, b, role) in enumerate(outs):
        y = 95 + i * 85
        f.box(760, y, 290, 66, a, role, b, sub_size=12)
        f.arrow(712, 235, 758, y + 33)
    f.box(410, 330, 300, 90, L.t("status of the seed", "シードの結果"), "nccl",
          [L.t("the first exit other than 0 or 1;", "0 と 1 以外の最初の終了ステータス。"),
           L.t("else 1 if any rank exited 1, else 0", "無ければ、どれかが 1 なら 1、ほかは 0")], sub_size=11)
    f.arrow(560, 267, 560, 328)
    f.step(400, 375, 4)
    f.box(760, 350, 290, 70, L.t("restart MPS", "MPS を起動し直す"), "nccl",
          [L.t("seed status 3 or more, or a rank", "シードの結果が 3 以上か、"), L.t("stopped by timeout or a signal (124+)", "ランクが止められた (124 以上) とき")], sub_size=11)
    f.arrow(712, 375, 758, 385)
    # Verdict.
    f.text(550, 485, L.t("Verdict per mode", "モードごとの判定"), size=15, weight="bold")
    f.step(440, 480, 5)
    f.box(60, 510, 470, 80, L.t("modes that keep I1", "I1 を守るモード"), "sim",
          [L.t("PASS when every seed is 0", "全シードが 0 なら PASS")], sub_size=13)
    f.box(570, 510, 470, 80, "violate-i1", "sig",
          [L.t("PASS when every seed is 0 or 1", "全シードが 0 か 1 で、"), L.t("and at least one is 1", "1 つ以上が 1 なら PASS")],
          sub_size=13)
    return f
