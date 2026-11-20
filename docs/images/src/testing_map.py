# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "testing-map"


def draw(L):
    cols = [L.t("scheduler", "スケジューラ"), L.t("transport", "トランスポート"), L.t("send queue", "送信キュー"),
            L.t("mapping", "マッピング"), L.t("plugin op table", "関数表"), L.t("plugin + NCCL", "NCCL と結合")]
    rows = [("test_srdsim_schedule", [0]), ("model_srdsim_schedule_exhaustive", [0]),
            ("test_srdsim_transport", [1]), ("model_srdsim_framing", [1]), ("test_srdsim_adversarial", [0, 1]),
            ("test_srdsim_async", [2]), ("test_srdsim_get", [1]), ("test_srdsim_mapping", [3]),
            ("test_srdsim_liveness", [0, 1]), ("test_plugin (tests/plugin)", [0, 1, 2, 4]),
            (L.t("CI: plugin builds and exports", "CI: プラグインのビルドとエクスポート"), [5]),
            (L.t("tests/gpu (on a GPU, by hand)", "tests/gpu (GPU で手動で実行する)"), [5])]
    f = Fig(1100, 120 + 38 * len(rows), L, L.t("What each test exercises", "どのテストが何を確かめるか"))
    x0, cw, y0 = 380, 115, 70
    for j, c in enumerate(cols):
        f.text(x0 + j * cw + cw / 2, y0, c, size=13, weight="bold")
    for i, (name, hit) in enumerate(rows):
        y = y0 + 20 + i * 38
        if i % 2 == 0:
            f.items.append(f'<rect x="20" y="{y}" width="1060" height="38" fill="#F5F5F5"/>')
        f.text(30, y + 25, name, size=13, anchor="start", mono=not name.startswith(("CI", "tests/gpu")) and (" " not in name or name.startswith("test_plugin")))
        for j in range(len(cols)):
            if j in hit:
                f.mark(x0 + j * cw + cw / 2, y + 19, True)
    yb = y0 + 20 + len(rows) * 38 + 22
    f.text(30, yb, L.t("Host tests and the plugin harness run on every pull request, also under ASan, UBSan and TSan; the GPU test is run by hand.",
                       "ホストのテストとプラグインのハーネスはプルリクエストのたびに走り、ASan、UBSan、TSan の下でも走る。GPU のテストは手動で実行する。"),
           size=12, anchor="start", color="#555555")
    return f
