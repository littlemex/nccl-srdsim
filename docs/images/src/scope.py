# SPDX-License-Identifier: Apache-2.0
from svgkit import Fig

NAME = "scope"


def draw(L):
    f = Fig(1100, 430, L, L.t("What the simulator reproduces, and what it does not", "シミュレータが再現するものと、しないもの"))
    f.group(20, 55, 520, 350, L.t("reproduced", "再現する"), "sim")
    inside = [L.t("RMA through NCCL's GIN proxy backend", "NCCL の GIN プロキシバックエンドを通る RMA"),
              L.t("contract-legal orders among pending operations", "保留中の操作の、契約が許す順番"),
              L.t("orders that break it on purpose (violate-i1)", "わざと契約を破る順番 (violate-i1)"),
              L.t("puts, signals, strong signals and gets", "put、シグナル、強いシグナル、get"),
              L.t("host and device windows", "ホストとデバイスのウィンドウ")]
    for i, s in enumerate(inside):
        f.mark(50, 105 + i * 56, True)
        f.text(72, 110 + i * 56, s, size=14, anchor="start", limit=(20, 540))
    f.group(560, 55, 520, 350, L.t("not reproduced", "再現しない"), "neutral", dash=True)
    outside = [(L.t("collectives", "集団通信"), L.t("they use the network plugin interface", "ネットワークプラグインのインターフェースを使う")),
               (L.t("device-side GIN (GDAKI, EFA-GDA)", "デバイス側の GIN (GDAKI、EFA-GDA)"), L.t("no host on the path", "ホストを通らない")),
               (L.t("NIC queues, doorbells, PCIe order", "NIC のキュー、doorbell、PCIe の順序"), L.t("hardware", "ハードウェアの話")),
               (L.t("loss and duplication", "損失と重複"), L.t("SRD is reliable", "SRD は信頼できる")),
               (L.t("bandwidth and latency", "帯域とレイテンシー"), L.t("data goes via host memory", "ホストのメモリを通る"))]
    for i, (s, why) in enumerate(outside):
        f.mark(590, 105 + i * 56, False)
        f.text(612, 110 + i * 56, s, size=14, anchor="start", limit=(560, 1080))
        f.text(612, 128 + i * 56, why, size=12, anchor="start", color="#555555", limit=(560, 1080))
    return f
