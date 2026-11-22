# libfabric プロバイダ

[English](libfabric-provider.md) | **日本語**

`libsrdsim-fi.so` は、`efa` という名前で登録する libfabric のプロバイダである。aws-ofi-nccl や NIXL の libfabric バックエンドのように libfabric を通して EFA を使うコードは、プロバイダの名前が `efa` のときだけ EFA 向けの分岐に入る。このプロバイダを使うと、そのコードを 1 台のホストのプロセス同士で、SRD の配送規則のもとで動かせる。EFA インスタンスでデバイスを使わずに動かすことも、EFA の無いホストで動かすこともできる。設計と未決の点は [issue #75](https://github.com/littlemex/nccl-srdsim/issues/75) にある。

## 状態

| 部分 | 状態 |
|---|---|
| `efa` としての登録、選び方、`fi_getinfo` | 済み |
| fabric、domain、address vector、completion queue、endpoint の各オブジェクトと、EFA の endpoint オプション | 済み |
| プロセス間の send、receive、RMA write (RMA プラグインと同じ fabric モデルの上で) | まだ。転送の呼び出しはすべて `-FI_ENOSYS` を返す |
| immediate data つきの write、read、メモリ登録 (ホストと CUDA) | まだ |
| 1 台での aws-ofi-nccl と nccl-tests、NIXL の libfabric 転送テスト | まだ |

## ビルド

プロバイダには libfabric のヘッダとライブラリが要り、`pkg-config` で見つける。CI は `versions.env` で固定した libfabric のリリースに対してビルドする。EFA インスタンスでは、使う側が使っている libfabric (たいていは `/opt/amazon/efa` のもの) に対してビルドする。

```bash
export PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig
cmake -S . -B build -DSRDSIM_BUILD_FI_PROVIDER=ON
cmake --build build -j --target srdsim-fi
```

## 選び方

プロバイダを使うのは、`FI_PROVIDER_PATH` の優先指定の形 (`+` に続けてライブラリのパス) でこれを名指ししたプロセスだけである。libfabric はそのとき、本物を含む `efa` という名前のほかのプロバイダより、これを優先する。このため EFA インスタンスでも、そのプロセスにはエミュレートしたデバイスだけが見える。libfabric の既定のプロバイダのディレクトリには置かないので、名指ししないプロセスは本物のプロバイダを使う。

```bash
export FI_PROVIDER_PATH="+$PWD/build/libsrdsim-fi.so"
fi_info -p efa
```

`fi_info` は domain `srdsim0-rdm` を 1 つ表示する。

## 報告する内容

| 属性 | 値 | 理由 |
|---|---|---|
| endpoint | `FI_EP_RDM`、プロトコル `FI_PROTO_EFA`、アドレス形式 `FI_ADDR_EFA`、32 バイトのアドレス | 本物の RDM endpoint と同じ |
| 順序 | send と receive は `FI_ORDER_SAS`、RMA は無し | 本物のプロバイダは両側通信のメッセージをソフトウェアで並べ直す。SRD 自体は何も順序づけないので、RMA の順序を求める hints は断る |
| メモリ登録 | `FI_MR_VIRT_ADDR`、`FI_MR_ALLOCATED`、`FI_MR_PROV_KEY`、`FI_HMEM` のときは `FI_MR_HMEM`。`FI_MR_LOCAL` は求めない | 本物の RDM endpoint と同じ |
| CQ データ | 4 バイト | EFA で `fi_writedata` が運ぶ大きさ |
| 進行 | データも制御も手動 | 本物のプロバイダと同じ。自動進行を求める hints は断る |

endpoint のオプションには、libfabric の `prov/efa/src/rdm/efa_rdm_ep_fiops.c` と同じように答える。`FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES` は、本物の RDM endpoint と同じく断る。`FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES` は受け付け、`SRDSIM_FI_WRITE_IN_ORDER=0` ([設定](configuration.ja.md#libfabric-プロバイダ)) のときは断る。これで、使う側の「これを持たないデバイス」向けの経路も試せる。`FI_OPT_EFA_EMULATED_READ` と `FI_OPT_EFA_EMULATED_WRITE` は false を返す。

## テスト

`tests/fi/test_fi_provider.cc` は、aws-ofi-nccl と NIXL が組む hints を使って、libfabric を通してプロバイダに届く。CI は、自前の efa プロバイダつきでビルドした libfabric に対してこれを流す。あわせて、名指ししたときだけ `fi_info` がプロバイダを表示すること、write-after-write の順序を名乗るように壊したプロバイダに対してテストが失敗することを確かめる。

```bash
ctest --test-dir build -R fi_provider --output-on-failure
```
