# libfabric プロバイダ

[English](libfabric-provider.md) | **日本語**

`libsrdsim-fi.so` は、`efa` という名前で登録する libfabric のプロバイダである。aws-ofi-nccl や NIXL の libfabric バックエンドのように libfabric を通して EFA を使うコードは、プロバイダの名前が `efa` のときだけ EFA 向けの分岐に入る。このプロバイダを使うと、そのコードを 1 台のホストのプロセス同士で、SRD の配送規則のもとで動かせる。EFA インスタンスでデバイスを使わずに動かすことも、EFA の無いホストで動かすこともできる。設計と未決の点は [issue #75](https://github.com/littlemex/nccl-srdsim/issues/75) にある。

## 状態

| 部分 | 状態 |
|---|---|
| `efa` としての登録、選び方、`fi_getinfo` | 済み |
| fabric、domain、address vector、completion queue、endpoint の各オブジェクトと、EFA の endpoint オプション | 済み |
| プロセス間の send と receive (タグなしとタグつき)、RMA write、CQ データつき write、read を SRD の順序規則で運ぶ | 済み (`src/fi/srdsim_xfer.c`) |
| ホストメモリと CUDA メモリの登録 (プロバイダが決めるキーつき) | 済み。libfabric の登録マップを使う。CUDA には libfabric を `--with-cuda` で configure する必要がある |
| 1 台の GPU ホストで aws-ofi-nccl を通した NCCL (send/recv と RDMA の両プロトコル) | 動く。1 枚の GPU に 2 ランクで `all_reduce` が `srd` でも `inorder` でも正しい |
| 1 台での NIXL の libfabric バックエンド | 動く。NIXL 自身の libfabric 向け単体テスト (バックエンド、送り主の解決、レールの参照数) がこのプロバイダに対して合格し、トポロジ探索もエミュレートした NIC の PCIe アドレスと速度を読め、2 エージェント間の実データ転送も端から端まで通る |

## ビルド

プロバイダは、libfabric が自分のプロバイダを読み込み可能なライブラリとしてビルドするのと同じ方法でビルドする。`scripts/build_fi_provider.sh` が `src/fi` を libfabric のソースツリーに `prov/srdsim` として加え、プロバイダを DSO にしてその libfabric をビルドし、prefix の下にインストールする。プロバイダは hints の照合、fabric、domain、completion queue、address vector の各オブジェクト、メモリ登録に libfabric 自身のユーティリティ層を使うので、これらの呼び出しには libfabric のほかのプロバイダとまったく同じように答える。プロバイダは libfabric の既定のプロバイダのディレクトリの外、`PREFIX/lib/srdsim/` に移す。CI は `versions.env` で固定した libfabric のリリースに対してビルドする。

```bash
git clone --quiet --depth 1 --branch v2.7.0 https://github.com/ofiwg/libfabric.git libfabric-src
./scripts/build_fi_provider.sh libfabric-src "$HOME/srdsim-libfabric"
```

スクリプトには autoconf、automake、libtool、C コンパイラが要る。prefix の後ろに書いた configure のオプションは libfabric の `configure` に渡す。プロバイダは、一緒にビルドした libfabric とでしか動かしたことがない。

## 選び方

プロバイダを使うのは、`FI_PROVIDER_PATH` の優先指定の形 (`+` に続けてライブラリのパス) でこれを名指ししたプロセスだけである。libfabric はそのとき、本物を含む `efa` という名前のほかのプロバイダより、これを優先する。このため EFA インスタンスでも、そのプロセスにはエミュレートしたデバイスだけが見える。libfabric の既定のプロバイダのディレクトリには置かないので、名指ししないプロセスは本物のプロバイダを使う。

```bash
export LD_LIBRARY_PATH="$HOME/srdsim-libfabric/lib"
export FI_PROVIDER_PATH="+$HOME/srdsim-libfabric/lib/srdsim/libsrdsim-fi.so"
"$HOME/srdsim-libfabric/bin/fi_info" -p efa
```

`fi_info` は domain `srdsim0-rdm` を 1 つ表示する。

## 報告する内容

| 属性 | 値 | 理由 |
|---|---|---|
| endpoint | `FI_EP_RDM`、プロトコル `FI_PROTO_EFA`、アドレス形式 `FI_ADDR_EFA`、32 バイトのアドレス | 本物の RDM endpoint と同じ |
| 順序 | send と receive は `FI_ORDER_SAS`、RMA は無し | 本物のプロバイダは両側通信のメッセージをソフトウェアで並べ直す。SRD 自体は何も順序づけないので、RMA の順序を求める hints は断る |
| メモリ登録 | RMA には `FI_MR_VIRT_ADDR`、`FI_MR_ALLOCATED`、`FI_MR_PROV_KEY`。`FI_HMEM` のときは `FI_MR_HMEM`。hints が `FI_MR_LOCAL` を出したときはそれも返す。キーは 4 バイト | 本物の RDM endpoint と同じ。使う側が登録を出せるときはそれを使う |
| 能力 | メッセージ、タグつきメッセージ、RMA。`FI_HMEM` は求められたときだけ。atomic は無し | 本物のプロバイダは atomic をソフトウェアで模していて、このプロバイダが対象にするものはどれも atomic を使わない |
| CQ データ | 4 バイト | EFA で `fi_writedata` が運ぶ大きさ |
| 進行 | データも制御も手動 | 本物のプロバイダと同じ。自動進行を求める hints は断る |
| スレッド | hints が何も指定しなければ `FI_THREAD_SAFE`、指定があればその値。ただしドメインのオブジェクトは、答えが何であれ `FI_THREAD_SAFE` と同じようにロックする | 名乗る値は libfabric が hints 側に決めさせる (`ofi_alter_info`)。util 層はその 1 つの値だけを見てロックを選ぶ。`FI_THREAD_COMPLETION` か `FI_THREAD_DOMAIN` を求めると、この構成では 2 つのロックが外れる。完了のリング (`ofi_cq_init`) と、endpoint 自身のロック (`ofi_endpoint_init` が `FI_THREAD_SAFE` 以外のとき外す) である。endpoint のリストとアドレスベクタのロックは影響を受けない。あちらが外れるのは `FI_PROGRESS_CONTROL_UNIFIED` のときだけで、このプロバイダは制御の進行を手動と答えるため。この 2 つのモードでは使う側が CQ ごとの直列化を負う決まりで、純正のプロバイダはそこで止める (`tcp` で実測し、`efa` はソースで確認した。`efa` の CQ も `util_cq` である)。このプロバイダはそれでも守る。上のコードのバグを見つけるための道具なので、使う側が自分のスレッドの誤りを追うときに、まず道具の内部でリングが壊れていないかを確かめさせられては意味がないため |

この守りの代償も書いておく。`FI_THREAD_COMPLETION` を求めたうえで、自分が負うべき直列化を忘れた使う側は、ここでは正しく動いて実機で壊れる。守りが本物のロックなので、計測器にも見えない。プロバイダはドメインを守るときにモードを名指しした警告をログに出し、`SRDSIM_FI_GUARD_CQ=0` で libfabric の選択をそのままにできる。試験で確かめているのは `FI_THREAD_COMPLETION` のときの完了のリングである。`tests/fi/test_fi_cq_threads.cc` が 1 つのキューを 2 スレッドから読み、完了が必ず 1 回ずつ返ることを確かめ、CI は守りを外してもう一度流し、そのリングの中の競合として報告されて落ちることを確かめる。`FI_THREAD_DOMAIN`、endpoint のロック、その他のオブジェクトは同じ 1 つのフィールドから決まるが、専用の試験は無い。

endpoint のオプションには、libfabric の `prov/efa/src/rdm/efa_rdm_ep_fiops.c` と同じように答える。`FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES` は、本物の RDM endpoint と同じく断る。`FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES` は受け付け、`SRDSIM_FI_WRITE_IN_ORDER=0` ([設定](configuration.ja.md#libfabric-プロバイダ)) のときは断る。これで、使う側の「これを持たないデバイス」向けの経路も試せる。`FI_OPT_EFA_EMULATED_READ` と `FI_OPT_EFA_EMULATED_WRITE` は false を返し、`FI_OPT_EFA_USE_DEVICE_RDMA` を false にした後は true を返す。`FI_OPT_EFA_EMULATED_ATOMICS` は true を返す。`FI_HMEM` は、デバイスメモリを運べるようになるまでは、GPU の有無にかかわらず提供する。

## テスト

`tests/fi/test_fi_provider.cc` は、aws-ofi-nccl と NIXL が組む hints を使って libfabric を通してプロバイダに届き、libfabric の efa プロバイダが RDM endpoint について返すのと同じ答えを期待する。CI は、自前の efa プロバイダつきでビルドした libfabric に対してこれを流す。あわせて、名指ししたときだけプロバイダが読み込まれること、write-after-write の順序を名乗るように壊したプロバイダに対してテストが失敗することを確かめる。

CI はさらに、libfabric とプロバイダとこれらのテストを AddressSanitizer と ThreadSanitizer つきで建て直し、同じ一式をもう一度流す。この 2 周目が、プロバイダ自身のエージェントスレッドと使う側のスレッドの関係を見る場所で、計測器なしの 1 周目では何も見えない。手元で再現するときの注意が 2 つある。libfabric は `--disable-opx` で構成する必要がある。opx の `target_clones` な原子演算が `libfabric.so` の中で唯一の ifunc で、ローダがそれを実行ファイルの ThreadSanitizer の初期化より前に解決するため、計測済みのリゾルバが `main` に入る前に `ld.so` の中で落ちる。もう 1 つ、ThreadSanitizer は clang のものを使う。gcc のものは、このプロバイダが payload を公開するのに使う `atomic_thread_fence` を模擬できないと警告する。

```bash
export PKG_CONFIG_PATH="$HOME/srdsim-libfabric/lib/pkgconfig" LD_LIBRARY_PATH="$HOME/srdsim-libfabric/lib"
cmake -S . -B build -DSRDSIM_BUILD_TESTS=OFF -DSRDSIM_BUILD_FI_TEST=ON \
  -DSRDSIM_FI_PROVIDER="$HOME/srdsim-libfabric/lib/srdsim/libsrdsim-fi.so"
cmake --build build -j --target test_fi_provider
ctest --test-dir build -R fi_provider --output-on-failure
```
