# FdiTools (Python)

*[English](README.md) | 日本語*

周波数領域システム同定ツールボックス **FdiTools** の Python 移植版です
([WataruOhnishi/FdiTools](https://github.com/WataruOhnishi/FdiTools)(MATLAB 版・v3.0)の Python 版)。
MATLAB 版の API・アルゴリズムを [`numpy`](https://numpy.org/) /
[`scipy`](https://scipy.org/) / [`python-control`](https://python-control.readthedocs.io/)
の上で再現しています。

> 主要参考文献: R. Pintelon and J. Schoukens, *System Identification: A
> Frequency Domain Approach*, 2nd ed. Wiley-IEEE Press, 2012.

## インストール

**Python 3.9 以上**が必要です。依存は `numpy` / `scipy` / `python-control`
(描画には `matplotlib`)。**仮想環境(`venv`)**の利用を推奨します ―― このプロジェクト
専用の独立した Python で、他の Python 環境に干渉しません。所要時間は数分です。

> **Windows での注意:** `python` と打って Microsoft Store が開く場合、Python は
> 実際にはインストールされていません。[python.org](https://www.python.org/downloads/)
> から入れる(*“Add python.exe to PATH”* にチェック)か、
> `winget install Python.Python.3.12` を実行し、以降は下記の `py` ランチャーを使ってください。

### 1. コードを取得

```bash
git clone https://github.com/WataruOhnishi/FdiTools_Python.git
cd FdiTools_Python
```

### 2. 仮想環境の作成と有効化

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1        # プロンプトに (.venv) が付く
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> PowerShell で `Activate.ps1` の実行が拒否される場合は、一度
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を実行するか、有効化せずに
> 以降のコマンドで `.\.venv\Scripts\python.exe ...` を直接呼んでください。

### 3. ツールボックスのインストール

```bash
pip install -e .          # fditools + numpy / scipy / control
pip install -e ".[test]"  # さらに pytest + matplotlib(例とテストに必要)
```

`-e`(*editable*)はソースの変更が即反映されるモードで、開発時に便利です。
これで **どのディレクトリからでも** `import fditools` が使えるようになります
(MATLAB の `addpath` に相当 ―― [*import の仕組み*](#importの仕組みパスの話)参照)。

### 4. 動作確認

```bash
python -c "import fditools; print(fditools.__version__)"   # -> 0.2.0
pytest                                                     # 45 件パスするはず
python examples/step2_nonparametric_frf.py                 # 例を実行
```

### VS Code で使う

`FdiTools_Python` フォルダを開き、拡張機能「**Python**」(Microsoft)を入れ、
`Ctrl+Shift+P → Python: Select Interpreter` で `.venv` を選びます。以降、新しい
ターミナルでは自動で `.venv` が有効化され、▷ *実行*ボタンもそれを使います。

### importの仕組み(「パス」の話)

MATLAB の `addpath` と違い、フォルダを検索パスに足すのではありません。`pip install`
が **`fditools` パッケージ**を環境に登録するので、その venv 内ならどこからでも
`import fditools` が使えます(スクリプトごとの設定は不要)。

インストールしたくない場合は、リポジトリのルート(`fditools/` を**含む**フォルダ。
`fditools/` 自体ではない)を `sys.path` に足します:

```python
import sys
sys.path.insert(0, r"C:\path\to\FdiTools_Python")   # fditools/ を含むフォルダ
import fditools
```

いずれの場合も、venv はプロジェクト・マシンごとです。新しいクローンや別の PC では
手順 2–3 を再実行してください(`.venv` フォルダ自体は git に**コミットされません**)。

## Jupyter notebook で使う

ノートブックでも同じように使えます。プロジェクトの **`.venv` に**Jupyter を入れます
(`fditools` を見えるようにするため):

```bash
pip install jupyterlab ipykernel      # venv 有効化状態で一度だけ
jupyter lab                           # ブラウザで開く
```

または **VS Code** で `.ipynb` を作成し、右上の *Select Kernel* で `.venv` を選ぶだけ
(追加インストール不要)。

ノートブックでは図が**インライン表示**されます ―― `plt.show()` 不要・ウィンドウで
ブロックされません:

```python
import numpy as np, control, fditools as fdi

P0 = control.tf([(2*np.pi*120)**2], [1, 2*0.02*2*np.pi*120, (2*np.pi*120)**2])
harm = dict(fs=2000.0, df=1.0, fl=5.0, fh=400.0, fr=1.02)
ms = fdi.multisine(harm, control.tf([1], [1]),
                   dict(itp="r", ctp="c", dtp="f", gtp="q"))

u = np.tile(np.squeeze(ms.x[0, 0, :]), 6)
t = np.arange(u.size) / harm["fs"]
y = control.forced_response(P0, t, u).outputs

x, _ = fdi.pretreat(u, ms.nrofs, harm["fs"], 1, 0)
y, _ = fdi.pretreat(y, ms.nrofs, harm["fs"], 1, 0)
Pest = fdi.time2frf_ml(x, y, ms)
fig, _ = fdi.bode_fdi(Pest)            # インライン表示
```

> ノートブックから `examples/` のヘルパー(例:`from _data import benchmark_plant`)を
> 使うときは、先にそのフォルダをパスに足してください:
> `import sys; sys.path.insert(0, "examples")`。

## トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| `python` と打つと Microsoft Store が開く | 本物の Python が `PATH` にない。`py` か venv の `.\.venv\Scripts\python.exe` を使う。または *設定 → アプリ → アプリの詳細設定 → アプリ実行エイリアス → `python.exe`/`python3.exe`* をオフにする。 |
| `ModuleNotFoundError: No module named 'fditools'` | 有効なインタープリタが `pip install -e .` を実行した環境と違う。venv を有効化(または VS Code で `.venv` を選択)して `pip install -e .` を再実行。 |
| PowerShell: *“Activate.ps1 cannot be loaded … running scripts is disabled”* | 一度 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を実行するか、有効化せず `.\.venv\Scripts\python.exe` を直接呼ぶ。 |
| スクリプトが「固まる」/ 次のコマンドを受け付けない | `plt.show()` が**図ウィンドウを閉じるまでブロック**します。ウィンドウを閉じる(Ctrl+C ではなく)か、`FDI_NOSHOW=1` で PNG 保存のみにする。 |
| 例の実行で `ModuleNotFoundError: No module named '_data'` | `python examples/<name>.py` の形で実行する(スクリプト自身のフォルダが `_data`/`_plot` を import 可能にする)。 |
| `FileNotFoundError: ident_python.mat not found` | 通常は起きません(変換済みモデルをリポジトリに同梱)。削除した場合のみ発生 ―― MATLAB で再生成(`cd MATLAB/Examples/private; convert_ident_to_python`)。チュートリアルは `benchmark_plant()` で合成プラントにフォールバックもします。 |
| 図ウィンドウが全く出ない | `matplotlib` が入っているか(`pip install matplotlib`)、**`.py` ファイル**を ▷ *Run Python File* で実行しているか確認(「Run Selection in Interactive Window」ではない)。ノートブックでは図はインライン。 |
| `pytest` が *“could not create cache path … [WinError 123]”* を警告 | 無害です ―― リポジトリのルートから `pytest` を実行するか無視してください。テストは通ります。 |

## モジュール対応表

| MATLAB フォルダ | Python サブパッケージ | 内容 |
|---|---|---|
| `1_ExcitationDesign` | `fditools.excitation` | `multisine`, `sweptsine`, `prbs`, `multisine2hdr`, 位相設計補助 |
| `2_NonparametricFRF` | `fditools.nonparametric` | `pretreat`, `time2frf_ml`, **`time2frf_lpm`**, `time2frf_h1`, `time2frf_log`, `splinefit` |
| `3_NonlinearDistortions` | `fditools.nonlinear` | `time2bla`, **`time2bla_mimo`**, `time2nld` |
| `4_ParametricEstimation` | `fditools.parametric` | `lsfdi`, `wlsfdi`, `nlsfdi`, `mlfdi`, `gtlsfdi`, `btlsfdi`, `ssfdi`, **`frf2modal`** |
| `5_SelectionValidation` | `fditools.validation` | `chi2test`, `costtest`, `residtest` |
| `A_CalculationAuxiliary` | `fditools.auxiliary` | `ba2theta`, `theta2ba`, `ba2hm`, `hm2ba`, `hfrf`, `cr_rao`, **`frfconf`**, `f2t`, `t2f`, `dbm`, `phs`, `fdel_fdi`, `fcat_fdi`, `fdicohere`, `bode_fdi` |

すべてトップレベルでも再エクスポートされます(例:`import fditools as fdi; fdi.multisine(...)`)。

### v3.0 の新機能

* **`time2frf_lpm`** ―― 局所多項式法(LPM)。FRF と**過渡**を同時推定するため、短く
  過渡を含む記録でも低バイアスな FRF が得られます(周期/ブロードバンド、SISO/SIMO、
  直交多重正弦波 MIMO)。
* **`frf2modal`** ―― 構造化(rank-1 残差)MIMO モーダル同定。比例/一般(粘性)減衰に
  対応し、モーダルパラメータと実係数の `control.StateSpace` を返します。
* **MIMO** に `time2frf_ml` / `time2frf_lpm` / `time2bla_mimo` が対応。直交多実験(または
  単一 zip)多重正弦波で、実験を `(N, nch, ne)` 配列として渡します。
* **`frfconf`** ―― 信頼半径係数(PS2012 eq.2-40)。**`bode_fdi`** に不確かさの
  `line`/`band` 表示を追加。
* FRF の標準偏差は `UserData.sG`(`= sqrt(2)*sCR`、旧 `sGhat`)になりました。
  `UserData.nrofp`(平均周期数)と `UserData.method` も保持します。

## リポジトリ構成

```
fditools/        Python パッケージ(本体)
tests/           pytest テスト一式
examples/        Python サンプル(Step 1–5 + チュートリアル)
docs/img/        この README に載せる図
pyproject.toml   Python パッケージング / 依存関係
MATLAB/          元の MATLAB ツールボックス(参照用)
  src/             MATLAB ソース関数
  Contents.m       MATLAB ツールボックス目次
  Examples/        MATLAB サンプル + 計測データ(private/*.mat)
  README.md        元の MATLAB README
```

Python が主実装で、MATLAB 関連はすべて `MATLAB/` 配下にあります。

## クイックスタート

```python
import numpy as np
import control
import fditools as fdi

# 真のプラント(デモ用)
P0 = control.tf([(2*np.pi*120)**2], [1, 2*0.02*2*np.pi*120, (2*np.pi*120)**2])

# 1) 準対数(quasi-log)多重正弦波を設計
harm = dict(fs=2000.0, df=1.0, fl=5.0, fh=400.0, fr=1.02)
options = dict(itp="r", ctp="c", dtp="f", gtp="q")
ms = fdi.multisine(harm, control.tf([1], [1]), options)

# 実験: プラントに 6 周期入力
u = np.tile(np.squeeze(ms.x[0, 0, :]), 6)
T = np.arange(u.size) / harm["fs"]
y = control.forced_response(P0, T, u).outputs

# 2) 非パラメトリック FRF(最尤法)
xp, _ = fdi.pretreat(u, ms.nrofs, harm["fs"], 1, 0)
yp, _ = fdi.pretreat(y, ms.nrofs, harm["fs"], 1, 0)
Pest = fdi.time2frf_ml(xp, yp, ms)          # -> fditools.FrfData

# 3) パラメトリック推定
n, mh, ml = 2, 0, 0
Hml, Hls = fdi.mlfdi(Pest, n, mh, ml, 500, 1e-10, 0, "c")
Hbtls, Hgtls = fdi.btlsfdi(Pest, n, mh, ml, 1.0, 500, 1e-10, "c")
sys_ml = control.tf(Hml[0, 0])              # control.TransferFunction
```

実行可能な一連のデモ(任意で描画)は
[`examples/tutorial_1_qlog.py`](examples/tutorial_1_qlog.py) にあります。

## サンプル

すべてのサンプルは [`examples/`](examples/) にあり、図をスクリプトの隣に PNG 保存し、
対話ウィンドウを開きます(`FDI_NOSHOW=1` で PNG 保存のみ)。元の MATLAB スクリプト
([`MATLAB/Examples/`](MATLAB/Examples))の移植です。

```bash
python examples/step2_nonparametric_frf.py      # 任意のサンプルを実行
```

### Step 1–5 ワークフロー

MATLAB `Step_1`–`Step_5` の移植。元のモータベンチ計測データ
(`MATLAB/Examples/private/*.mat`、SciPy で直接読込 ―― **MATLAB 不要**)で動作します。

**Step 1 — 励振設計** &nbsp;(`step1_excitation_design.py` ← `Step_1_ExcitationDesign.m`): multisine / PRBS / 掃引正弦。

![Step 1 multisine](docs/img/step1_multisine.png)

**Step 2 — 非パラメトリック FRF** &nbsp;(`step2_nonparametric_frf.py` ← `Step_2_NonparametricFRF.m`): 最尤 FRF、モータ側・負荷側(`MultisineTypeA.mat`)。

![Step 2 motor](docs/img/step2_frf_motor.png)
![Step 2 load](docs/img/step2_frf_load.png)

**Step 3 — 非線形歪み** &nbsp;(`step3_nonlinear_distortions.py` ← `Step_3_NonlinearDistortions.m`): 線形 / 偶数次 / 奇数次 / ノイズの分離(`MultisineTypeB.mat`)。

![Step 3](docs/img/step3_nonlinear.png)

**Step 4 — パラメトリック推定** &nbsp;(`step4_parametric_estimation.py` ← `Step_4_ParametricEstimation.m`): SIMO の決定論的(WLS/NLS)・確率的(ML/BTLS)推定。

![Step 4 deterministic](docs/img/step4_deterministic.png)
![Step 4 stochastic](docs/img/step4_stochastic.png)

**Step 5 — 選択と検証** &nbsp;(`step5_selection_validation.py` ← `Step_5_SelectionValidation.m`): 残差白色性・コスト関数・カイ二乗検定。

![Step 5 residuals](docs/img/step5_residuals.png)
![Step 5 cost](docs/img/step5_cost.png)
![Step 5 chi2](docs/img/step5_chi2.png)

### チュートリアル

MATLAB `Tutorial_*` の移植。ベンチマークプラント `mdl.Pv(1,1)` を同定します。
変換済みモデル(`MATLAB/Examples/private/ident_python.mat`)は**リポジトリに同梱**して
あるので、本物のプラントですぐ動きます ―― **MATLAB 不要**。(そのファイルが無い場合は
合成プラントにフォールバックし、その旨を表示します。)

**Tutorial 1 — ランダム雑音** &nbsp;(`tutorial_1_random.py`): Welch 法 FRF(SciPy)+ NLS フィット。

![Tutorial 1 random](docs/img/tutorial_1_random.png)

**Tutorial 1 — 掃引正弦** &nbsp;(`tutorial_1_chirp.py`): 周期 H1 + NLS フィット。

![Tutorial 1 chirp](docs/img/tutorial_1_chirp.png)

**Tutorial 1 — 準対数多重正弦波** &nbsp;(`tutorial_1_qlog.py`): 全推定器の比較。

![Tutorial 1 qlog](docs/img/tutorial_1_qlog.png)

**Tutorial 2 — 反復(experiment 結合)** &nbsp;(`tutorial_2_iterative.py`): 3 実験を `fcat_fdi`/`fdel_fdi` で結合。

![Tutorial 2 iterative](docs/img/tutorial_2_iterative.png)

**Tutorial 3 — 入力非線形** &nbsp;(`tutorial_3_nonlinear_in.py`)。

![Tutorial 3 input](docs/img/tutorial_3_nonlinear_in.png)

**Tutorial 3 — 出力非線形** &nbsp;(`tutorial_3_nonlinear_out.py`): Simulink モデル `model_nl_out.slx`(出力を多項式で帰還)を状態空間 ODE で等価再現。

![Tutorial 3 output](docs/img/tutorial_3_nonlinear_out.png)

### v3.0 の機能

**LPM** &nbsp;(`tutorial_lpm.py`): 短く過渡を含む記録に対し、局所多項式法は過渡を
モデル化して低バイアスを保つ一方、素の ML(過渡除去なし)は漏れ込みで偏ります。

![LPM tutorial](docs/img/tutorial_lpm.png)

**MIMO + モーダル** &nbsp;(`tutorial_4_mimo.py`): 直交 2 入力多重正弦波で 2×2 プラントを
`time2frf_ml`/`time2frf_lpm` で同定し、`frf2modal` でモーダルパラメータと実状態空間
モデルを復元。

![MIMO tutorial](docs/img/tutorial_4_mimo.png)

### ベンチマークモデル `20160829_ident.mat`

元の `20160829_ident.mat` は MATLAB の*制御オブジェクト*(`mdl.Pv` は 2×1 `zpk`、`mdl.Pp`
も)を保持しており SciPy では読めません。SciPy で読める状態空間ファイル
**`MATLAB/Examples/private/ident_python.mat`** に変換済みで、**リポジトリにコミット**
してあるため、MATLAB なしで一通り動かせます。

このファイルを**再生成**したいとき(元モデルを変更した場合など)だけ MATLAB が必要です:

```matlab
>> cd MATLAB/Examples/private
>> convert_ident_to_python      % ident_python.mat を書き出す
```

プラントは直接ロードできます:

```python
from examples._data import load_ident, benchmark_plant
P0 = load_ident("Pv", (0, 0))   # control.StateSpace == MATLAB の mdl.Pv(1,1)
P0, label = benchmark_plant()   # 変換済みなら本物、無ければ合成
```

## API の規約

* **伝達関数配列**(`Hm`)は SISO `control.TransferFunction` の 2 次元 `object` ndarray
  で、`Hm[o, i]`(出力, 入力)で参照 ―― MATLAB の `Hm(o, i)` に対応。
* **`FrfData`** は MATLAB の `frd` + `UserData` に相当。`Pest.freq`(Hz)、
  `Pest.response`(`(nrofo, nrofi, nroff)`)、`Pest.userdata`(`.X`, `.Y`, `.sX2`,
  `.sY2`, `.cXY`, `.sCR`, `.sG`, `.nrofp`, `.FRFn`, `.ms` …)。`(nroff, nrofh)` 行列は
  `Pest.frf_columns()`、本物の `control.FrequencyResponseData` は `Pest.to_frd()`。
* **2 通りの呼び出し**: 反復推定器は `FrfData`(構造化)でも古典的な位置引数列でも
  受け付けます(MATLAB と同様)。
* **モデル次数** `mh`/`ml`: スカラや平坦なリストは伝達関数 1 つ分、`(nrofo, nrofi)`
  配列は MATLAB の列優先規約に従います。
* **検証テストの `SYS`** は「モデル名 → 伝達関数配列」の `dict`(MATLAB の struct 相当)。
* **MIMO**(v3.0): `time2frf_ml`/`time2frf_lpm` に実験を `(N, nch, ne)` 配列で渡すと、
  結果は `(ny, nu, nl)` の `FrfData`(`UserData.sG` は `(ny, nu, nl)`、`UserData.method`
  は `'orthogonal'`/`'zippered'`/`'lpm'`)。`time2bla_mimo` は `dict` を返します。
* **`bode_fdi`**(v3.0 シグネチャ): `bode_fdi(sys, unc=..., sigma=..,
  style='line'|'band', legend=[...], ...)` ―― `unc` は UserData のフィールド名、
  `(freq, mag)` の組、または振幅ベクトル。

## MATLAB 版との既知の差異

* **ランダム位相**(`randph`)は NumPy の Mersenne-Twister を使用。Python 内では
  再現可能ですが、MATLAB の `rng` と**ビット一致はしません**。
* **`msinl2p`** は `multisine` が使う入力のみのクレストファクタ最小化を移植。追加
  ハーモニクス(snow, `Fa`)と入出力(`H`)分岐は未移植で `NotImplementedError`。
* **`splinefit`** は SciPy ベースの最小二乗スプライン。原典のロバスト反復・微分拘束は未移植。
* **`ssfdi`** は MATLAB の「work in progress」関数の移植(対話的な次数入力は必須引数
  `order` に置換)。
* **`gtlsfdi`/`btlsfdi`** は原典の `try chol(A)…catch` 挙動(`catch` 側が常に実行)を
  忠実に再現し、結果が MATLAB と一致します。
* **描画**(`bode_fdi`)は `matplotlib` が必要で、遅延 import します。
* **`frf2modal`** の比例減衰は実モード形状で一貫実装(比例減衰の正しい構造)。一般減衰は
  MATLAB 同様に複素モード形状を保持します。
* MIMO LPM は**直交**多実験(鋭い共振向け)と単一**zip**実験(`u` が `(N, nu)`、
  per-channel 分解能 1/nu ―― 平滑/熱系で 1 実験が複数入力を兼ねる場合に便利)の両方に対応。
* MATLAB の **`iodata`** OO コンテナは意図的に再現していません。MIMO は `(N, nch, ne)`
  (直交)または `(N, nu)`(zip)配列を渡す関数形式で扱います。

## ライセンス

元の FdiTools プロジェクトに準じます([LICENSE](LICENSE) 参照)。
