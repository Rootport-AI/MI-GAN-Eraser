# CLAUDE.md — MI-GAN-Eraser 開発者ガイド

## プロジェクト概要

MI-GAN-Eraserは、AI画像インペインティング（不要物消去）を行うWindows向けWebアプリケーション。
Flask（Python）バックエンドとHTML5 Canvas フロントエンドで構成される。
推論にはMI-GAN（ICCV 2023）のONNXパイプラインを使用する。

元はLaMaEraserとして開発され、バックエンドをLaMa/PyTorchからMI-GAN/ONNX Runtimeに移行した。

## ディレクトリ構成

```
MI-GAN-Eraser/
├── app.py                      ... Flaskバックエンド（ONNX Runtime推論）
├── index.html                  ... フロントエンドHTML（Jinja2テンプレート）
├── main.js                     ... フロントエンドJS（930行、キャンバス操作全般）
├── style.css                   ... フロントエンドCSS
├── nouislider.min.css          ... スライダーUIライブラリCSS（サードパーティ）
├── nouislider.min.js           ... スライダーUIライブラリJS（サードパーティ）
├── migan_pipeline_v2.onnx      ... MI-GANモデル（29.5MB、リポジトリに同梱）
├── require.txt                 ... pip依存定義（7パッケージ）
├── setup.bat                   ... セットアップスクリプト（venv作成+pip+モデルコピー）
├── run.bat                     ... 起動スクリプト（venv起動+NVIDIA DLL PATH+app.py実行）
├── run_listen.bat              ... LAN公開起動（run.bat --listen のラッパー）
├── README.md                   ... ユーザー向けドキュメント
├── LICENSE                     ... MITライセンス
├── demo.gif                    ... デモ画像
├── .gitignore
├── images/
│   ├── defimg.png              ... デフォルト表示画像
│   └── favicon*.png            ... ファビコン
├── appfiles/                   ... setup.batが作成（.gitignore対象）
│   ├── venv/                   ... Python仮想環境
│   └── migan_pipeline_v2.onnx  ... setup.batがルートからコピー
└── output/                     ... 推論結果の一時保存先（.gitignore対象）
```

## アーキテクチャ

### Flask静的ファイル配信

app.pyはリポジトリルートで直接実行される。Flaskの初期化は:
```python
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='/static')
```
- `index.html` はルート直下のテンプレートとして `render_template('index.html')` で返す
- CSS/JS/画像は `{{ url_for('static', filename='style.css') }}` → `/static/style.css` でルート直下から配信
- `images/` サブディレクトリは `{{ url_for('static', filename='images/favicon64.png') }}` で参照

**注意**: index.html内のurl_forパスにサブディレクトリ（`css/`, `js/`）は付けない。ファイルはルート直下にある。

### APIエンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/` | index.htmlを返す |
| GET | `/output/<filename>` | 処理結果画像を返す |
| POST | `/process` | インペインティング処理を実行 |

POST /process:
- リクエスト: FormData（`image`: PNG Blob, `mask`: PNG Blob）
- 成功: `{"success": true, "result_url": "/output/tmpimg.png"}`
- エラー: `{"error": "メッセージ"}`, ステータス400または500

### コマンドライン引数

```
python app.py                    # localhost:7859（デフォルト）
python app.py --port 8080        # ポート変更
python app.py --listen           # LAN公開（0.0.0.0）、debug強制OFF
python app.py --debug            # デバッグモード（--listen時は無効）
```

## ONNX推論パイプライン

### モデル仕様（migan_pipeline_v2.onnx）

入力テンソル形状は **NCHW (rank 4)** であることに注意:

| 入出力 | 名前 | dtype | shape | 説明 |
|--------|------|-------|-------|------|
| 入力 | `image` | uint8 | (1, 3, H, W) | RGB, NCHW |
| 入力 | `mask` | uint8 | (1, 1, H, W) | 255=既知領域, 0=マスク領域 |
| 出力 | [0] | uint8 | (1, 3, H, W) | RGB, NCHW |

### 前処理フロー（app.py内）

1. image: cv2でBGR読み込み → RGB変換 → HWC (H,W,3) → transpose → NCHW (1,3,H,W)
2. mask: RGBA PNGとして読み込み → alphaチャンネル抽出 → 閾値処理（alpha>25→0, alpha<=25→255） → (H,W) → newaxis → (1,1,H,W)

### 後処理フロー

1. 出力 (1,3,H,W) → [0]でバッチ除去 (3,H,W) → transpose → HWC (H,W,3)
2. RGB→BGR変換 → cv2.imwriteでPNG保存

### マスク極性の注意

フロントエンドから送られるマスク: 塗った部分=alpha高い=消したい部分
MI-GANの規約: 255=既知（残す）, 0=マスク（消す）
→ バックエンドで **反転** する: alpha > 25 → 0, alpha <= 25 → 255

## フロントエンド（main.js）

930行のJavaScript。主な構成:
- **キャンバスレイヤー**: background-layer（元画像）, canvas-layer（マスク描画）, preview-layer（直線プレビュー）
- **描画ツール**: ブラシ(mode=1), 消しゴム(mode=0), バケツ塗り(mode=2)
- **座標変換**: transform.screenToWorld() でスクリーン座標→キャンバス座標変換
- **undo/redo**: 各状態を{mask, bg, width, height}のdataURLとして保持（最大30）
- **デフォルトブラシ色**: #00FF00（緑）。カラーパレットの選択肢は6色

バックエンドとの通信（823〜887行目付近）:
- backgroundLayerとcanvas-layerをそれぞれtoDataURL('image/png')でBlob化
- FormDataに`image`, `mask`として追加
- `fetch('/process', {method:'POST', body:formData})` で送信
- レスポンスの`data.result_url`から結果画像を取得しbackgroundLayerに描画

## セットアップ・起動の仕組み

### setup.bat
1. `appfiles/` ディレクトリ作成
2. `appfiles/venv/` にPython仮想環境を作成
3. pip upgrade → require.txtからパッケージインストール
4. `migan_pipeline_v2.onnx` をルートから `appfiles/` にコピー（既存ならスキップ）

### run.bat
1. `appfiles/venv/` をアクティベート
2. NVIDIA DLLパス（cudnn, cublas）をPATHに追加（GPU推論用）
3. カレントディレクトリをリポジトリルートに設定
4. `python app.py %*` を実行（引数はそのまま透過）

### NVIDIA DLLのPATH設定が必要な理由
onnxruntime-gpuはcudnn64_9.dll等を必要とするが、pipでインストールした
nvidia-cudnn-cu12のDLLはvenv内の `Lib/site-packages/nvidia/cudnn/bin/` に
配置される。run.batで明示的にPATHに追加しないとDLLが見つからず、
CUDAExecutionProviderが使えずCPUフォールバックする。

## 依存パッケージ（require.txt）

```
onnxruntime-gpu>=1.17.0    ... ONNX推論エンジン（CUDA+CPU）
nvidia-cudnn-cu12           ... cuDNN DLL（GPU推論に必要）
numpy>=1.23.5
Pillow>=10.0.0
opencv-python>=4.8.0
flask>=2.3.0
werkzeug>=2.3.0
```

nvidia-cudnn-cu12はnvidia-cublas-cu12を自動で依存解決する。

## 変更時の注意事項

### main.jsを変更する場合
- main.jsは仕様上「変更しない」方針だったが、ブラシ色変更で一部修正が入っている
- `/process`へのfetch呼び出し（823〜887行目）のFormDataフィールド名(`image`, `mask`)やレスポンス形式(`data.success`, `data.result_url`, `data.error`)はバックエンドと密結合しているので、変更時は双方を同期すること
- デフォルトブラシ色は3箇所に分散: 26行目（初期値）、684行目（0キーリセット）、816行目（リセットボタン）

### index.htmlを変更する場合
- 静的ファイル参照は `url_for('static', filename='ファイル名')` でルート直下を参照する
- `css/` や `js/` サブディレクトリは付けない（ファイルはルート直下に配置）
- `images/` サブディレクトリのファイルは `filename='images/xxx.png'` で参照

### ONNXモデルを更新する場合
- モデルの入出力テンソル形状（NCHW rank 4）を必ず確認すること
- 入力名が `image`, `mask` であることを確認すること
- マスクの極性（0=マスク, 255=既知）を確認すること
- モデルファイルはリポジトリルートに配置し、setup.batがappfiles/にコピーする

### setup.batを変更する場合
- 2回目以降の実行でもエラーにならないよう、既存ファイルチェックを入れること
- moveではなくcopyを使うこと（gitでdeleted扱いになるリスク回避）

## 既知の制限事項

- Undo/Redoは背景レイヤーとマスクレイヤーを一体で管理しているが、動作が不安定な場合がある
- MI-GANは内部で512x512にリサイズして推論するため、画像全体を覆うような大きなマスクでは品質が低下する
- Windows環境のみ対応（batファイル依存）
- `secure_filename` と `shutil` がimportされているが現在未使用（将来の拡張用に残存）

## ライセンス

- 本プロジェクト: MIT License (Rootport-AI)
- MI-GAN: MIT License (Picsart AI Research)
- noUISlider: MIT License
- Font Awesome: CC BY 4.0 (icons), MIT (CSS) — CDN経由で使用
