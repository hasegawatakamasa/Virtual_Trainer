# Virtual Trainer

バーチャルトレーナーは、AIを活用してリアルタイムで筋力トレーニングのフォームを分析し、フィードバックを提供するアプリケーションです。

## 機能

- **リアルタイム姿勢推定:** Webカメラ映像からあなたの骨格を検出し、動きを追跡します。
- **フォーム自動判定:** アームカールなどのエクササイズにおいて、肘の角度などを基に正しいフォームかを判定します。
- **レップカウント:** 正しいフォームでの繰り返し回数を自動でカウントします。
- **即時フィードバック:** 「Normal」や「Elbow Error」といった形で、フォームに対するフィードバックを画面上に表示します。

## AIモデルのセットアップ

### 1. 前提条件

- Python 3.8以上
- pip

### 2. リポジトリのクローン

```sh
git clone https://github.com/your-repo/Virtual_Trainer.git
cd Virtual_Trainer/AI_Model
```

### 3. 仮想環境の作成と有効化

プロジェクトルートに移動した後、仮想環境を作成して有効化します。これにより、プロジェクト固有のライブラリがグローバル環境に影響を与えるのを防ぎます。

```sh
# 仮想環境を作成 (例: venv)
python3 -m venv venv

# 仮想環境を有効化
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 4. 必要なライブラリのインストール

有効化した仮想環境内で、必要なライブラリをインストールします。

```sh
pip install -r requirements.txt
```

### 5. モデルファイルの配置

AIモデルを動作させるには、2つのモデルファイルが必要です。

1.  **姿勢推定モデル (YOLOv11n-Pose):**
    - [Ultralyticsのドキュメント](https://docs.ultralytics.com/ja/tasks/pose/#models) にアクセスし、`yolo11n-pose.pt` をダウンロードしてください。
2.  **フォーム判定モデル (GRU):**
    - 別途提供される `best_gru_model_v7_quantized.pth` を用意してください。

ダウンロード・用意した両方のファイル (`yolo11n-pose.pt` と `best_gru_model_v7_quantized.pth`) を、`AI_Model` ディレクトリ直下に配置してください。

### 6. 実行

すべてのファイルとライブラリが揃ったら、以下のコマンドでアプリケーションを起動します。

```sh
python main.py
```

Webカメラが起動し、リアルタイムでのフォーム判定が開始されます。ウィンドウを閉じるか、ターミナルで `q` キーを押すと終了します。

---

## iOSアプリ

`VirtualTrainerApp` ディレクトリにXcodeプロジェクトが含まれています。

### セットアップ手順

1. **Xcodeプロジェクトを開く:**
   ```bash
   open VirtualTrainerApp/VirtualTrainerApp.xcodeproj
   ```

2. **音声ファイルを準備:**
   
   音声フィードバック機能を使用するには、VOICEVOXで生成された音声ファイルが必要です。
   
   **必要なファイル:**
   - `zundamon_elbow_error.wav` - フォームエラー時の音声
   - `1.wav〜10.wav` - 回数カウント音声
   
   **準備方法:**
   - `AI_Model/sounds/`から対応ファイルをコピー、または
   - [VOICEVOX](https://voicevox.hiroshiba.jp/)で「ずんだもん」音声を生成
   
   **Xcodeへの追加:**
   1. `VirtualTrainerApp/Resources/Audio/` フォルダを右クリック
   2. "Add Files to VirtualTrainerApp" を選択
   3. 音声ファイル（全11ファイル）を選択して追加
   
   詳細な手順は `VirtualTrainerApp/Resources/Audio/README.md` をご確認ください。

### 機能

- **リアルタイム姿勢推定:** カメラ映像からの骨格検出・動き追跡
- **フォーム自動判定:** 肘の角度に基づくエクササイズフォーム評価
- **音声フィードバック:** フォームエラー時とレップカウント時の音声指導
- **レップカウント:** 正しいフォームでの繰り返し回数自動カウント
- **設定機能:** 音声ON/OFF、デバッグモード等

### ライセンス・クレジット

音声機能は **VOICEVOX（ずんだもん）** を使用：
- 音声合成: [VOICEVOX](https://voicevox.hiroshiba.jp/)
- キャラクター: ずんだもん
- 利用規約に準拠したクレジット表示を実装済み