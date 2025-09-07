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

## iOSアプリ (開発中)

- `VirtualTrainerApp` ディレクトリにXcodeプロジェクトが含まれています。
- `open VirtualTrainerApp/VirtualTrainerApp.xcodeproj` で開くことができます。