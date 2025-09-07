# Virtual_Trainer


## セットアップ手順
### 1. モデルファイルのダウンロード
#### YOLO11n-poseモデルを以下のページからダウンロードしてください。
#### https://docs.ultralytics.com/ja/tasks/pose/#models
### 2. ファイル配置
#### ダウンロードした yolo11n-pose.pt ファイルを main.py と同じディレクトリに配置してください。



# 1. リポジトリをクローン
```sh
git clone https://github.com/your-repo/Virtual_Trainer.git
cd Virtual_Trainer
```

# 2. AIモデルの環境構築
cd AI_Model
pip install -r requirements.txt

# 3. iOSアプリのセットアップ
cd ../VirtualTrainerApp
open VirtualTrainerApp.xcodeproj # Xcodeでプロジェクトを開く
