# 音声ファイル設定手順

このディレクトリには、VirtualTrainerAppの音声フィードバック機能で使用する音声ファイルを配置する必要があります。

## 必要な音声ファイル

### 1. フォームエラー音声
- **ファイル名**: `zundamon_elbow_error.wav`
- **内容**: "肘が開きすぎなのだ"（ずんだもん音声）
- **用途**: エクササイズ中の肘エラー検出時に再生

### 2. 回数カウント音声
- **ファイル名**: `1.wav`, `2.wav`, `3.wav`, ..., `10.wav`
- **内容**: 各数字の読み上げ（ずんだもん音声）
- **用途**: エクササイズ回数完了時の音声フィードバック

## 音声ファイルの準備方法

### Option 1: VOICEVOXで生成
1. [VOICEVOX](https://voicevox.hiroshiba.jp/)をダウンロード・インストール
2. キャラクター「ずんだもん」を選択
3. 以下のテキストで音声を生成：
   - フォームエラー用: "肘が開きすぎなのだ"
   - 回数カウント用: "いち", "に", "さん", "し", "ご", "ろく", "しち", "はち", "きゅう", "じゅう"
4. WAVファイルとして書き出し

### Option 2: 既存ファイルをコピー
`AI_Model/sounds/`ディレクトリから該当ファイルをコピー：
```bash
# AI_Modelディレクトリから音声ファイルをコピー
cp ../../AI_Model/sounds/001_ずんだもん（ノーマル）_肘が開きすぎなのだ.wav ./zundamon_elbow_error.wav
cp ../../AI_Model/sounds/1.wav ./1.wav
cp ../../AI_Model/sounds/2.wav ./2.wav
# ... (3.wav〜10.wavも同様)
```

## Xcodeプロジェクトへの追加

1. Xcodeで VirtualTrainerApp プロジェクトを開く
2. Project Navigatorで `VirtualTrainerApp/Resources/Audio/` フォルダを右クリック
3. "Add Files to VirtualTrainerApp" を選択
4. 上記の音声ファイル（全11ファイル）を選択
5. "Add to target" で `VirtualTrainerApp` にチェックが入っていることを確認
6. "Add" をクリック

## ライセンス・著作権

使用する音声は **VOICEVOX（ずんだもん）** で生成されたものです：

- **音声合成ソフトウェア**: [VOICEVOX](https://voicevox.hiroshiba.jp/)
- **キャラクター**: ずんだもん
- **利用規約**: [VOICEVOX利用規約](https://voicevox.hiroshiba.jp/) に準拠
- **クレジット**: アプリ内設定画面にて適切にクレジット表示済み

## トラブルシューティング

### 音声が再生されない場合
1. 音声ファイルがXcodeプロジェクトに正しく追加されているか確認
2. ファイル名が正確か確認（大文字小文字含む）
3. 音声設定がONになっているか確認（アプリ内設定画面）

### 音声ファイルが見つからない場合
アプリログで以下のようなメッセージが表示されます：
```
[AudioFeedbackService] Audio file not found: zundamon_elbow_error.wav
[AudioFeedbackService] Rep count audio file not found: 1.wav
```

この場合、上記の手順に従って音声ファイルを正しく配置・追加してください。