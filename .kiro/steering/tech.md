# Technology Stack - 推しトレ

## Architecture Overview

推しトレは**デュアルプラットフォーム**アーキテクチャを採用し、Python版とiOS版の2つの実装を提供します：

- **Python版**: デスクトップ環境での高精度分析とプロトタイピング
- **iOS版**: モバイルデバイスでのリアルタイムオンデバイス推論

## Core AI/ML Technology

### Computer Vision Models
- **YOLO11n-pose**: リアルタイム17ポイント姿勢推定
- **GRU (Gated Recurrent Unit)**: 時系列フォーム分類モデル
- **Core ML**: iOS向けオンデバイス推論エンジン

### Model Conversion Pipeline
- **統合変換スクリプト**: `AI_Model/convert_models.py` - YOLOとGRU両モデル対応の統合変換スクリプト
- **量子化対応**: `best_gru_model_v7_quantized.pth` - 軽量化済みGRUモデル

**Note**: 以前は複数の変換スクリプト（convert_models_onnx.py、convert_models_system.py等）が存在しましたが、現在は`AI_Model/convert_models.py`に統合されています。

## Python Implementation

### Core Dependencies
```bash
# AI/ML Framework
torch==2.8.0              # PyTorch for deep learning
torchvision==0.23.0       # Computer vision utilities
ultralytics==8.3.195      # YOLO implementation
ultralytics-thop==2.0.17  # YOLO model profiling and optimization

# Computer Vision
opencv-python==4.12.0.88  # Image processing and camera handling
numpy==2.2.6              # Numerical computing
pillow==11.3.0            # Image manipulation

# Data Processing
polars==1.33.0            # High-performance data frames
scipy==1.16.1             # Scientific computing
matplotlib==3.10.6        # Visualization

# HTTP and Requests
requests==2.32.5          # HTTP library for web requests
urllib3==2.5.0            # HTTP client library
certifi==2025.8.3         # Certificate verification

# System and Utilities
psutil==7.0.0             # System monitoring and resource management
setuptools==80.9.0        # Python package management
PyYAML==6.0.2             # YAML configuration parsing
```

### Development Environment
- **Python Version**: 3.8+
- **Virtual Environment**: `venv` (recommended)
- **Package Manager**: `pip`
- **IDE Support**: VS Code, PyCharm

### Common Commands
```bash
# Environment Setup
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Dependencies Installation
pip install -r AI_Model/requirements.txt

# Model Conversion
python AI_Model/convert_models.py       # Unified model conversion script

# Testing Scripts (for model validation)
python AI_Model/test_overheadpress.py   # Test overhead press form detection
python AI_Model/test_squat.py           # Test squat form detection
python AI_Model/test_sideraise.py       # Test side raise form detection

# Application Launch (Exercise-Specific)
python AI_Model/main_overheadpress.py   # オーバーヘッドプレス
python AI_Model/main_squat.py           # スクワット
python AI_Model/main_sideraises.py      # サイドレイズ
python AI_Model/main_pushup.py          # 腕立て伏せ
```

## iOS Implementation

### iOS Technology Stack
- **Platform**: iOS 16+ (with iOS 18 optimization)
- **Development Environment**: Xcode 16.1
- **Language**: Swift 5.9
- **UI Framework**: SwiftUI
- **Concurrency**: Swift Concurrency (async/await)

### Core Frameworks
```swift
// UI and Application Framework
import SwiftUI                    // Declarative UI framework
import Combine                   // Reactive programming

// AI and Computer Vision
import CoreML                    // On-device machine learning
import Vision                    // Computer vision framework

// Camera and Media
import AVFoundation              // Audio/video capture and processing
import CoreImage                 // Image processing

// Data Persistence
import CoreData                  // Object graph and persistence framework

// System Integration
import MetricKit                 // Performance monitoring
import Charts                    // Native SwiftUI charts for progress visualization
import UserNotifications         // Local notification delivery and management
import BackgroundTasks           // Background processing and calendar sync
import Security                  // Keychain services for secure token storage
import EventKit                  // Calendar access (future: native calendar integration)
import AuthenticationServices    // OAuth 2.0 web authentication flow
```

## Audio Feedback System

### VOICEVOX Integration
- **Multi-Character System**: ずんだもん・四国めたん の2キャラクター対応
- **Voice Synthesis**: VOICEVOX-generated audio files for each character
- **Audio Files**: WAV format, キャラクター別サブディレクトリ構成でapp bundleに格納
- **Content**: Japanese voice feedback for form correction, rep counting, and speed guidance
- **Character Selection**: Runtime character switching with persistent user preferences  
- **License Compliance**: Character-specific VOICEVOX attribution and licensing

### iOS Audio Architecture
```swift
// Audio Playback Management
import AVFoundation              // AVAudioPlayer and AVAudioSession
- AudioFeedbackService           // Centralized audio management
- AVAudioSession configuration   // .playback category with .mixWithOthers
- Cooldown timing control        // Prevent audio spam (3s intervals)
- Exercise zone awareness        // Context-aware audio triggering
```

### Resource File Structure
```
Resources/
├── Audio/
│   ├── ずんだもん/                        // ずんだもん character audio files
│   │   ├── zundamon_form_error.wav           // Form error feedback
│   │   ├── zundamon_fast_warning.wav         // Speed warning
│   │   ├── zundamon_slow_encouragement.wav   // Speed encouragement
│   │   ├── zundamon_count_01.wav - 10.wav    // Rep counting (1-10)
│   │   ├── zundamon_100_reps.wav             // 100回達成祝福
│   │   ├── zundamon_500_reps.wav             // 500回達成祝福
│   │   ├── zundamon_1000_reps.wav            // 1000回達成祝福
│   │   ├── zundamon_streak_3days.wav         // 3日連続達成
│   │   ├── zundamon_streak_7days.wav         // 7日連続達成
│   │   ├── zundamon_streak_30days.wav        // 30日連続達成
│   │   ├── zundamon_personal_best.wav        // 自己記録更新
│   │   ├── zundamon_new_record.wav           // 新記録達成
│   │   ├── zundamon_level_up.wav             // レベルアップ
│   │   ├── zundamon_bond_max.wav             // 好感度MAX
│   │   ├── zundamon_comeback.wav             // カムバック歓迎
│   │   ├── zundamon_share_prompt.wav         // SNSシェア促進
│   │   └── zundamon_waiting.wav              // 待機中音声
│   ├── 四国めたん/                        // 四国めたん character audio files  
│   │   ├── shikoku_form_error.wav            // Form error feedback
│   │   ├── shikoku_fast_warning.wav          // Speed warning
│   │   ├── shikoku_slow_encouragement.wav    // Speed encouragement
│   │   ├── shikoku_count_01.wav - 10.wav     // Rep counting (1-10)
│   │   ├── shikoku_100_reps.wav              // 100回達成祝福
│   │   ├── shikoku_500_reps.wav              // 500回達成祝福
│   │   ├── shikoku_1000_reps.wav             // 1000回達成祝福
│   │   ├── shikoku_streak_3days.wav          // 3日連続達成
│   │   ├── shikoku_streak_7days.wav          // 7日連続達成
│   │   ├── shikoku_streak_30days.wav         // 30日連続達成
│   │   ├── shikoku_personal_best.wav         // 自己記録更新
│   │   ├── shikoku_new_record.wav            // 新記録達成
│   │   ├── shikoku_level_up.wav              // レベルアップ
│   │   ├── shikoku_bond_max.wav              // 好感度MAX
│   │   ├── shikoku_comeback.wav              // カムバック歓迎
│   │   ├── shikoku_share_prompt.wav          // SNSシェア促進
│   │   └── shikoku_waiting.wav               // 待機中音声
│   └── README.md                              // Setup instructions
└── Image/
    ├── OshinoAi/                          // 推乃 藍（デフォルトトレーナー）character images
    │   └── normal.png                         // 推乃 藍 character portrait
    ├── ずんだもん/                        // ずんだもん character images
    │   └── zundamon_1.png                    // ずんだもん character portrait
    └── 四国めたん/                        // 四国めたん character images (planned)
        └── shikoku_metan_1.png               // 四国めたん character portrait (to be added)
```

### Key iOS Components

#### Core Services
- **MLModelManager**: Core ML モデル管理とオンデバイス推論
- **CameraManager**: AVFoundation カメラセッション管理
- **FormAnalyzer**: リアルタイムフォーム分析エンジンと種目別設定対応
- **RepCounterManager**: 自動回数カウント状態機械
- **AudioFeedbackService**: マルチキャラクター音声フィードバック管理（フォームエラー・回数カウント・速度フィードバック・アチーブメント）
- **SpeedAnalyzer**: 動作速度分析とリアルタイム速度フィードバック
- **VoicePreviewService**: ランダム音声プレビュー再生（回数カウント・フォームエラー・速度フィードバック等からランダム選択）とサンプル管理

#### Data Persistence Services
- **CoreDataManager**: Core Data スタック管理と永続化調整
- **TrainingSessionService**: トレーニングセッション記録と履歴管理
- **AchievementSystem**: アチーブメント判定と解除ロジック
- **OshiReactionManager**: 推しキャラクターリアクション管理と好感度システム

#### Calendar and Notification Services
- **GoogleCalendarAuthService**: OAuth 2.0認証フローとトークン管理（ASWebAuthenticationSession使用）
- **GoogleCalendarAPIClient**: Google Calendar API通信クライアント（イベント取得・解析）
- **CalendarEventAnalyzer**: カレンダーイベント解析と隙間時間検出ロジック
- **CalendarSyncCoordinator**: カレンダー同期調整とバックグラウンド更新管理
- **CalendarPrivacyManager**: プライバシー準拠のカレンダーデータ処理
- **OshiTrainerNotificationService**: 推しトレーナー通知作成と配信（UNUserNotificationCenter統合）
- **NotificationScheduler**: 通知スケジューリングと時間帯フィルタリング
- **NotificationAnalyticsService**: 通知効果測定とタップ率・実施率の追跡
- **NotificationSettingsManager**: 通知設定管理（頻度・時間帯・曜日カスタマイズ）
- **KeychainManager**: OAuth トークンのセキュアストレージ管理（kSecAttrAccessibleWhenUnlockedThisDeviceOnly）

#### Resource Management System
- **ResourceCleanupCoordinator**: システムリソース管理と統合クリーンアップ調整
- **IntegratedCleanupService**: カメラセッション・音声リソースの統合クリーンアップサービス
- **ResourceCleanupError**: リソース管理エラーハンドリング専用エラー型

#### UI Components
- **KeypointOverlayView**: COCO-Pose 17ポイント可視化
- **ExerciseSelectionView**: 種目選択画面とナビゲーション管理
- **ExerciseCardView**: 種目表示カードコンポーネント
- **FutureExpansionBanner**: 将来拡張予定種目バナーコンポーネント（近日公開メッセージ表示）
- **QuickPreviewOverlay**: 近日公開種目のクイックプレビューオーバーレイ（詳細情報・カロリー・難易度表示）
- **OshiTrainerSettingsView**: 推しトレーナー選択とプレビュー画面
- **VoiceCharacterSettingsView**: 音声キャラクター選択とプレビュー画面（レガシー）
- **SwipableTrainerSelectionView**: スワイプ対応トレーナー選択カルーセル
- **SwipableCharacterSelectionView**: スワイプ対応キャラクター選択カルーセル（レガシー）
- **TrainerImageView**: トレーナー画像表示コンポーネント with async loading
- **CharacterImageView**: キャラクター画像表示コンポーネント with async loading（レガシー）
- **VoicePreviewButton**: 音声プレビュー再生ボタンコンポーネント
- **TrainerSelectionSuccessMessage**: トレーナー選択成功メッセージ表示
- **HintView**: ヒント・ガイダンス表示コンポーネント
- **CreditDisplayView**: VOICEVOXクレジット表示
- **LiveAudioTextView**: リアルタイム音声フィードバックテキスト表示
- **RecordsTabView**: トレーニング記録閲覧タブビュー
- **ProgressVisualizationView**: 進捗グラフとチャート表示
- **WeeklyChartView**: 週間トレーニング量チャート
- **SettingsView**: 統合設定画面（通知・カレンダー・トレーナー設定）
- **CalendarSettingsView**: Googleカレンダー連携設定と認証管理
- **NotificationSettingsView**: 通知頻度・時間帯・曜日のカスタマイズUI
- **NotificationStatsView**: 通知効果統計表示（タップ率・実施率）

#### Data Models
- **OshiTrainer**: 推しトレーナーデータモデル（性格・口調・音声・画像）
- **OshiTrainerSettings**: トレーナー選択管理とUserDefaults連携
- **ExerciseType+TargetInfo**: ExerciseTypeの目標情報拡張（duration, reps, type, displayText, guidanceText）
- **ImageLoadResult**: 画像ロード結果状態管理（success/fallback）
- **VoiceCharacter**: 音声キャラクター定義（ずんだもん・四国めたん）
- **VoiceSettings**: キャラクター選択管理とUserDefaults連携（レガシー）
- **DisplayState**: UI表示状態管理
- **AudioTextData**: 音声フィードバックテキストデータモデル
- **TrainingRecord**: Core Data エンティティ for トレーニング記録
- **VirtualTrainerApp.xcdatamodeld**: Core Data モデル定義
- **CalendarModels**: Googleカレンダーイベント・隙間時間・通知候補のデータモデル
- **CalendarErrors**: カレンダー連携固有のエラー定義（認証失敗・API通信エラー等）

### Performance Optimization
- **Neural Engine**: A12 Bionic+ チップの専用AI処理ユニット活用
- **Frame Rate Control**: 最大15FPS でのリアルタイム処理
- **Memory Management**: CVPixelBuffer プールによる効率的なメモリ使用
- **Background Processing**: DispatchQueue による非同期AI推論
- **Resource Lifecycle Management**: 自動リソースクリーンアップによるメモリリーク防止
- **Integrated Cleanup System**: カメラセッション終了時の包括的リソース解放

## Development Environment Requirements

### For Python Development
- **OS**: macOS, Linux, Windows
- **Python**: 3.8+
- **RAM**: 8GB+ (推奨: 16GB)
- **GPU**: CUDA対応GPU (オプション、CPU推論も可能)
- **Webcam**: USBカメラまたは内蔵カメラ

### For iOS Development
- **OS**: macOS 14+ (Sonoma)
- **Xcode**: 16.1+
- **iOS Target**: iOS 16+ (推奨: iOS 18)
- **Device**: iPhone with A12 Bionic+ (Neural Engine対応)
- **Memory**: 開発用Mac 16GB+ RAM推奨

## Environment Variables and Configuration

### Python Configuration
```bash
# Optional: CUDA GPU usage
export CUDA_VISIBLE_DEVICES=0

# Model file paths (default: AI_Model/)
export YOLO_MODEL_PATH="./AI_Model/yolo11n-pose.pt"
export GRU_MODEL_PATH="./AI_Model/best_gru_model_v7_quantized.pth"
```

### iOS Configuration
- **Info.plist**:
  - カメラ使用許可設定（NSCameraUsageDescription）
  - 通知使用許可設定（NSUserNotificationsUsageDescription）
  - バックグラウンドモード（fetch, processing for calendar sync）
  - カスタムURLスキーム（OAuth 2.0リダイレクト用）
- **Bundle Identifier**: アプリ識別子
- **Deployment Target**: iOS 16.0 minimum
- **Device Capabilities**: camera, neural-engine, push-notifications
- **App Groups**: （将来的な拡張用）通知拡張とのデータ共有

## Port Configuration

### Python Application
- **Default**: No network ports (standalone application)
- **Webcam**: System camera device access
- **Display**: Local GUI window

### iOS Application
- **Network**:
  - HTTPS only (Google Calendar API通信)
  - OAuth 2.0認証エンドポイント（accounts.google.com）
  - Calendar API エンドポイント（www.googleapis.com/calendar/v3）
- **Camera**: AVCaptureDevice access
- **Storage**:
  - Local CoreData/UserDefaults
  - iOS Keychain（OAuth トークン）
  - UNUserNotificationCenter（通知スケジュール）

## Build and Deployment

### Python Distribution
```bash
# Model Conversion
python AI_Model/convert_models.py       # Unified model conversion script

# Exercise-Specific Applications (全て音声フィードバック付き)
python AI_Model/main_overheadpress.py   # オーバーヘッドプレス
python AI_Model/main_squat.py           # スクワット
python AI_Model/main_sideraises.py      # サイドレイズ
python AI_Model/main_pushup.py          # 腕立て伏せ
python AI_Model/main_side.py            # サイドレイズ検証用
```

### iOS Build Process
```bash
# Open Xcode Project
open VirtualTrainerApp/VirtualTrainerApp.xcodeproj

# Build Configurations
# - Debug: Development with debug symbols
# - Release: Production optimized build
```

## Security and Privacy

### Data Privacy
- **Limited External APIs**:
  - AI処理は完全オンデバイス（外部送信なし）
  - Googleカレンダー連携のみGoogle API使用（ユーザー明示的同意）
  - カレンダーイベントの時刻のみ使用、詳細・タイトルは保存しない
- **No Unnecessary Data Collection**: トレーニングデータと通知効果測定のみローカル保存
- **Camera Privacy**: システムレベルでの権限管理
- **Secure Storage**:
  - iOS Keychain Services（kSecAttrAccessibleWhenUnlockedThisDeviceOnly）
  - OAuth トークンの最高セキュリティレベル暗号化
- **HTTPS Only**: 全てのネットワーク通信はHTTPS暗号化

### Model Security
- **Code Signing**: iOS Distribution署名
- **Model Encryption**: Core ML自動暗号化
- **Runtime Protection**: iOS App Sandbox

## Performance Benchmarks

### Target Performance
- **Inference Time**: < 50ms (Core ML on Neural Engine)
- **Frame Rate**: 10-15 FPS sustainable
- **Memory Usage**: < 200MB RSS
- **Battery Impact**: Minimal (Neural Engine efficiency)

### Monitoring Tools
- **Xcode Instruments**: メモリ、CPU、Neural Engine使用率
- **MetricKit**: プロダクション環境でのパフォーマンス計測
- **Console Logs**: デバッグ情報とエラートラッキング