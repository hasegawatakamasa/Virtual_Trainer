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
- **Primary Conversion**: `convert_models.py` - メイン変換スクリプト
- **System-wide Conversion**: `convert_models_system.py` - 包括的変換パイプラインとエラーハンドリング
- **PyTorch → ONNX**: `convert_models_onnx.py` - ONNX中間フォーマット変換
- **ONNX → Core ML**: `onnx_to_coreml.py` - iOS向け最適化とCore ML変換
- **YOLO専用**: `convert_yolo_only.py` - YOLO11nポーズ専用変換
- **GRU初期化**: `create_fresh_gru.py` - 新規GRUモデル作成とアーキテクチャ設定
- **量子化対応**: `best_gru_model_v7_quantized.pth` - 軽量化済みGRUモデル

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

# System and Utilities
psutil==7.0.0             # System monitoring and resource management
setuptools==80.9.0        # Python package management
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

# Model Conversion (Multiple Options)
python convert_models.py                # Primary model conversion script
python convert_models_system.py         # Comprehensive conversion pipeline
python convert_models_onnx.py           # PyTorch to ONNX conversion
python onnx_to_coreml.py                # ONNX to Core ML conversion
python convert_yolo_only.py             # YOLO model-only conversion
python create_fresh_gru.py              # Create new GRU model

# Application Launch
python AI_Model/main.py
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

// System Integration
import MetricKit                 // Performance monitoring
```

## Audio Feedback System

### VOICEVOX Integration
- **Voice Synthesis**: ずんだもん character voice using VOICEVOX-generated audio files
- **Audio Files**: WAV format, locally stored in app bundle
- **Content**: Japanese voice feedback for form correction and rep counting
- **License Compliance**: Proper VOICEVOX attribution and licensing

### iOS Audio Architecture
```swift
// Audio Playback Management
import AVFoundation              // AVAudioPlayer and AVAudioSession
- AudioFeedbackService           // Centralized audio management
- AVAudioSession configuration   // .playback category with .mixWithOthers
- Cooldown timing control        // Prevent audio spam (3s intervals)
- Exercise zone awareness        // Context-aware audio triggering
```

### Audio File Structure
```
Resources/Audio/
├── zundamon_elbow_error.wav    // Form error feedback
├── 1.wav through 10.wav        // Rep counting (1-10)
└── README.md                   // Setup instructions
```

### Key iOS Components
- **MLModelManager**: Core ML モデル管理とオンデバイス推論
- **CameraManager**: AVFoundation カメラセッション管理
- **FormAnalyzer**: リアルタイムフォーム分析エンジンと種目別設定対応
- **RepCounterManager**: 自動回数カウント状態機械
- **AudioFeedbackService**: VOICEVOX音声フィードバック管理（フォームエラー・回数カウント・速度フィードバック）
- **SpeedAnalyzer**: 動作速度分析とリアルタイム速度フィードバック
- **KeypointOverlayView**: COCO-Pose 17ポイント可視化
- **ExerciseSelectionView**: 種目選択画面とナビゲーション管理
- **ExerciseCardView**: 種目表示カードコンポーネント

### Performance Optimization
- **Neural Engine**: A12 Bionic+ チップの専用AI処理ユニット活用
- **Frame Rate Control**: 最大15FPS でのリアルタイム処理
- **Memory Management**: CVPixelBuffer プールによる効率的なメモリ使用
- **Background Processing**: DispatchQueue による非同期AI推論

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
- **Info.plist**: カメラ使用許可設定
- **Bundle Identifier**: アプリ識別子
- **Deployment Target**: iOS 16.0 minimum
- **Device Capabilities**: camera, neural-engine

## Port Configuration

### Python Application
- **Default**: No network ports (standalone application)
- **Webcam**: System camera device access
- **Display**: Local GUI window

### iOS Application
- **Network**: None (完全オンデバイス処理)
- **Camera**: AVCaptureDevice access
- **Storage**: Local CoreData/UserDefaults

## Build and Deployment

### Python Distribution
```bash
# Development Build
python AI_Model/main.py

# Model Conversion Options
python convert_models.py                # Primary conversion script
python convert_models_system.py         # Complete pipeline with error handling
python convert_models_onnx.py           # PyTorch to ONNX only
python onnx_to_coreml.py                # ONNX to Core ML only
python convert_yolo_only.py             # YOLO conversion only
python create_fresh_gru.py              # New GRU model setup

# Sound-Enhanced Version
python AI_Model/main_sound.py
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
- **No External APIs**: 全ての処理をローカルデバイスで完結
- **No Data Collection**: ユーザーデータの外部送信なし
- **Camera Privacy**: システムレベルでの権限管理
- **Secure Storage**: iOS KeychainServices使用

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