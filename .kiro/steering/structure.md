# Project Structure - 推しトレ

## Root Directory Organization

```
Virtual_Trainer/
├── AI_Model/                    # Python implementation and AI models
├── VirtualTrainerApp/          # iOS native application
├── Resources/Audio/            # Audio files for iOS app (*.wav files excluded from Git)
├── docs/                       # Project documentation
├── .kiro/                      # Kiro spec-driven development files
├── .claude/                    # Claude Code configuration
├── README.md                  # Main project documentation
├── CLAUDE.md                  # Claude Code project instructions
└── README_IMPLEMENTATION.md   # Detailed implementation guide
```

## AI_Model Directory (Python Implementation)

```
AI_Model/
├── main_overheadpress.py             # オーバーヘッドプレス専用エントリーポイント
├── main_squat.py                     # スクワット専用エントリーポイント
├── main_sideraises.py                # サイドレイズ専用エントリーポイント
├── main_pushup.py                    # 腕立て伏せ専用エントリーポイント
├── main_side.py                      # サイドレイズ検証用エントリーポイント
├── convert_models.py                 # 統合モデル変換ユーティリティ（YOLO・GRU対応）
├── requirements.txt                  # Python dependencies
├── best_gru_model_v7_quantized.pth  # Trained GRU model for form classification
├── yolo11n-pose.pt                  # YOLO11 pose estimation model
├── yolo11n-pose.onnx                # ONNX format for cross-platform compatibility
├── sounds/                           # VOICEVOX audio files for Python version
├── venv/                             # Virtual environment (development)
└── __pycache__/                      # Python bytecode cache
```

### Key Python Files
- **main_overheadpress.py**: オーバーヘッドプレス専用実装（音声フィードバック付き）
- **main_squat.py**: スクワット専用実装（音声フィードバック付き）
- **main_sideraises.py**: サイドレイズ専用実装（音声フィードバック付き）
- **main_pushup.py**: 腕立て伏せ専用実装（音声フィードバック付き）
- **main_side.py**: サイドレイズ検証用実装
- **convert_models.py**: 統合モデル変換ユーティリティ（YOLO・GRU両モデル対応）
- **sounds/**: VOICEVOX-generated audio files (ずんだもん voice) for error feedback and rep counting
- **requirements.txt**: Dependencies including torch, ultralytics, opencv-python

## Model Conversion Utilities (AI_Model Directory)

**Note**: Model conversion scripts have been consolidated into `AI_Model/convert_models.py` for simplified workflow. 以前の複数スクリプト（convert_models_onnx.py, convert_models_system.py, convert_yolo_only.py等）は統合されました。

### Conversion Workflow
1. **Model Training**: Train PyTorch models in Python environment
2. **Format Conversion**: Use convert_models.py to create ONNX intermediates
3. **iOS Optimization**: Convert ONNX to Core ML format for iOS deployment
4. **Integration**: Import Core ML models into Xcode project

### Script Responsibilities
- **convert_models.py**: 統合モデル変換スクリプト - YOLOとGRU両モデルに対応（位置: AI_Model/）

## VirtualTrainerApp Directory (iOS Implementation)

```
VirtualTrainerApp/
├── VirtualTrainerApp.xcodeproj/    # Xcode project configuration
├── VirtualTrainerApp/              # Main app source code
├── VirtualTrainerAppTests/         # Unit tests
└── VirtualTrainerAppUITests/       # UI automation tests
```

### iOS App Source Structure

```
VirtualTrainerApp/VirtualTrainerApp/
├── VirtualTrainerAppApp.swift      # SwiftUI App entry point
├── AppDelegate.swift               # UIApplicationDelegate for background tasks and notification handling
├── ContentView.swift               # Root view (legacy)
├── Info.plist                      # App configuration and permissions
├── VirtualTrainerApp.entitlements  # App capabilities and sandbox settings
├── VirtualTrainerApp.xcdatamodeld/ # Core Data model definition
├── Assets.xcassets/                # App icons and visual assets
├── Preview Content/                # SwiftUI preview resources
├── Resources/                      # App resources
│   ├── Audio/                      # VOICEVOX audio files (manual setup required)
│   └── Image/                      # Character portrait images
│       ├── OshinoAi/               # 推乃 藍（デフォルトトレーナー）character images
│       │   └── normal.png         # 推乃 藍 character portrait
│       ├── ずんだもん/             # ずんだもん character images
│       │   └── zundamon_1.png     # ずんだもん portrait
│       └── 四国めたん/             # 四国めたん character images
│           └── shikoku_metan_1.png # 四国めたん portrait (to be added)
├── MLModels/                       # Core ML model packages
├── Models/                         # Data models and structures
├── Views/                          # SwiftUI view components
├── Services/                       # Business logic and AI services
└── Utilities/                      # Helper functions and extensions
```

### Resources Directory Structure
```
Resources/Audio/
├── README.md                     # Setup instructions for audio files
├── .gitkeep                     # Ensures directory tracking in Git
├── ずんだもん/                    # ずんだもん character audio files
│   ├── zundamon_form_error.wav           # Form error feedback
│   ├── zundamon_fast_warning.wav         # Speed warning
│   ├── zundamon_slow_encouragement.wav   # Speed encouragement
│   └── zundamon_count_01.wav - 10.wav    # Rep counting (1-10)
└── 四国めたん/                    # 四国めたん character audio files
    ├── shikoku_form_error.wav            # Form error feedback
    ├── shikoku_fast_warning.wav          # Speed warning
    ├── shikoku_slow_encouragement.wav    # Speed encouragement
    └── shikoku_count_01.wav - 10.wav     # Rep counting (1-10)

Resources/Image/
├── OshinoAi/                     # 推乃 藍（デフォルトトレーナー）character images
│   └── normal.png                        # 推乃 藍 character portrait
├── ずんだもん/                    # ずんだもん character images
│   └── zundamon_1.png                    # ずんだもん character portrait
└── 四国めたん/                    # 四国めたん character images (planned)
    └── shikoku_metan_1.png               # 四国めたん character portrait (to be added)
```

**Note**: Audio files (*.wav) are currently populated in the development environment. For new setups, developers can copy files from `AI_Model/sounds/` or generate new ones using VOICEVOX following instructions in `Resources/Audio/README.md`.

## Code Organization Patterns

### iOS App Architecture (MVVM + Services)

#### Models (`Models/`)
```
Models/
├── PoseKeypoints.swift      # 17-point COCO pose keypoint structure
├── FormClassification.swift # Exercise form classification enums
├── RepState.swift          # Rep counting state management
├── ExerciseSession.swift   # Workout session data model
├── ExerciseType.swift      # Exercise type definitions and metadata
├── ExerciseType+TargetInfo.swift # ExerciseType extension for structured target information
├── SpeedFeedback.swift     # Speed analysis and feedback state models
├── OshiTrainer.swift       # Oshi trainer character model (personality, voice, image)
├── OshiTrainerSettings.swift # Oshi trainer selection and UserDefaults integration
├── VoiceCharacter.swift    # Multi-character voice system with image support (ずんだもん・四国めたん)
├── DisplayState.swift      # UI display state management
├── AudioTextData.swift     # Audio feedback text data models
├── TrainingRecord.swift    # Core Data entity for training records
├── ImageLoadResult.swift   # Image loading result state (success/fallback)
├── TimerState.swift        # Timer state management
├── TimerStartTrigger.swift # Timer start trigger definitions
├── TimerMilestone.swift    # Timer milestone events (30s, 45s, 60s)
├── TimedSessionConfiguration.swift # Timer session settings
├── SessionInterruption.swift # Session interruption types
├── InterruptedSessionData.swift # Interrupted session data storage
├── SessionCompletionData.swift # Session completion data model
├── TimerError.swift        # Timer-related error definitions
├── CalendarModels.swift    # Google Calendar event and gap time models
└── CalendarErrors.swift    # Calendar-specific error definitions
```

#### Views (`Views/`)
```
Views/
├── ExerciseSelectionView.swift       # Exercise selection screen with grid layout
├── ExerciseCardView.swift            # Individual exercise card component
├── ExerciseDetailView.swift          # Exercise details and start button
├── LastWorkoutSection.swift          # Previous workout display section
├── ExerciseTrainingView.swift        # Main training interface
├── CameraPreviewView.swift           # Camera feed display
├── KeypointOverlayView.swift         # Pose skeleton visualization
├── FeedbackOverlayView.swift         # Real-time feedback UI
├── OshiTrainerSettingsView.swift     # Oshi trainer selection and preview screen
├── VoiceCharacterSettingsView.swift  # Voice character selection and preview screen (legacy)
├── CreditDisplayView.swift           # VOICEVOX license and credit display
├── PermissionView.swift              # Camera permission requests
├── RecordsTabView.swift              # Training records and history display
├── ProgressVisualizationView.swift   # Progress charts and statistics
├── WeeklyChartView.swift             # Weekly training activity chart
├── SessionResultView.swift           # Session completion result display
├── SettingsView.swift                # Integrated settings screen (notifications, calendar, trainer)
├── CalendarSettingsView.swift        # Google Calendar integration and OAuth settings
├── NotificationSettingsView.swift    # Notification frequency, time range, and day customization
├── NotificationStatsView.swift       # Notification analytics (tap rate, completion rate)
└── Components/
    ├── LiveAudioTextView.swift       # Live audio feedback text display component
    ├── SwipableTrainerSelectionView.swift # Swipable trainer selection carousel
    ├── SwipableCharacterSelectionView.swift # Swipable character selection carousel (legacy)
    ├── TrainerImageView.swift        # Trainer image loading and display
    ├── CharacterImageView.swift      # Character image loading and display (legacy)
    ├── VoicePreviewButton.swift      # Voice preview playback button
    ├── TrainerSelectionSuccessMessage.swift # Trainer selection success message
    ├── HintView.swift                # Hint and guidance display component
    ├── FutureExpansionBanner.swift   # Future exercise expansion banner component
    ├── QuickPreviewOverlay.swift     # Coming-soon exercise preview overlay
    ├── TimerDisplayView.swift        # 60-second timer countdown display
    ├── StartMessageOverlay.swift     # Session start countdown overlay
    └── FinishOverlayView.swift       # Session finish animation overlay
```

#### Services (`Services/`)
```
Services/
├── MLModelManager.swift               # Core ML model loading and inference
├── CameraManager.swift                # AVFoundation camera management
├── FormAnalyzer.swift                # Exercise form analysis algorithms with exercise-specific settings
├── RepCounterManager.swift           # Automatic rep counting logic
├── SpeedAnalyzer.swift               # Movement speed analysis and feedback control
├── AudioFeedbackService.swift        # VOICEVOX audio feedback (form, rep counting, speed feedback, achievements)
├── VoicePreviewService.swift         # Random voice preview playback with haptic feedback and AVAudioPlayerDelegate
├── CoreDataManager.swift             # Core Data stack management and persistence
├── TrainingSessionService.swift      # Training session recording and history
├── AchievementSystem.swift           # Achievement detection and unlocking logic
├── OshiReactionManager.swift         # Oshi character reactions and affinity system
├── SessionTimerManager.swift         # 60-second timer session management
├── InterruptionHandler.swift         # Session interruption handling and recovery
├── SessionCompletionCoordinator.swift # Session completion flow coordination
├── ResourceCleanupCoordinator.swift  # System resource management and cleanup coordination
├── IntegratedCleanupService.swift    # Unified resource cleanup service
├── KeychainManager.swift             # Secure OAuth token storage with iOS Keychain
├── GoogleCalendarAuthService.swift   # OAuth 2.0 authentication and token management
├── GoogleCalendarAPIClient.swift     # Google Calendar API client for event fetching
├── CalendarEventAnalyzer.swift       # Calendar event analysis and gap time detection
├── CalendarSyncCoordinator.swift     # Calendar synchronization and background update coordination
├── CalendarPrivacyManager.swift      # Privacy-compliant calendar data processing
├── OshiTrainerNotificationService.swift # Oshi trainer notification creation and delivery
├── NotificationScheduler.swift       # Notification scheduling and time filtering
├── NotificationAnalyticsService.swift # Notification effectiveness tracking (tap rate, completion rate)
└── NotificationSettingsManager.swift # Notification preferences management (frequency, time range, weekdays)
```

#### Utilities (`Utilities/`)
```
Utilities/
├── AppError.swift               # Centralized error handling
├── UserDefaultsKeys.swift       # Configuration and settings keys
├── KeychainKeys.swift           # Keychain item keys for OAuth tokens
├── ResourceCleanupError.swift   # Resource management error definitions
├── OshiTrainerError.swift       # Oshi trainer system error definitions (trainerNotFound, imageLoadFailed, etc.)
├── CharacterImageError.swift    # Character image loading error definitions
└── ColorExtensions.swift        # UI color theme extensions and utilities
```

#### Core ML Models (`MLModels/`)
```
MLModels/
├── YOLO11nPose.mlpackage/       # Pose detection Core ML model
└── GRUFormClassifier.mlpackage/ # Form classification Core ML model
```

## File Naming Conventions

### Swift Files
- **PascalCase** for class names: `MLModelManager.swift`
- **Descriptive suffixes** indicating purpose:
  - `View.swift` for SwiftUI views
  - `Manager.swift` for service classes
  - `Model.swift` for data structures

### Python Files
- **snake_case** for file names: `convert_models_system.py`
- **Descriptive prefixes** for related functionality:
  - `convert_*` for model conversion scripts
  - `main_*` for application entry points

### Model Files
- **Descriptive names** with version info: `best_gru_model_v7_quantized.pth`
- **Standard extensions**:
  - `.pt` for PyTorch models
  - `.onnx` for ONNX format
  - `.mlpackage` for Core ML models

## Import Organization

### Swift Import Hierarchy
```swift
// 1. Foundation and System Frameworks
import Foundation
import SwiftUI
import Combine

// 2. Apple Frameworks (alphabetical)
import AVFoundation
import CoreML
import Vision

// 3. Third-party Dependencies (none currently)

// 4. Local Modules (relative imports)
// Swift uses automatic module discovery
```

### Python Import Structure
```python
# 1. Standard Library
import os
import cv2
import numpy as np

# 2. Third-party Packages
import torch
import ultralytics
from ultralytics import YOLO

# 3. Local Modules
# from . import local_module
```

## Key Architectural Principles

### 1. Separation of Concerns
- **Models**: Pure data structures with no business logic
- **Views**: UI components with minimal business logic
- **Services**: Encapsulated business logic and external integrations
- **Utilities**: Shared helper functions and extensions

### 2. Dependency Injection
- Services passed as `@StateObject` or `@ObservedObject` to views
- Protocol-based abstractions for testability
- Centralized configuration through `AppSettings`

### 3. Reactive Programming
- **Combine** for event-driven architecture
- `@Published` properties for automatic UI updates
- Event-driven communication between services
- **NavigationStack** for iOS 16+ navigation management

### 4. Error Handling
- **Result types** for recoverable errors
- **AppError enum** for centralized error definitions
- Graceful degradation with fallback behaviors

### 5. Performance Optimization
- **Async/await** for non-blocking AI inference
- **DispatchQueue** management for background processing
- **Memory pooling** for CVPixelBuffer management
- **Frame rate limiting** to prevent resource exhaustion

### 6. Privacy and Security
- **On-device processing** with no external API calls
- **Secure storage** using iOS Keychain when needed
- **Permission-based access** to camera resources
- **Data minimization** with no persistent user data collection

### 7. Resource Management and Cleanup
- **Centralized resource coordination** via ResourceCleanupCoordinator
- **Integrated cleanup services** for camera sessions and audio resources
- **Automatic resource lifecycle management** to prevent memory leaks
- **Error handling** for resource cleanup failures with proper error propagation

## Development Workflow Patterns

### Model Development Lifecycle
1. **Python Prototyping**: Develop and test in `AI_Model/`
2. **Model Training**: Train and optimize PyTorch models
3. **Model Conversion**: Use conversion scripts to create Core ML versions
4. **iOS Integration**: Import Core ML models into Xcode project
5. **Testing and Validation**: Verify accuracy and performance on device

### Swift Development Flow
1. **Service Implementation**: Business logic in service classes
2. **Model Definition**: Data structures and protocols
3. **View Development**: SwiftUI components with reactive bindings
4. **Integration Testing**: End-to-end feature validation
5. **Performance Profiling**: Instruments-based optimization

### Cross-Platform Consistency
- **Shared algorithms**: Form analysis logic consistent between Python and Swift
- **Model compatibility**: Same training data and evaluation metrics
- **User experience parity**: Consistent behavior across platforms