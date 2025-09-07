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
├── convert_models_*.py         # Model conversion utilities
├── README.md                  # Main project documentation
├── CLAUDE.md                  # Claude Code project instructions
└── README_IMPLEMENTATION.md   # Detailed implementation guide
```

## AI_Model Directory (Python Implementation)

```
AI_Model/
├── main.py                           # Core Python application entry point
├── main_sound.py                     # Sound-enhanced version
├── requirements.txt                  # Python dependencies
├── best_gru_model_v7_quantized.pth  # Trained GRU model for form classification
├── yolo11n-pose.pt                  # YOLO11 pose estimation model
├── yolo11n-pose.onnx                # ONNX format for cross-platform compatibility
├── venv/                             # Virtual environment (development)
└── __pycache__/                      # Python bytecode cache
```

### Key Python Files
- **main.py**: Desktop application with real-time pose detection and form analysis
- **main_sound.py**: Enhanced version with audio feedback capabilities
- **requirements.txt**: Dependencies including torch, ultralytics, opencv-python

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
├── ContentView.swift               # Root view (legacy)
├── Info.plist                      # App configuration and permissions
├── VirtualTrainerApp.entitlements  # App capabilities and sandbox settings
├── Assets.xcassets/                # App icons and visual assets
├── Preview Content/                # SwiftUI preview resources
├── Resources/Audio/                # VOICEVOX audio files (manual setup required)
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
├── zundamon_elbow_error.wav     # Form error feedback (manual setup)
└── 1.wav - 10.wav               # Rep counting audio files (manual setup)
```

**Note**: Audio files (*.wav) are excluded from Git version control for repository size optimization. Developers must manually add these files following the instructions in `Resources/Audio/README.md`.

## Code Organization Patterns

### iOS App Architecture (MVVM + Services)

#### Models (`Models/`)
```
Models/
├── PoseKeypoints.swift      # 17-point COCO pose keypoint structure
├── FormClassification.swift # Exercise form classification enums
├── RepState.swift          # Rep counting state management
└── ExerciseSession.swift   # Workout session data model
```

#### Views (`Views/`)
```
Views/
├── ExerciseTrainingView.swift    # Main training interface
├── CameraPreviewView.swift       # Camera feed display
├── KeypointOverlayView.swift     # Pose skeleton visualization
├── FeedbackOverlayView.swift     # Real-time feedback UI
└── PermissionView.swift          # Camera permission requests
```

#### Services (`Services/`)
```
Services/
├── MLModelManager.swift          # Core ML model loading and inference
├── CameraManager.swift           # AVFoundation camera management
├── FormAnalyzer.swift           # Exercise form analysis algorithms
├── RepCounterManager.swift      # Automatic rep counting logic
└── AudioFeedbackService.swift   # VOICEVOX audio feedback and AVAudioPlayer management
```

#### Utilities (`Utilities/`)
```
Utilities/
├── AppError.swift               # Centralized error handling
└── UserDefaultsKeys.swift       # Configuration and settings keys
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