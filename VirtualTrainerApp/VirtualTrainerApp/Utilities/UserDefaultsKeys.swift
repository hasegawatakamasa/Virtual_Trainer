import Foundation
import AVFoundation

/// エクササイズ履歴アイテム
struct ExerciseHistoryItem: Codable {
    let exerciseType: ExerciseType
    let date: Date
    let repCount: Int
    let accuracy: Double
    
    init(exerciseType: ExerciseType, date: Date = Date(), repCount: Int = 0, accuracy: Double = 0.0) {
        self.exerciseType = exerciseType
        self.date = date
        self.repCount = repCount
        self.accuracy = accuracy
    }
}

/// UserDefaultsのキー定数
enum UserDefaultsKeys {
    // MARK: - General Settings
    static let debugMode = "debugMode"
    static let firstLaunch = "firstLaunch"
    static let appVersion = "appVersion"
    
    // MARK: - Camera Settings
    static let cameraPosition = "cameraPosition"
    static let preferredCameraDevice = "preferredCameraDevice"
    
    // MARK: - Exercise Configuration
    static let topThreshold = "topThreshold"
    static let bottomThreshold = "bottomThreshold"
    static let minFramesForAnalysis = "minFramesForAnalysis"
    
    // MARK: - Performance Settings
    static let targetFPS = "targetFPS"
    static let useNeuralEngine = "useNeuralEngine"
    static let adaptiveFrameRate = "adaptiveFrameRate"
    
    // MARK: - Session Data
    static let lastSessionId = "lastSessionId"
    static let totalRepCount = "totalRepCount"
    static let totalSessionCount = "totalSessionCount"
    static let bestAccuracy = "bestAccuracy"
    
    // MARK: - Privacy Settings
    static let analyticsEnabled = "analyticsEnabled"
    static let crashReportingEnabled = "crashReportingEnabled"
    
    // MARK: - UI Preferences
    static let showKeypoints = "showKeypoints"
    static let showDebugInfo = "showDebugInfo"
    static let showPerformanceStats = "showPerformanceStats"
    
    // MARK: - Audio Feedback Settings
    static let audioFeedbackEnabled = "audioFeedbackEnabled"
    
    // MARK: - Exercise Selection Settings
    static let lastSelectedExercise = "lastSelectedExercise"
    static let exerciseHistory = "exerciseHistory"
    static let defaultExercise = "defaultExercise"

    // MARK: - Google Calendar Settings
    static let notificationSettings = "notificationSettings"
    static let calendarLastSyncTime = "calendarLastSyncTime"
    static let calendarConsentGiven = "calendarConsentGiven"
    static let isCalendarConnected = "isCalendarConnected"
    static let connectedGoogleEmail = "connectedGoogleEmail"
}

/// UserDefaultsのデフォルト値
extension UserDefaultsKeys {
    /// デフォルト値を設定
    static func registerDefaults() {
        let defaults: [String: Any] = [
            debugMode: false,
            firstLaunch: true,
            cameraPosition: AVCaptureDevice.Position.front.rawValue,
            topThreshold: 130.0,
            bottomThreshold: 100.0,
            minFramesForAnalysis: 10,
            targetFPS: 15.0,
            useNeuralEngine: true,
            adaptiveFrameRate: true,
            totalRepCount: 0,
            totalSessionCount: 0,
            bestAccuracy: 0.0,
            analyticsEnabled: false,
            crashReportingEnabled: true,
            showKeypoints: true,
            showDebugInfo: false,
            showPerformanceStats: false,
            audioFeedbackEnabled: true,
            lastSelectedExercise: ExerciseType.overheadPress.rawValue,
            defaultExercise: ExerciseType.overheadPress.rawValue
        ]
        
        UserDefaults.standard.register(defaults: defaults)
    }
}

/// UserDefaultsのタイプセーフなアクセサー
struct AppSettings {
    private let userDefaults = UserDefaults.standard
    
    // MARK: - General
    var debugMode: Bool {
        get { userDefaults.bool(forKey: UserDefaultsKeys.debugMode) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.debugMode) }
    }
    
    var isFirstLaunch: Bool {
        get { userDefaults.bool(forKey: UserDefaultsKeys.firstLaunch) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.firstLaunch) }
    }
    
    // MARK: - Camera
    var cameraPosition: AVCaptureDevice.Position {
        get {
            let rawValue = userDefaults.integer(forKey: UserDefaultsKeys.cameraPosition)
            return AVCaptureDevice.Position(rawValue: rawValue) ?? .front
        }
        set {
            userDefaults.set(newValue.rawValue, forKey: UserDefaultsKeys.cameraPosition)
        }
    }
    
    // MARK: - Exercise Configuration
    var topThreshold: Double {
        get { userDefaults.double(forKey: UserDefaultsKeys.topThreshold) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.topThreshold) }
    }
    
    var bottomThreshold: Double {
        get { userDefaults.double(forKey: UserDefaultsKeys.bottomThreshold) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.bottomThreshold) }
    }
    
    var minFramesForAnalysis: Int {
        get { userDefaults.integer(forKey: UserDefaultsKeys.minFramesForAnalysis) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.minFramesForAnalysis) }
    }
    
    // MARK: - Performance
    var targetFPS: Double {
        get { userDefaults.double(forKey: UserDefaultsKeys.targetFPS) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.targetFPS) }
    }
    
    var useNeuralEngine: Bool {
        get { userDefaults.bool(forKey: UserDefaultsKeys.useNeuralEngine) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.useNeuralEngine) }
    }
    
    // MARK: - Statistics
    var totalRepCount: Int {
        get { userDefaults.integer(forKey: UserDefaultsKeys.totalRepCount) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.totalRepCount) }
    }
    
    var totalSessionCount: Int {
        get { userDefaults.integer(forKey: UserDefaultsKeys.totalSessionCount) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.totalSessionCount) }
    }
    
    var bestAccuracy: Double {
        get { userDefaults.double(forKey: UserDefaultsKeys.bestAccuracy) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.bestAccuracy) }
    }
    
    // MARK: - UI
    var showKeypoints: Bool {
        get { userDefaults.bool(forKey: UserDefaultsKeys.showKeypoints) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.showKeypoints) }
    }
    
    var showDebugInfo: Bool {
        get { userDefaults.bool(forKey: UserDefaultsKeys.showDebugInfo) }
        set { userDefaults.set(newValue, forKey: UserDefaultsKeys.showDebugInfo) }
    }
    
    // MARK: - Exercise Selection
    var lastSelectedExercise: ExerciseType {
        get {
            let rawValue = userDefaults.string(forKey: UserDefaultsKeys.lastSelectedExercise) ?? ExerciseType.overheadPress.rawValue
            return ExerciseType(rawValue: rawValue) ?? .overheadPress
        }
        set {
            userDefaults.set(newValue.rawValue, forKey: UserDefaultsKeys.lastSelectedExercise)
        }
    }
    
    var defaultExercise: ExerciseType {
        get {
            let rawValue = userDefaults.string(forKey: UserDefaultsKeys.defaultExercise) ?? ExerciseType.overheadPress.rawValue
            return ExerciseType(rawValue: rawValue) ?? .overheadPress
        }
        set {
            userDefaults.set(newValue.rawValue, forKey: UserDefaultsKeys.defaultExercise)
        }
    }
    
    /// 種目選択履歴を保存
    func saveExerciseHistory(_ item: ExerciseHistoryItem) {
        var history = exerciseHistory
        history.append(item)
        
        // 最大100件まで保持
        if history.count > 100 {
            history.removeFirst(history.count - 100)
        }
        
        if let data = try? JSONEncoder().encode(history) {
            userDefaults.set(data, forKey: UserDefaultsKeys.exerciseHistory)
        }
    }
    
    /// 種目選択履歴を取得
    var exerciseHistory: [ExerciseHistoryItem] {
        guard let data = userDefaults.data(forKey: UserDefaultsKeys.exerciseHistory),
              let history = try? JSONDecoder().decode([ExerciseHistoryItem].self, from: data) else {
            return []
        }
        return history
    }
    
    // MARK: - RepCounterConfig生成
    func createRepCounterConfig() -> RepCounterConfig {
        let top = topThreshold
        let bottom = bottomThreshold
        let frames = minFramesForAnalysis
        let debug = debugMode
        
        // 0の場合はデフォルト値を使用（フォールバック）
        let finalTop = top > 0 ? top : RepCounterConfig.default.topThreshold
        let finalBottom = bottom > 0 ? bottom : RepCounterConfig.default.bottomThreshold
        let finalFrames = frames > 0 ? frames : RepCounterConfig.default.minFramesForAnalysis
        
        return RepCounterConfig(
            topThreshold: finalTop,
            bottomThreshold: finalBottom,
            minFramesForAnalysis: finalFrames,
            debugMode: debug
        )
    }
}

/// シングルトンアクセス
extension AppSettings {
    static var shared = AppSettings()
}