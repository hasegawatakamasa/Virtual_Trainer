import Foundation
import Combine

/// 回数カウント管理サービス
@MainActor
class RepCounterManager: ObservableObject {
    // MARK: - Published Properties
    @Published var repState = RepState()
    @Published var recentHistory: [RepHistoryEntry] = []
    @Published var currentSession: ExerciseSession?
    
    // MARK: - Private Properties
    private let config: RepCounterConfig
    private var eventSubject = PassthroughSubject<RepCountEvent, Never>()
    private var exerciseType: ExerciseType
    private(set) var speedAnalyzer: SpeedAnalyzer
    private var keypointsCollected: [FilteredKeypoints] = []
    
    // MARK: - Public Properties
    
    /// 回数カウントイベントのパブリッシャー
    var eventPublisher: AnyPublisher<RepCountEvent, Never> {
        eventSubject.eraseToAnyPublisher()
    }
    
    // MARK: - Initialization
    init(exerciseType: ExerciseType = .overheadPress, config: RepCounterConfig = AppSettings.shared.createRepCounterConfig()) {
        self.exerciseType = exerciseType
        self.config = config
        self.speedAnalyzer = SpeedAnalyzer()
        startNewSession()
    }
    
    // MARK: - Public Methods
    
    /// フォーム分析結果を基に状態を更新
    func updateState(analysisResult: FormAnalysisResult, formClassification: FormClassification? = nil) {
        let angle = analysisResult.elbowAngle
        let inZone = analysisResult.isInExerciseZone
        
        // キーポイントデータを収集（速度分析用）
        if inZone && repState.state == .bottom {
            keypointsCollected.append(analysisResult.keypoints)
        }
        
        updateState(angle: angle, inZone: inZone, formClassification: formClassification)
    }
    
    /// 角度とゾーン情報を基に状態を更新
    func updateState(angle: Double, inZone: Bool, formClassification: FormClassification? = nil) {
        let previousState = repState.state
        let previousInZone = repState.isInZone
        
        
        // 現在の状態を更新
        repState.lastAngle = angle
        repState.isInZone = inZone
        repState.lastUpdated = Date()
        
        // ゾーンの入退出イベント
        if !previousInZone && inZone {
            eventSubject.send(.zoneEntered)
            if config.debugMode {
                print("🎯 エクササイズゾーン入場")
            }
        } else if previousInZone && !inZone {
            eventSubject.send(.zoneExited)
            if config.debugMode {
                print("🚪 エクササイズゾーン退出")
            }
        }
        
        // エクササイズゾーン内でのみカウント処理
        guard inZone else { 
            return 
        }
        
        var stateChanged = false
        
        // 状態遷移ロジック
        switch repState.state {
        case .top:
            // 上位置から下位置への遷移
            if angle < config.bottomThreshold {
                if config.debugMode {
                    print("📉 TOP -> BOTTOM (\(String(format: "%.1f", angle))°)")
                }
                repState.state = .bottom
                // 新しいrep開始時にキーポイント履歴をクリア
                keypointsCollected.removeAll()
                stateChanged = true
            }
            
        case .bottom:
            // 下位置から上位置への遷移（回数カウント）
            if angle > config.topThreshold {
                if config.debugMode {
                    print("📈 BOTTOM -> TOP (\(String(format: "%.1f", angle))°) - 回数カウント!")
                }
                repState.state = .top
                incrementCount(angle: angle, formClassification: formClassification)
                stateChanged = true
            }
        }
        
        // 状態変化のイベント通知
        if stateChanged {
            eventSubject.send(.stateChanged(from: previousState, to: repState.state))
            if config.debugMode {
                print("🔄 状態変更: \(previousState.description) -> \(repState.state.description)")
            }
        }
    }
    
    /// 回数を手動でカウント
    func incrementCount(angle: Double = 0.0, formClassification: FormClassification? = nil) {
        // 速度分析を実行
        let keypointsCount = keypointsCollected.count
        let currentSpeed = speedAnalyzer.analyzeSpeed(keypointsCount: keypointsCount, isExerciseActive: repState.isInZone)
        
        repState.count += 1
        
        // 履歴に記録
        let historyEntry = RepHistoryEntry(
            repNumber: repState.count,
            completedAt: Date(),
            elbow: angle,
            formClassification: formClassification
        )
        recentHistory.append(historyEntry)
        
        // 最新の履歴のみ保持（メモリ効率のため）
        if recentHistory.count > 50 {
            recentHistory.removeFirst(recentHistory.count - 50)
        }
        
        // セッションに記録
        currentSession?.addRep(angle: angle, classification: formClassification)
        
        // 統計更新
        AppSettings.shared.totalRepCount += 1
        
        // イベント通知（速度情報付き）
        eventSubject.send(.repCompleted(count: repState.count))
        
        // 速度フィードバック用のイベント送信（必要に応じて）
        if currentSpeed.needsFeedback && speedAnalyzer.shouldPlayFeedback(for: currentSpeed, isExerciseActive: repState.isInZone) {
            eventSubject.send(.speedFeedbackNeeded(speed: currentSpeed))
            // recordFeedbackPlayedは実際に音声が再生された後に呼ばれるべき
        }
        
        if config.debugMode {
            print("🏃 Speed Analysis: \(currentSpeed.displayName) (keypoints: \(keypointsCount))")
        }
        
    }
    
    /// セッションをリセット
    func reset() {
        var previousSession = currentSession
        
        // 現在のセッションを終了
        previousSession?.end()
        
        // 状態をリセット
        repState.reset()
        recentHistory.removeAll()
        
        // 新しいセッション開始
        startNewSession()
        
        // イベント通知
        eventSubject.send(.sessionReset)
        
        if config.debugMode {
            print("🔄 Session reset")
        }
    }
    
    /// 現在のセッションを終了
    func endCurrentSession() {
        currentSession?.end()
        
        // 統計更新
        if let session = currentSession {
            AppSettings.shared.totalSessionCount += 1
            if session.formAccuracy > AppSettings.shared.bestAccuracy {
                AppSettings.shared.bestAccuracy = session.formAccuracy
            }
        }
    }
    
    /// セッションサマリーを取得
    func getSessionSummary() -> ExerciseSession.Summary? {
        return currentSession?.generateSummary()
    }
    
    // MARK: - Private Methods
    
    private func startNewSession() {
        currentSession = ExerciseSession(exerciseType: exerciseType)
    }
}

// MARK: - Statistics and Analytics
extension RepCounterManager {
    
    /// 平均回数/分を計算
    var averageRepsPerMinute: Double {
        guard repState.count > 0 else { return 0.0 }
        
        let sessionDuration = repState.sessionDuration / 60.0 // 分に変換
        guard sessionDuration > 0 else { return 0.0 }
        
        return Double(repState.count) / sessionDuration
    }
    
    /// 直近のフォーム精度を計算
    var recentFormAccuracy: Double {
        let recentEntries = Array(recentHistory.suffix(10))
        let validEntries = recentEntries.compactMap { $0.formClassification }
        
        guard !validEntries.isEmpty else { return 0.0 }
        
        let correctCount = validEntries.filter { $0 == .normal }.count
        return Double(correctCount) / Double(validEntries.count)
    }
    
    /// セッション統計
    struct SessionStats {
        let totalReps: Int
        let duration: TimeInterval
        let averageRepsPerMinute: Double
        let formAccuracy: Double
        let lastRepTime: Date?
        
        var durationFormatted: String {
            let minutes = Int(duration) / 60
            let seconds = Int(duration) % 60
            return String(format: "%d:%02d", minutes, seconds)
        }
    }
    
    /// 現在のセッション統計を取得
    var sessionStats: SessionStats {
        return SessionStats(
            totalReps: repState.count,
            duration: repState.sessionDuration,
            averageRepsPerMinute: averageRepsPerMinute,
            formAccuracy: recentFormAccuracy,
            lastRepTime: recentHistory.last?.completedAt
        )
    }
}

// MARK: - Debug Support
extension RepCounterManager {
    
    /// デバッグ情報を取得
    var debugInfo: String {
        return """
        RepCounterManager Debug Info:
        - Count: \(repState.count)
        - State: \(repState.state.description)
        - Last Angle: \(String(format: "%.1f", repState.lastAngle))°
        - In Zone: \(repState.isInZone)
        - Session Duration: \(repState.sessionDuration)s
        - Recent History: \(recentHistory.count) entries
        - Config: Top=\(config.topThreshold)°, Bottom=\(config.bottomThreshold)°
        """
    }
    
    /// テスト用の模擬データを生成
    func generateMockData() {
        guard config.debugMode else { return }
        
        for i in 1...5 {
            let mockAngle = Double.random(in: 90...120)
            let mockClassification: FormClassification = .random([.normal, .elbowError])
            incrementCount(angle: mockAngle, formClassification: mockClassification)
            
            // 少し間隔を空ける
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * 0.1) { }
        }
    }
}

// MARK: - FormClassification Extension
private extension FormClassification {
    static func random(_ cases: [FormClassification]) -> FormClassification {
        return cases.randomElement() ?? .normal
    }
}