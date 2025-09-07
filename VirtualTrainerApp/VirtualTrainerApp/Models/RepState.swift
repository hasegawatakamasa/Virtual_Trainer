import Foundation

/// エクササイズの状態
enum ExerciseState: String, Codable, CaseIterable {
    case top = "top"
    case bottom = "bottom"
    
    var description: String {
        switch self {
        case .top:
            return "上位置"
        case .bottom:
            return "下位置"
        }
    }
}

/// 回数カウントの状態管理
struct RepState: Codable, Equatable {
    /// 現在の回数
    var count: Int
    
    /// 現在の運動状態
    var state: ExerciseState
    
    /// 最後に測定された角度
    var lastAngle: Double
    
    /// エクササイズゾーン内にいるか
    var isInZone: Bool
    
    /// 最後に状態が更新された時刻
    var lastUpdated: Date
    
    /// セッション開始時刻
    let sessionStartTime: Date
    
    init(
        count: Int = 0,
        state: ExerciseState = .top,
        lastAngle: Double = 0.0,
        isInZone: Bool = false,
        sessionStartTime: Date = Date()
    ) {
        self.count = count
        self.state = state
        self.lastAngle = lastAngle
        self.isInZone = isInZone
        self.lastUpdated = Date()
        self.sessionStartTime = sessionStartTime
    }
    
    /// 状態をリセット
    mutating func reset() {
        self.count = 0
        self.state = .top
        self.lastAngle = 0.0
        self.isInZone = false
        self.lastUpdated = Date()
    }
    
    /// セッションの継続時間
    var sessionDuration: TimeInterval {
        return Date().timeIntervalSince(sessionStartTime)
    }
}

/// 回数カウンターの設定
struct RepCounterConfig: Codable {
    /// 上位置の閾値（度）
    let topThreshold: Double
    
    /// 下位置の閾値（度）
    let bottomThreshold: Double
    
    /// 最小フレーム数（フォーム分析用）
    let minFramesForAnalysis: Int
    
    /// デバッグモード
    let debugMode: Bool
    
    static let `default` = RepCounterConfig(
        topThreshold: 130.0,
        bottomThreshold: 100.0,
        minFramesForAnalysis: 10,
        debugMode: false
    )
}

/// 回数カウントのイベント
enum RepCountEvent {
    case repCompleted(count: Int)
    case stateChanged(from: ExerciseState, to: ExerciseState)
    case zoneEntered
    case zoneExited
    case sessionReset
}

/// 回数カウント履歴のエントリ
struct RepHistoryEntry: Codable, Identifiable {
    let id: UUID
    let repNumber: Int
    let completedAt: Date
    let elbow: Double
    let formClassification: FormClassification?
    
    init(repNumber: Int, completedAt: Date, elbow: Double, formClassification: FormClassification?) {
        self.id = UUID()
        self.repNumber = repNumber
        self.completedAt = completedAt
        self.elbow = elbow
        self.formClassification = formClassification
    }
    
    var duration: TimeInterval? {
        // 前の記録との時間差を計算するため、外部で設定
        return nil
    }
}