import Foundation

/// エクササイズセッション情報
struct ExerciseSession: Codable, Identifiable {
    let id: UUID
    let startTime: Date
    var endTime: Date?
    var totalReps: Int
    var formAccuracy: Double // 0.0-1.0
    let userId: UUID
    
    /// セッション中のフォーム分類履歴
    var formHistory: [FormClassification.Result]
    
    /// 回数履歴
    var repHistory: [RepHistoryEntry]
    
    init(userId: UUID = UUID(), startTime: Date = Date()) {
        self.id = UUID()
        self.userId = userId
        self.startTime = startTime
        self.endTime = nil
        self.totalReps = 0
        self.formAccuracy = 0.0
        self.formHistory = []
        self.repHistory = []
    }
    
    /// セッションの継続時間
    var duration: TimeInterval {
        let end = endTime ?? Date()
        return end.timeIntervalSince(startTime)
    }
    
    /// セッションが進行中かどうか
    var isActive: Bool {
        return endTime == nil
    }
    
    /// セッション終了
    mutating func end() {
        guard isActive else { return }
        endTime = Date()
        calculateFormAccuracy()
    }
    
    /// 回数を追加
    mutating func addRep(angle: Double, classification: FormClassification?) {
        totalReps += 1
        let entry = RepHistoryEntry(
            repNumber: totalReps,
            completedAt: Date(),
            elbow: angle,
            formClassification: classification
        )
        repHistory.append(entry)
    }
    
    /// フォーム分類結果を追加
    mutating func addFormResult(_ result: FormClassification.Result) {
        formHistory.append(result)
    }
    
    /// フォーム精度を計算
    private mutating func calculateFormAccuracy() {
        let validResults = formHistory.filter { result in
            result.classification == .normal || result.classification == .elbowError
        }
        
        guard !validResults.isEmpty else {
            formAccuracy = 0.0
            return
        }
        
        let normalCount = validResults.filter { $0.classification == .normal }.count
        formAccuracy = Double(normalCount) / Double(validResults.count)
    }
    
    /// セッションのサマリー
    struct Summary {
        let duration: TimeInterval
        let totalReps: Int
        let formAccuracy: Double
        let averageRepDuration: TimeInterval?
        let bestStreak: Int // 連続正常フォーム数
        
        var durationFormatted: String {
            let minutes = Int(duration) / 60
            let seconds = Int(duration) % 60
            return String(format: "%d:%02d", minutes, seconds)
        }
        
        var formAccuracyFormatted: String {
            return String(format: "%.1f%%", formAccuracy * 100)
        }
    }
    
    /// セッションサマリーを生成
    func generateSummary() -> Summary {
        let averageRepDuration: TimeInterval? = {
            guard repHistory.count > 1 else { return nil }
            let totalDuration = repHistory.last!.completedAt.timeIntervalSince(repHistory.first!.completedAt)
            return totalDuration / Double(repHistory.count - 1)
        }()
        
        // 連続正常フォーム数を計算
        let bestStreak: Int = {
            var currentStreak = 0
            var maxStreak = 0
            
            for result in formHistory {
                if result.classification == .normal && result.isReliable {
                    currentStreak += 1
                    maxStreak = max(maxStreak, currentStreak)
                } else if result.classification == .elbowError {
                    currentStreak = 0
                }
            }
            
            return maxStreak
        }()
        
        return Summary(
            duration: duration,
            totalReps: totalReps,
            formAccuracy: formAccuracy,
            averageRepDuration: averageRepDuration,
            bestStreak: bestStreak
        )
    }
}