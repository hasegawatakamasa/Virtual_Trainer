import Foundation

/// セッション完了時のデータを格納する構造体
/// タイマー終了後の結果データや統計情報を管理
struct SessionCompletionData: Codable, Hashable {
    /// セッション開始時刻
    let startTime: Date

    /// セッション終了時刻
    let endTime: Date

    /// 設定されたセッション時間（秒）
    let configuredDuration: TimeInterval

    /// 実際のセッション実行時間（秒）
    let actualDuration: TimeInterval

    /// 完了したレップ数
    let completedReps: Int

    /// セッション完了の理由
    let completionReason: CompletionReason

    /// フォームエラーの回数
    let formErrorCount: Int

    /// 速度警告の回数
    let speedWarningCount: Int

    /// セッション中の平均速度（レップ/分）
    let averageRepsPerMinute: Double?

    /// 最高連続正確レップ数
    let maxConsecutiveCorrectReps: Int

    /// 使用された音声キャラクター
    let voiceCharacter: String

    /// 使用されたエクササイズタイプ
    let exerciseType: String

    /// フォームエラー数（互換性のための計算プロパティ）
    var formErrors: Int {
        return formErrorCount
    }
}

// MARK: - CompletionReason

extension SessionCompletionData {
    /// セッション完了の理由を表すenum
    enum CompletionReason: String, Codable, CaseIterable {
        /// 正常完了 - 設定時間まで完了
        case completed = "completed"

        /// タイマー完了による終了
        case timerCompleted = "timer_completed"

        /// ユーザーによるキャンセル
        case cancelled = "cancelled"

        /// アプリの中断による終了
        case interrupted = "interrupted"

        /// エラーによる終了
        case error = "error"

        /// 日本語表示名を取得
        var displayName: String {
            switch self {
            case .completed:
                return "完了"
            case .timerCompleted:
                return "タイマー完了"
            case .cancelled:
                return "キャンセル"
            case .interrupted:
                return "中断"
            case .error:
                return "エラー"
            }
        }

        /// 成功したセッションかどうか
        var isSuccessful: Bool {
            return self == .completed || self == .timerCompleted
        }
    }
}

// MARK: - SessionCompletionData Extensions

extension SessionCompletionData {
    /// セッションの完了率を計算（0.0 - 1.0）
    var completionRate: Double {
        guard configuredDuration > 0 else { return 0.0 }
        return min(actualDuration / configuredDuration, 1.0)
    }

    /// セッションが成功したかどうか
    var isSuccessful: Bool {
        return completionReason.isSuccessful
    }

    /// セッションの効率性スコア（0.0 - 1.0）
    /// フォームエラーと速度警告の少なさを基準に計算
    var efficiencyScore: Double {
        guard completedReps > 0 else { return 0.0 }

        let totalWarnings = formErrorCount + speedWarningCount
        let errorRate = Double(totalWarnings) / Double(completedReps)

        // エラー率が低いほど高いスコア
        return max(0.0, 1.0 - (errorRate * 0.5))
    }

    /// セッションの統計情報を文字列で取得
    var statisticsSummary: String {
        let durationMinutes = Int(actualDuration / 60)
        let durationSeconds = Int(actualDuration.truncatingRemainder(dividingBy: 60))

        var summary = "時間: \(durationMinutes):\(String(format: "%02d", durationSeconds))\n"
        summary += "レップ数: \(completedReps)\n"
        summary += "完了率: \(Int(completionRate * 100))%"

        if let avgRpm = averageRepsPerMinute {
            summary += "\n平均速度: \(String(format: "%.1f", avgRpm))レップ/分"
        }

        return summary
    }
}