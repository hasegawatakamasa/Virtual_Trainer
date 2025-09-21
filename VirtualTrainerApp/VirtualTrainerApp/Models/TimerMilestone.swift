import Foundation

/// タイマーの音声マイルストーンを表すenum
/// 特定の残り時間で再生される音声フィードバックのタイミングを定義
enum TimerMilestone: Double, CaseIterable, Codable {
    /// 残り30秒時点での音声フィードバック
    case thirtySeconds = 30.0

    /// 残り10秒時点での音声フィードバック
    case tenSeconds = 10.0

    /// 残り5秒時点での音声フィードバック
    case fiveSeconds = 5.0

    /// タイマー終了時点（0秒）での音声フィードバック
    case zero = 0.0
}

// MARK: - TimerMilestone Extensions

extension TimerMilestone {
    /// 残り時間（秒）を取得
    var remainingTime: TimeInterval {
        return self.rawValue
    }

    /// 日本語表示名を取得
    var displayName: String {
        switch self {
        case .thirtySeconds:
            return "残り30秒"
        case .tenSeconds:
            return "残り10秒"
        case .fiveSeconds:
            return "残り5秒"
        case .zero:
            return "終了"
        }
    }

    /// 音声ファイルのキーを取得（AudioTextData用）
    var audioKey: String {
        switch self {
        case .thirtySeconds:
            return "timer_30_seconds"
        case .tenSeconds:
            return "timer_10_seconds"
        case .fiveSeconds:
            return "timer_5_seconds"
        case .zero:
            return "timer_complete"
        }
    }

    /// マイルストーンの重要度を取得（数値が高いほど重要）
    var priority: Int {
        switch self {
        case .zero:
            return 4
        case .fiveSeconds:
            return 3
        case .tenSeconds:
            return 2
        case .thirtySeconds:
            return 1
        }
    }

    /// 残り時間から該当するマイルストーンを取得
    /// - Parameter remainingTime: 残り時間（秒）
    /// - Returns: 該当するマイルストーン（存在しない場合はnil）
    static func milestoneForRemainingTime(_ remainingTime: TimeInterval) -> TimerMilestone? {
        // 0.5秒の許容範囲でマイルストーンをチェック
        let tolerance: TimeInterval = 0.5

        for milestone in TimerMilestone.allCases.sorted(by: { $0.remainingTime > $1.remainingTime }) {
            if abs(remainingTime - milestone.remainingTime) <= tolerance {
                return milestone
            }
        }
        return nil
    }
}