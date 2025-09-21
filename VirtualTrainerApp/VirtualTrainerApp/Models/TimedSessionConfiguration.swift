import Foundation

/// タイマーセッションの設定を管理する構造体
/// セットタイマー機能の動作設定とパラメータを定義
struct TimedSessionConfiguration: Codable, Hashable {
    /// セッション時間（秒）
    /// デフォルト: 60秒
    var duration: TimeInterval = 60

    /// タイマー開始トリガーの種類
    /// デフォルト: レップ待機開始
    var startTrigger: TimerStartTrigger = .waitingForRep

    /// 開始メッセージを表示するかどうか
    /// デフォルト: true
    var showStartMessage: Bool = true

    /// カウントダウン音声を再生するかどうか
    /// デフォルト: true
    var playCountdownAudio: Bool = true

    /// マイルストーン音声を再生するマイルストーンのセット
    /// デフォルト: すべてのマイルストーン
    var enabledMilestones: Set<TimerMilestone> = Set(TimerMilestone.allCases)

    /// セッション開始前の準備時間（秒）
    /// デフォルト: 3秒
    var preparationTime: TimeInterval = 3.0

    /// レップ検知の感度設定
    /// デフォルト: 標準
    var repDetectionSensitivity: RepDetectionSensitivity = .standard

    /// 自動一時停止機能を有効にするかどうか
    /// デフォルト: true（動作が検知されない場合に一時停止）
    var enableAutoPause: Bool = true

    /// 自動一時停止までの非アクティブ時間（秒）
    /// デフォルト: 10秒
    var autoPauseDelay: TimeInterval = 10.0
}

// MARK: - RepDetectionSensitivity

extension TimedSessionConfiguration {
    /// レップ検知の感度設定
    enum RepDetectionSensitivity: String, Codable, CaseIterable {
        /// 低感度 - より明確な動作のみを検知
        case low = "low"

        /// 標準感度 - バランスの取れた検知
        case standard = "standard"

        /// 高感度 - 微細な動作も検知
        case high = "high"

        /// 日本語表示名を取得
        var displayName: String {
            switch self {
            case .low:
                return "低"
            case .standard:
                return "標準"
            case .high:
                return "高"
            }
        }

        /// 感度の説明を取得
        var description: String {
            switch self {
            case .low:
                return "明確な動作のみを検知します"
            case .standard:
                return "バランスの取れた検知を行います"
            case .high:
                return "微細な動作も検知します"
            }
        }
    }
}

// MARK: - TimedSessionConfiguration Extensions

extension TimedSessionConfiguration {
    /// デフォルト設定を取得
    static var `default`: TimedSessionConfiguration {
        return TimedSessionConfiguration()
    }

    /// 60秒セッション用の推奨設定
    static var sixtySecondSession: TimedSessionConfiguration {
        var config = TimedSessionConfiguration()
        config.duration = 60
        config.startTrigger = .waitingForRep
        config.enabledMilestones = [.thirtySeconds, .tenSeconds, .zero]
        return config
    }

    /// 短時間セッション用の推奨設定（30秒）
    static var shortSession: TimedSessionConfiguration {
        var config = TimedSessionConfiguration()
        config.duration = 30
        config.startTrigger = .immediate
        config.enabledMilestones = [.tenSeconds, .fiveSeconds, .zero]
        config.preparationTime = 2.0
        return config
    }

    /// 長時間セッション用の推奨設定（120秒）
    static var longSession: TimedSessionConfiguration {
        var config = TimedSessionConfiguration()
        config.duration = 120
        config.startTrigger = .waitingForRep
        config.enabledMilestones = [.thirtySeconds, .tenSeconds, .zero]
        config.autoPauseDelay = 15.0
        return config
    }

    /// 設定の妥当性を検証
    var isValid: Bool {
        return duration > 0 &&
               preparationTime >= 0 &&
               autoPauseDelay >= 0 &&
               !enabledMilestones.isEmpty
    }

    /// 設定の表示用サマリーを取得
    var displaySummary: String {
        let minutes = Int(duration / 60)
        let seconds = Int(duration.truncatingRemainder(dividingBy: 60))
        let durationString = minutes > 0 ? "\(minutes)分\(seconds)秒" : "\(seconds)秒"

        return "時間: \(durationString) | 開始: \(startTrigger.displayName)"
    }
}