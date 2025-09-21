import Foundation

/// タイマーの開始トリガーを表すenum
/// セットタイマーがどのようにして開始されるかを定義
enum TimerStartTrigger: String, CaseIterable, Codable {
    /// レップ待機開始 - ユーザーの最初の動作検知を待ってから開始
    case waitingForRep = "waiting_for_rep"

    /// 手動開始 - ユーザーがボタンを押して手動で開始
    case manual = "manual"

    /// 即座開始 - 設定完了後すぐに開始（カウントダウンなし）
    case immediate = "immediate"
}

// MARK: - TimerStartTrigger Extensions

extension TimerStartTrigger {
    /// 日本語表示名を取得
    var displayName: String {
        switch self {
        case .waitingForRep:
            return "動作検知で開始"
        case .manual:
            return "手動開始"
        case .immediate:
            return "即座開始"
        }
    }

    /// 開始方法の説明テキストを取得
    var description: String {
        switch self {
        case .waitingForRep:
            return "最初の動作を検知したらタイマーを開始します"
        case .manual:
            return "開始ボタンを押してタイマーを開始します"
        case .immediate:
            return "設定完了後すぐにタイマーを開始します"
        }
    }

    /// アイコン名を取得（SF Symbols用）
    var iconName: String {
        switch self {
        case .waitingForRep:
            return "figure.run"
        case .manual:
            return "play.circle"
        case .immediate:
            return "bolt.circle"
        }
    }
}