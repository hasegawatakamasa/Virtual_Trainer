import Foundation

/// タイマーの現在の状態を表すenum
/// セットタイマー機能における状態遷移を管理
enum TimerState: String, CaseIterable, Codable {
    /// 未開始状態 - タイマーがまだ開始されていない
    case notStarted = "not_started"

    /// レップ待機中 - ユーザーの動作検知を待っている状態
    case waitingForRep = "waiting_for_rep"

    /// 開始メッセージ表示中 - 「始めます」等の開始通知を表示している状態
    case showingStart = "showing_start"

    /// 実行中 - タイマーがカウントダウン中
    case running = "running"

    /// 一時停止中 - ユーザーによって一時停止された状態
    case paused = "paused"

    /// 完了 - タイマーが正常に終了した状態
    case completed = "completed"

    /// キャンセル - ユーザーによってキャンセルされた状態
    case cancelled = "cancelled"
}

// MARK: - TimerState Extensions

extension TimerState {
    /// タイマーがアクティブな状態かどうかを判定
    var isActive: Bool {
        switch self {
        case .running, .showingStart, .waitingForRep:
            return true
        default:
            return false
        }
    }

    /// タイマーが終了状態かどうかを判定
    var isFinished: Bool {
        switch self {
        case .completed, .cancelled:
            return true
        default:
            return false
        }
    }

    /// 日本語表示名を取得
    var displayName: String {
        switch self {
        case .notStarted:
            return "未開始"
        case .waitingForRep:
            return "動作待機中"
        case .showingStart:
            return "開始準備中"
        case .running:
            return "実行中"
        case .paused:
            return "一時停止"
        case .completed:
            return "完了"
        case .cancelled:
            return "キャンセル"
        }
    }
}