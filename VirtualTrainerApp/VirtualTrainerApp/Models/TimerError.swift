//
//  TimerError.swift
//  VirtualTrainerApp
//
//  Created on 2025/09/15.
//

import Foundation

/// タイマー機能に関連するエラー
public enum TimerError: LocalizedError {
    /// タイマーが既に実行中
    case timerAlreadyRunning

    /// セッションが中断された
    case sessionInterrupted(SessionInterruptionType)

    /// 復旧に失敗
    case recoveryFailed(String)

    /// タイマーが開始されていない
    case timerNotStarted

    /// 無効な設定
    case invalidConfiguration(String)

    /// リソースが利用できない
    case resourceUnavailable(String)

    /// データ保存エラー
    case dataSaveError(String)

    /// データ読み込みエラー
    case dataLoadError(String)

    /// ネットワークエラー（将来の拡張用）
    case networkError(String)

    /// 不明なエラー
    case unknown(String)

    /// エラーの説明
    public var errorDescription: String? {
        switch self {
        case .timerAlreadyRunning:
            return "タイマーは既に実行中です"

        case .sessionInterrupted(let type):
            return "セッションが中断されました: \(type.displayName)"

        case .recoveryFailed(let reason):
            return "復旧に失敗しました: \(reason)"

        case .timerNotStarted:
            return "タイマーが開始されていません"

        case .invalidConfiguration(let detail):
            return "無効な設定です: \(detail)"

        case .resourceUnavailable(let resource):
            return "リソースが利用できません: \(resource)"

        case .dataSaveError(let detail):
            return "データの保存に失敗しました: \(detail)"

        case .dataLoadError(let detail):
            return "データの読み込みに失敗しました: \(detail)"

        case .networkError(let detail):
            return "ネットワークエラー: \(detail)"

        case .unknown(let detail):
            return "不明なエラー: \(detail)"
        }
    }

    /// 失敗の理由
    public var failureReason: String? {
        switch self {
        case .timerAlreadyRunning:
            return "既存のタイマーセッションが実行中のため、新しいセッションを開始できません"

        case .sessionInterrupted(let type):
            return type.description

        case .recoveryFailed(let reason):
            return reason

        case .timerNotStarted:
            return "タイマーの開始処理が完了していません"

        case .invalidConfiguration(let detail):
            return detail

        case .resourceUnavailable(let resource):
            return "\(resource)が利用できない状態です"

        case .dataSaveError(let detail):
            return detail

        case .dataLoadError(let detail):
            return detail

        case .networkError(let detail):
            return detail

        case .unknown(let detail):
            return detail
        }
    }

    /// 復旧の提案
    public var recoverySuggestion: String? {
        switch self {
        case .timerAlreadyRunning:
            return "現在のセッションを終了してから、新しいセッションを開始してください"

        case .sessionInterrupted(let type):
            if type.isRecoverable {
                return "アプリを再起動してセッションを再開してください"
            } else {
                return "新しいセッションを開始してください"
            }

        case .recoveryFailed:
            return "アプリを再起動して、もう一度お試しください"

        case .timerNotStarted:
            return "タイマーを開始してから操作を行ってください"

        case .invalidConfiguration:
            return "設定を確認して、もう一度お試しください"

        case .resourceUnavailable:
            return "しばらく待ってから、もう一度お試しください"

        case .dataSaveError:
            return "ストレージの空き容量を確認してください"

        case .dataLoadError:
            return "アプリを再起動してください"

        case .networkError:
            return "ネットワーク接続を確認してください"

        case .unknown:
            return "アプリを再起動して、もう一度お試しください"
        }
    }

    /// ユーザーに表示するアラートタイトル
    public var alertTitle: String {
        switch self {
        case .timerAlreadyRunning:
            return "タイマー実行中"

        case .sessionInterrupted:
            return "セッション中断"

        case .recoveryFailed:
            return "復旧失敗"

        case .timerNotStarted:
            return "タイマー未開始"

        case .invalidConfiguration:
            return "設定エラー"

        case .resourceUnavailable:
            return "リソースエラー"

        case .dataSaveError:
            return "保存エラー"

        case .dataLoadError:
            return "読み込みエラー"

        case .networkError:
            return "ネットワークエラー"

        case .unknown:
            return "エラー"
        }
    }

    /// エラーが致命的かどうか
    public var isFatal: Bool {
        switch self {
        case .timerAlreadyRunning, .timerNotStarted, .invalidConfiguration:
            return false
        case .sessionInterrupted(let type):
            return !type.isRecoverable
        case .recoveryFailed, .resourceUnavailable, .dataSaveError, .dataLoadError:
            return true
        case .networkError:
            return false
        case .unknown:
            return true
        }
    }

    /// エラーコード
    public var code: Int {
        switch self {
        case .timerAlreadyRunning: return 1001
        case .sessionInterrupted: return 1002
        case .recoveryFailed: return 1003
        case .timerNotStarted: return 1004
        case .invalidConfiguration: return 1005
        case .resourceUnavailable: return 1006
        case .dataSaveError: return 1007
        case .dataLoadError: return 1008
        case .networkError: return 1009
        case .unknown: return 9999
        }
    }
}