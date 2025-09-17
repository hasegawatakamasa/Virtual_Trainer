//
//  SessionInterruption.swift
//  VirtualTrainerApp
//
//  Created on 2025/09/15.
//

import Foundation

/// セッション中断のタイプを定義
public enum SessionInterruptionType: String, CaseIterable, Codable {
    /// カメラ接続が失われた
    case cameraLost = "camera_lost"

    /// メモリ不足による中断
    case memoryPressure = "memory_pressure"

    /// バックグラウンドへの遷移
    case backgroundTransition = "background_transition"

    /// ユーザーによる手動中断
    case userInterruption = "user_interruption"

    /// システムによる中断（電話着信など）
    case systemInterruption = "system_interruption"

    /// アプリケーションの終了
    case appTermination = "app_termination"

    /// 不明な理由による中断
    case unknown = "unknown"

    /// 中断タイプの表示名
    var displayName: String {
        switch self {
        case .cameraLost:
            return "カメラ接続エラー"
        case .memoryPressure:
            return "メモリ不足"
        case .backgroundTransition:
            return "バックグラウンド移行"
        case .userInterruption:
            return "手動中断"
        case .systemInterruption:
            return "システム中断"
        case .appTermination:
            return "アプリ終了"
        case .unknown:
            return "不明な中断"
        }
    }

    /// 中断タイプの詳細説明
    var description: String {
        switch self {
        case .cameraLost:
            return "カメラとの接続が失われました"
        case .memoryPressure:
            return "メモリ不足のため中断されました"
        case .backgroundTransition:
            return "アプリがバックグラウンドに移行しました"
        case .userInterruption:
            return "ユーザーによってセッションが中断されました"
        case .systemInterruption:
            return "システムイベントによって中断されました"
        case .appTermination:
            return "アプリケーションが終了しました"
        case .unknown:
            return "予期しない理由でセッションが中断されました"
        }
    }

    /// 中断が回復可能かどうか
    var isRecoverable: Bool {
        switch self {
        case .cameraLost, .memoryPressure:
            return true
        case .backgroundTransition, .userInterruption, .systemInterruption:
            return false // 仕様により、継続はせず常に終了
        case .appTermination, .unknown:
            return false
        }
    }

    /// 部分結果を保存すべきかどうか（10秒以上経過している場合）
    var shouldSavePartialResult: Bool {
        switch self {
        case .userInterruption, .appTermination:
            return false // ユーザーの意図的な操作は保存しない
        default:
            return true // その他の中断は部分結果を保存
        }
    }

    /// 中断タイプのアイコン名（SF Symbols）
    var iconName: String {
        switch self {
        case .cameraLost:
            return "camera.fill.badge.ellipsis"
        case .memoryPressure:
            return "memorychip"
        case .backgroundTransition:
            return "app.badge"
        case .userInterruption:
            return "hand.raised.fill"
        case .systemInterruption:
            return "phone.fill"
        case .appTermination:
            return "xmark.app.fill"
        case .unknown:
            return "questionmark.circle.fill"
        }
    }
}

/// セッション中断の詳細情報
public struct SessionInterruptionInfo: Codable {
    /// 中断タイプ
    let type: SessionInterruptionType

    /// 中断発生時刻
    let timestamp: Date

    /// 中断時の経過時間（秒）
    let elapsedSeconds: Int

    /// 中断時の完了レップ数
    let completedReps: Int

    /// エラー情報（あれば）
    let errorDescription: String?

    /// 追加のコンテキスト情報
    let context: [String: String]?

    /// イニシャライザ
    init(
        type: SessionInterruptionType,
        timestamp: Date = Date(),
        elapsedSeconds: Int,
        completedReps: Int,
        errorDescription: String? = nil,
        context: [String: String]? = nil
    ) {
        self.type = type
        self.timestamp = timestamp
        self.elapsedSeconds = elapsedSeconds
        self.completedReps = completedReps
        self.errorDescription = errorDescription
        self.context = context
    }

    /// 部分結果として保存可能かどうか
    var canSaveAsPartialResult: Bool {
        // 10秒以上経過していて、保存すべきタイプの中断の場合
        return elapsedSeconds >= 10 && type.shouldSavePartialResult
    }

    /// ユーザーに表示するメッセージ
    var userMessage: String {
        var message = type.description

        if canSaveAsPartialResult {
            message += "\n\(elapsedSeconds)秒間のトレーニング結果を保存しました。"
        } else if elapsedSeconds < 10 {
            message += "\n10秒未満のため、結果は保存されませんでした。"
        }

        return message
    }
}