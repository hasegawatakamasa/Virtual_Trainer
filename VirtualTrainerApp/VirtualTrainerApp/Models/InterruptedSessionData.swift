//
//  InterruptedSessionData.swift
//  VirtualTrainerApp
//
//  Created on 2025/09/15.
//

import Foundation

/// 中断されたセッションのデータ
public struct InterruptedSessionData: Codable {
    /// セッション識別子
    let sessionId: UUID

    /// 開始時刻
    let startTime: Date

    /// 中断時刻
    let interruptionTime: Date

    /// 経過時間（秒）
    let elapsedTime: TimeInterval

    /// 完了したレップ数
    let completedReps: Int

    /// フォームエラー回数
    let formErrors: Int

    /// 中断タイプ
    let interruptionType: SessionInterruptionType

    /// エクササイズタイプ
    let exerciseType: String

    /// セッション設定
    let configuration: TimedSessionConfiguration?

    /// 中断時のメッセージ
    let interruptionMessage: String?

    /// セッションが部分的に保存されたかどうか
    let wasPartiallySaved: Bool

    /// イニシャライザ
    init(
        sessionId: UUID = UUID(),
        startTime: Date,
        interruptionTime: Date = Date(),
        elapsedTime: TimeInterval,
        completedReps: Int,
        formErrors: Int = 0,
        interruptionType: SessionInterruptionType,
        exerciseType: String,
        configuration: TimedSessionConfiguration? = nil,
        interruptionMessage: String? = nil,
        wasPartiallySaved: Bool = false
    ) {
        self.sessionId = sessionId
        self.startTime = startTime
        self.interruptionTime = interruptionTime
        self.elapsedTime = elapsedTime
        self.completedReps = completedReps
        self.formErrors = formErrors
        self.interruptionType = interruptionType
        self.exerciseType = exerciseType
        self.configuration = configuration
        self.interruptionMessage = interruptionMessage
        self.wasPartiallySaved = wasPartiallySaved
    }

    /// 部分結果として保存可能かどうか
    var canSaveAsPartialResult: Bool {
        // 10秒以上経過していて、中断タイプが保存可能な場合
        return elapsedTime >= 10 && interruptionType.shouldSavePartialResult
    }

    /// 完了率（パーセント）
    var completionRate: Double {
        guard let config = configuration else { return 0 }
        return (elapsedTime / Double(config.duration)) * 100
    }

    /// セッションの概要説明
    var summary: String {
        let timeStr = String(format: "%.0f秒", elapsedTime)
        let repsStr = "\(completedReps)回"
        let completionStr = String(format: "%.0f%%完了", completionRate)

        return "\(exerciseType): \(timeStr) / \(repsStr) (\(completionStr))"
    }

    /// 復帰時に表示するメッセージ
    var recoveryMessage: String {
        var message = "前回のセッションは\(interruptionType.displayName)により中断されました。\n"

        if wasPartiallySaved {
            message += "✅ \(Int(elapsedTime))秒間の記録（\(completedReps)回）は保存されています。"
        } else if elapsedTime < 10 {
            message += "⚠️ 10秒未満のため、記録は保存されませんでした。"
        } else {
            message += "❌ セッションの記録は保存されませんでした。"
        }

        return message
    }

    /// UserDefaultsに保存するためのキー
    static let userDefaultsKey = "interruptedSessionData"

    /// UserDefaultsに保存
    func save() {
        if let encoded = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(encoded, forKey: Self.userDefaultsKey)
        }
    }

    /// UserDefaultsから読み込み
    static func load() -> InterruptedSessionData? {
        guard let data = UserDefaults.standard.data(forKey: userDefaultsKey),
              let decoded = try? JSONDecoder().decode(InterruptedSessionData.self, from: data) else {
            return nil
        }
        return decoded
    }

    /// UserDefaultsから削除
    static func clear() {
        UserDefaults.standard.removeObject(forKey: userDefaultsKey)
    }

    /// 中断からの経過時間
    var timeSinceInterruption: TimeInterval {
        return Date().timeIntervalSince(interruptionTime)
    }

    /// 中断から復帰可能な時間内かどうか（例: 5分以内）
    var isRecoverable: Bool {
        return timeSinceInterruption < 300 // 5分以内
    }
}