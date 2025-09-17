//
//  InterruptionHandler.swift
//  VirtualTrainerApp
//
//  Created on 2025/09/15.
//

import Foundation
import SwiftUI
import Combine
#if canImport(UIKit)
import UIKit
#endif

/// セッション中断処理サービス
/// トレーニングセッションの中断検知・管理・復旧処理を行う
@MainActor
class InterruptionHandler: ObservableObject {
    static let shared = InterruptionHandler()

    // MARK: - Published Properties

    /// 中断状態かどうか
    @Published var isInterrupted: Bool = false

    /// 中断からの復帰可能状態かどうか
    @Published var canRecover: Bool = false

    /// 復旧オプションを表示するかどうか
    @Published var showRecoveryOptions: Bool = false

    /// 中断情報メッセージ
    @Published var interruptionMessage: String = ""

    /// 保存された中断セッションデータ
    @Published var savedInterruptedSession: InterruptedSessionData?

    // MARK: - Private Properties

    /// セッション中断管理クラス
    private let interruptedSessionManager = InterruptedSessionManager()

    /// タイマー管理サービス（注入可能）
    private var sessionTimerManager: SessionTimerManager?

    /// トレーニングセッションサービス
    private let trainingSessionService = TrainingSessionService.shared

    /// Combineのキャンセル可能オブジェクトセット
    private var cancellables = Set<AnyCancellable>()

    /// 中断開始時刻
    private var interruptionStartTime: Date?

    /// 中断前のセッション状態
    private var sessionStateBeforeInterruption: SessionState?

    // MARK: - Initialization

    private init() {
        setupNotifications()
        checkForPreviousInterruption()
    }

    // MARK: - Public Methods

    /// セッションタイマーマネージャーを設定
    /// - Parameter timerManager: セッションタイマーマネージャー
    func configureWithTimerManager(_ timerManager: SessionTimerManager) {
        self.sessionTimerManager = timerManager
    }

    /// セッション中断を処理する
    /// - Parameters:
    ///   - type: 中断タイプ
    ///   - elapsedTime: 経過時間（秒）
    ///   - completedReps: 完了レップ数
    ///   - errorDescription: エラーの詳細説明（オプション）
    func handleInterruption(
        type: SessionInterruptionType,
        elapsedTime: TimeInterval,
        completedReps: Int,
        errorDescription: String? = nil
    ) {
        print("🚨 セッション中断を処理中: \(type.displayName)")

        let interruptionInfo = SessionInterruptionInfo(
            type: type,
            elapsedSeconds: Int(elapsedTime),
            completedReps: completedReps,
            errorDescription: errorDescription
        )

        // 中断状態を設定
        isInterrupted = true
        interruptionStartTime = Date()

        // 中断タイプを検知・分類
        let detectedType = detectInterruptionType(originalType: type, context: createInterruptionContext())

        // 部分結果の保存判定（10秒以上の場合）
        if elapsedTime >= 10.0 && detectedType.shouldSavePartialResult {
            savePartialSession(
                interruptionInfo: interruptionInfo,
                detectedType: detectedType
            )
            // 60秒タイマーセッションの場合は、タイマーを停止しない（継続させる）
            // ただし、セッションの設定時間を超えている場合は停止
            if elapsedTime < 60.0 {
                // タイマーがまだ完了していない場合は継続
                print("⏱️ タイマーセッション継続中: \(elapsedTime)秒経過")
            } else {
                // タイマーを停止
                sessionTimerManager?.stopTimer()
            }
        } else {
            // 10秒未満の場合は破棄
            discardSession(reason: "セッション時間が10秒未満のため破棄")
            interruptionMessage = "セッションが10秒未満のため、結果は保存されませんでした。"
            // タイマーを停止
            sessionTimerManager?.stopTimer()
        }

        // UI更新
        updateUIForInterruption(type: detectedType, info: interruptionInfo)
    }

    /// 中断からの復帰処理
    /// セッション再開時に「前回のセッションは中断されました」メッセージを表示
    func handleReturnFromInterruption() {
        guard let savedSession = savedInterruptedSession else { return }

        print("🔄 中断からの復帰を処理中")

        // 復帰メッセージを設定
        interruptionMessage = savedSession.recoveryMessage
        showRecoveryOptions = true
        canRecover = savedSession.isRecoverable

        // 復帰可能時間を過ぎている場合は自動的にクリア
        if !savedSession.isRecoverable {
            clearInterruptedSession()
            interruptionMessage = "前回のセッションは時間が経過しているため復旧できません。"
        }
    }

    /// 復旧オプションを表示する
    /// ユーザーに中断されたセッションの処理方法を選択させる
    func displayRecoveryOptions() {
        guard savedInterruptedSession != nil else { return }

        showRecoveryOptions = true
        canRecover = true
    }

    /// 中断されたセッションデータをクリアする
    func clearInterruptedSession() {
        savedInterruptedSession = nil
        interruptedSessionManager.clearSavedSession()

        // UI状態をリセット
        isInterrupted = false
        canRecover = false
        showRecoveryOptions = false
        interruptionMessage = ""

        print("🧹 中断セッションデータをクリアしました")
    }

    // MARK: - Private Methods

    /// 中断タイプを検知・分類する
    /// - Parameters:
    ///   - originalType: 元の中断タイプ
    ///   - context: 中断発生時のコンテキスト情報
    /// - Returns: 検知された中断タイプ
    private func detectInterruptionType(
        originalType: SessionInterruptionType,
        context: [String: String]
    ) -> SessionInterruptionType {
        // コンテキスト情報を基に中断タイプを詳細に分類

        // メモリ不足の詳細検知
        if context["memoryWarning"] == "true" {
            return .memoryPressure
        }

        // バックグラウンド遷移の詳細検知
        if context["backgroundTransition"] == "true" {
            return .backgroundTransition
        }

        // システム中断の詳細検知（電話着信など）
        if context["audioSessionInterruption"] == "true" {
            return .systemInterruption
        }

        // カメラセッション関連の問題
        if context["cameraSessionError"] == "true" {
            return .cameraLost
        }

        // デフォルトは元のタイプを返す
        return originalType
    }

    /// 部分セッションを保存する
    /// - Parameters:
    ///   - interruptionInfo: 中断情報
    ///   - detectedType: 検知された中断タイプ
    private func savePartialSession(
        interruptionInfo: SessionInterruptionInfo,
        detectedType: SessionInterruptionType
    ) {
        guard let currentSession = trainingSessionService.currentSession,
              let startTime = currentSession.startTime else {
            print("⚠️ 保存可能な現在のセッションが見つかりません")
            return
        }

        // 中断セッションデータを作成
        let interruptedData = InterruptedSessionData(
            startTime: startTime,
            elapsedTime: TimeInterval(interruptionInfo.elapsedSeconds),
            completedReps: interruptionInfo.completedReps,
            formErrors: Int(currentSession.formErrors),
            interruptionType: detectedType,
            exerciseType: currentSession.exerciseType ?? "unknown",
            configuration: sessionTimerManager?.configuration,
            interruptionMessage: interruptionInfo.userMessage,
            wasPartiallySaved: true
        )

        // データを保存
        interruptedSessionManager.saveInterruptedSession(interruptedData)
        savedInterruptedSession = interruptedData

        // トレーニングセッションサービスで部分結果を保存
        trainingSessionService.saveCurrentSessionState()

        print("💾 部分セッションを保存しました: \(interruptionInfo.elapsedSeconds)秒, \(interruptionInfo.completedReps)回")

        // UI更新用のメッセージ
        interruptionMessage = interruptedData.recoveryMessage
    }

    /// セッションを破棄する
    /// - Parameter reason: 破棄理由
    private func discardSession(reason: String) {
        print("🗑️ セッションを破棄: \(reason)")

        // 現在のセッションをキャンセル
        trainingSessionService.cancelSession()

        // 保存された中断データもクリア
        clearInterruptedSession()

        // 破棄後に新しいセッションを自動開始
        // 現在のキャラクターを取得
        let currentCharacter = VoiceCharacter(rawValue: UserDefaults.standard.string(forKey: "selectedCharacter") ?? "ずんだもん") ?? .zundamon

        // 新しいセッションを開始
        trainingSessionService.startSession(
            exerciseType: .overheadPress,
            voiceCharacter: currentCharacter
        )

        // タイマーも再初期化（待機状態に戻す）
        sessionTimerManager?.reset()
        sessionTimerManager?.startWaitingForRep()

        // セッション再開を通知
        NotificationCenter.default.post(name: Notification.Name("SessionRestarted"), object: nil)

        print("🔄 新しいセッションを自動開始しました（タイマーリセット済み）")
    }

    /// 中断時のコンテキスト情報を作成
    /// - Returns: コンテキスト辞書
    private func createInterruptionContext() -> [String: String] {
        var context: [String: String] = [:]

        // メモリ警告状態
        context["memoryWarning"] = "\(ProcessInfo.processInfo.isLowPowerModeEnabled)"

        // バックグラウンド状態
        #if os(iOS)
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
            context["backgroundTransition"] = "\(windowScene.activationState != .foregroundActive)"
        }
        #else
        context["backgroundTransition"] = "false"  // macOSでは常にフォアグラウンドとして扱う
        #endif

        // タイマー状態
        if let timerManager = sessionTimerManager {
            context["timerState"] = timerManager.timerState.rawValue
            context["remainingTime"] = "\(timerManager.remainingTime)"
        }

        // セッション状態
        context["sessionActive"] = "\(trainingSessionService.isSessionActive)"

        return context
    }

    /// 中断時のUI更新
    /// - Parameters:
    ///   - type: 中断タイプ
    ///   - info: 中断情報
    private func updateUIForInterruption(type: SessionInterruptionType, info: SessionInterruptionInfo) {
        interruptionMessage = info.userMessage

        // 復旧可能性の判定
        canRecover = type.isRecoverable && info.canSaveAsPartialResult

        // 自動的に復旧オプションを表示するかどうか
        if canRecover {
            showRecoveryOptions = true
        }
    }

    /// 前回の中断セッションをチェック
    private func checkForPreviousInterruption() {
        if let savedData = interruptedSessionManager.loadInterruptedSession() {
            savedInterruptedSession = savedData

            // 復旧可能時間内かチェック
            if savedData.isRecoverable {
                print("🔍 前回の中断セッションを検出: \(savedData.summary)")
            } else {
                // 時間が経過している場合は自動削除
                clearInterruptedSession()
                print("🕐 前回の中断セッションは時間が経過しているため削除されました")
            }
        }
    }

    /// 通知の設定
    private func setupNotifications() {
        #if os(iOS)
        // メモリ警告の通知
        NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.handleMemoryWarning()
                }
            }
            .store(in: &cancellables)

        // バックグラウンド遷移の通知
        NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.handleBackgroundTransition()
                }
            }
            .store(in: &cancellables)
        #else
        // macOSではメモリ警告通知は非対応
        print("[InterruptionHandler] Memory warning notifications not available on macOS")
        #endif
    }

    /// メモリ警告の処理
    private func handleMemoryWarning() {
        guard trainingSessionService.isSessionActive else { return }

        print("⚠️ メモリ警告を検知")

        // 現在のセッション情報を取得
        let elapsedTime = trainingSessionService.sessionStats.sessionDuration
        let completedReps = trainingSessionService.sessionStats.totalReps

        // メモリ不足による中断として処理
        handleInterruption(
            type: .memoryPressure,
            elapsedTime: elapsedTime,
            completedReps: completedReps,
            errorDescription: "システムのメモリ不足により中断されました"
        )
    }

    /// バックグラウンド遷移の処理
    private func handleBackgroundTransition() {
        guard trainingSessionService.isSessionActive else { return }

        print("📱 バックグラウンド遷移を検知")

        // 現在のセッション情報を取得
        let elapsedTime = trainingSessionService.sessionStats.sessionDuration
        let completedReps = trainingSessionService.sessionStats.totalReps

        // バックグラウンド遷移による中断として処理
        handleInterruption(
            type: .backgroundTransition,
            elapsedTime: elapsedTime,
            completedReps: completedReps,
            errorDescription: "アプリがバックグラウンドに移行しました"
        )
    }
}

// MARK: - InterruptedSessionManager

/// 中断されたセッションデータの永続化管理クラス
private class InterruptedSessionManager {

    /// 中断セッションデータを保存
    /// - Parameter data: 保存する中断セッションデータ
    func saveInterruptedSession(_ data: InterruptedSessionData) {
        data.save()
        print("💾 中断セッションデータを保存しました: \(data.sessionId)")
    }

    /// 保存された中断セッションデータを読み込み
    /// - Returns: 中断セッションデータ（存在しない場合はnil）
    func loadInterruptedSession() -> InterruptedSessionData? {
        let data = InterruptedSessionData.load()
        if let data = data {
            print("📁 中断セッションデータを読み込みました: \(data.sessionId)")
        }
        return data
    }

    /// 保存された中断セッションデータをクリア
    func clearSavedSession() {
        InterruptedSessionData.clear()
        print("🧹 保存された中断セッションデータをクリアしました")
    }
}

// MARK: - Supporting Data Structures

/// セッション状態の内部表現
private struct SessionState {
    let isActive: Bool
    let remainingTime: Int
    let completedReps: Int
    let startTime: Date

    @MainActor
    init(from trainingService: TrainingSessionService, timerManager: SessionTimerManager?) {
        self.isActive = trainingService.isSessionActive
        self.remainingTime = timerManager?.remainingTime ?? 0
        self.completedReps = trainingService.sessionStats.totalReps
        self.startTime = trainingService.currentSession?.startTime ?? Date()
    }
}

// MARK: - Extensions

extension InterruptionHandler {
    /// デバッグ情報を取得
    var debugInfo: String {
        return """
        InterruptionHandler Debug Info:
        - Is Interrupted: \(isInterrupted)
        - Can Recover: \(canRecover)
        - Show Recovery Options: \(showRecoveryOptions)
        - Has Saved Session: \(savedInterruptedSession != nil)
        - Interruption Message: \(interruptionMessage)
        - Interruption Start Time: \(interruptionStartTime?.description ?? "None")
        """
    }
}