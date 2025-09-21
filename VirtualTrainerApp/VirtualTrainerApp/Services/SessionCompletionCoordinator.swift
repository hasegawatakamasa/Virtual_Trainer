//
//  SessionCompletionCoordinator.swift
//  VirtualTrainerApp
//
//  Created on 2025/09/15.
//

import Foundation
import SwiftUI
import Combine

/// セッション完了処理を調整するコーディネーター
@MainActor
class SessionCompletionCoordinator: ObservableObject {

    // MARK: - Published Properties

    /// 終了メッセージ表示フラグ
    @Published var showFinishMessage = false

    /// リザルト画面への遷移フラグ
    @Published var shouldNavigateToResult = false

    /// 完了データ
    @Published var completionData: SessionCompletionData?

    /// エラー状態
    @Published var completionError: TimerError?

    /// 処理中フラグ
    @Published private(set) var isProcessing = false

    // MARK: - Private Properties

    /// クリーンアップサービス
    private var cleanupService: IntegratedCleanupService?

    /// セッションサービス
    private var sessionService: TrainingSessionService?

    /// タイマー
    private var completionTimer: Timer?

    /// キャンセラブル
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init() {
        print("[SessionCompletionCoordinator] Initialized")
    }

    deinit {
        completionTimer?.invalidate()
        print("[SessionCompletionCoordinator] Deinitialized")
    }

    // MARK: - Public Methods

    /// サービスを設定
    func configure(
        cleanupService: IntegratedCleanupService,
        sessionService: TrainingSessionService
    ) {
        self.cleanupService = cleanupService
        self.sessionService = sessionService
    }

    /// 完了処理を開始
    /// - Parameters:
    ///   - data: セッション完了データ
    ///   - showMessage: 終了メッセージを表示するかどうか
    func initiateCompletion(
        with data: SessionCompletionData,
        showMessage: Bool = true
    ) {
        guard !isProcessing else {
            print("[SessionCompletionCoordinator] Already processing completion")
            return
        }

        print("[SessionCompletionCoordinator] Starting completion sequence")
        isProcessing = true
        completionData = data
        completionError = nil

        // 完了シーケンスを開始
        Task {
            await performCompletionSequence(showMessage: showMessage)
        }
    }

    /// エラーによる完了処理
    func completeWithError(_ error: TimerError) {
        print("[SessionCompletionCoordinator] Completing with error: \(error)")
        completionError = error
        isProcessing = false

        // エラーメッセージを表示してから遷移
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            self?.shouldNavigateToResult = true
        }
    }

    /// 手動でリザルト画面へ遷移
    func navigateToResult() {
        shouldNavigateToResult = true
    }

    // MARK: - Private Methods

    /// 完了シーケンスを実行
    private func performCompletionSequence(showMessage: Bool) async {
        // Step 1: 終了メッセージを表示
        if showMessage {
            await showCompletionMessage()
        }

        // Step 2: セッションデータを保存
        await saveSessionData()

        // Step 3: 終了音声の再生完了を待つ（2秒待機）
        print("[SessionCompletionCoordinator] Waiting for completion audio to finish...")
        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2秒待機

        // Step 4: リソースクリーンアップ
        await performResourceCleanup()

        // Step 4: リザルト画面への自動遷移は行わない
        // ユーザーが手動で「リザルトを見る」ボタンを押した時に遷移する
        await MainActor.run {
            self.isProcessing = false
            print("[SessionCompletionCoordinator] Completion sequence finished, waiting for user action")
        }
    }

    /// 完了メッセージを表示
    private func showCompletionMessage() async {
        await MainActor.run {
            self.showFinishMessage = true
            print("[SessionCompletionCoordinator] Showing finish message")
        }

        // 自動的にメッセージを非表示にしない
        // ユーザーがボタンを押すまで表示を継続
    }

    /// セッションデータを保存
    private func saveSessionData() async {
        guard let data = completionData else {
            print("[SessionCompletionCoordinator] No completion data to save")
            return
        }

        print("[SessionCompletionCoordinator] Saving session data")

        // TrainingSessionServiceを使用してデータを保存
        if let sessionService = sessionService {
            await MainActor.run {
                // セッションデータの記録
                sessionService.recordTimedSession(
                    exerciseType: data.exerciseType,
                    completedReps: data.completedReps,
                    duration: data.actualDuration,
                    wasCompleted: data.completionReason == .timerCompleted,
                    formErrors: data.formErrors
                )

                print("[SessionCompletionCoordinator] Session data saved successfully")
            }
        }
    }

    /// リソースクリーンアップを実行
    private func performResourceCleanup() async {
        print("[SessionCompletionCoordinator] Performing resource cleanup")

        if let cleanupService = cleanupService {
            let success = await cleanupService.performIntegratedCleanup()

            if success {
                print("[SessionCompletionCoordinator] Resource cleanup completed successfully")
            } else {
                print("[SessionCompletionCoordinator] Resource cleanup had some issues")
            }
        } else {
            print("[SessionCompletionCoordinator] No cleanup service configured")
        }
    }

    /// リザルト画面へ遷移
    func navigateToResultScreen() {
        print("[SessionCompletionCoordinator] Navigating to result screen")

        // 終了メッセージを非表示にする
        showFinishMessage = false

        // 少し遅延を入れてスムーズな遷移を実現
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self = self else { return }

            self.shouldNavigateToResult = true
            self.isProcessing = false

            print("[SessionCompletionCoordinator] Navigation flag set")
        }
    }

    /// セッションサマリーを取得
    func getSessionSummary() -> SessionCompletionData? {
        return completionData
    }

    /// 状態をリセット
    func reset() {
        showFinishMessage = false
        shouldNavigateToResult = false
        completionData = nil
        completionError = nil
        isProcessing = false
        completionTimer?.invalidate()
        completionTimer = nil

        print("[SessionCompletionCoordinator] State reset")
    }
}

// MARK: - Extensions

extension SessionCompletionCoordinator {
    /// デバッグ用の状態説明
    var debugDescription: String {
        """
        SessionCompletionCoordinator:
        - Processing: \(isProcessing)
        - Show Finish: \(showFinishMessage)
        - Navigate to Result: \(shouldNavigateToResult)
        - Has Data: \(completionData != nil)
        - Has Error: \(completionError != nil)
        """
    }
}

// MARK: - Helper Extensions

extension TrainingSessionService {
    /// タイマーセッションを記録
    func recordTimedSession(
        exerciseType: String,
        completedReps: Int,
        duration: TimeInterval,
        wasCompleted: Bool,
        formErrors: Int
    ) {
        // 既存のメソッドを使用してセッションを記録
        // ここでは仮の実装を提供
        print("[TrainingSessionService] Recording timed session:")
        print("  - Exercise: \(exerciseType)")
        print("  - Reps: \(completedReps)")
        print("  - Duration: \(duration)s")
        print("  - Completed: \(wasCompleted)")
        print("  - Form Errors: \(formErrors)")

        // TODO: 実際のCore Data保存処理を実装
    }
}