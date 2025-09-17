import Foundation
import Combine
import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

/// セッションタイマー管理サービス
/// タイマー機能付きトレーニングセッションの時間管理を行う
@MainActor
class SessionTimerManager: ObservableObject {

    // MARK: - Published Properties

    /// 残り時間（秒）
    @Published var remainingTime: Int

    /// タイマーの現在の状態
    @Published var timerState: TimerState = .notStarted

    /// 最後の10秒間かどうか（UI効果用）
    @Published var isLastTenSeconds: Bool = false

    /// 開始メッセージを表示するかどうか
    @Published var showStartMessage: Bool = false

    /// 手動開始可能かどうか
    @Published var canStartManually: Bool = false

    /// タイマーセッションの設定
    @Published var configuration: TimedSessionConfiguration

    // MARK: - Private Properties

    /// Combine用のタイマーパブリッシャー
    private var timerPublisher: Timer.TimerPublisher?

    /// タイマー購読用のCancellable
    private var timerCancellable: AnyCancellable?

    /// その他の購読用のCancellableセット
    private var cancellables = Set<AnyCancellable>()

    /// 開始メッセージ表示用のタスク
    private var startMessageTask: Task<Void, Never>?

    /// 手動開始用のタイマー（30秒後に手動開始ボタンを表示）
    private var manualStartTimer: Timer?

    /// バックグラウンド移行時刻
    private var backgroundTime: Date?

    /// タイマー開始時刻（バックグラウンド復帰時の時間調整用）
    private var timerStartTime: Date?

    /// 再生済みマイルストーンの記録
    private var playedMilestones: Set<TimerMilestone> = []

    // MARK: - Initialization

    /// SessionTimerManagerを初期化
    /// - Parameter configuration: タイマー設定（デフォルト設定を使用）
    init(configuration: TimedSessionConfiguration = .default) {
        self.configuration = configuration
        self.remainingTime = Int(configuration.duration)

        setupNotifications()
    }

    deinit {
        // deinitでは最小限の処理のみ
        // TimerとTaskは自動的に無効化される
    }

    // MARK: - Public Methods

    /// 最初のレップ検知時の処理
    /// 開始メッセージとタイマーを同時に開始する
    func handleFirstRep() {
        guard timerState == .waitingForRep else { return }

        // タイマーを即座に開始
        startTimerCountdown()

        // 開始メッセージを表示（タイマーと並行）
        if configuration.showStartMessage {
            showStartMessage = true

            // 2秒後にメッセージを非表示にする
            startMessageTask = Task {
                try? await Task.sleep(nanoseconds: 2_000_000_000) // 2秒

                await MainActor.run {
                    showStartMessage = false
                }
            }
        }
    }

    /// 手動開始ボタン押下時の処理
    func manualStart() {
        guard timerState == .waitingForRep || timerState == .notStarted else { return }

        // 手動開始タイマーをキャンセル
        manualStartTimer?.invalidate()
        manualStartTimer = nil

        // タイマーを即座に開始
        startTimerCountdown()

        // 開始メッセージを表示（タイマーと並行）
        if configuration.showStartMessage {
            showStartMessage = true

            startMessageTask = Task {
                try? await Task.sleep(nanoseconds: 2_000_000_000) // 2秒

                await MainActor.run {
                    showStartMessage = false
                }
            }
        }
    }

    /// タイマーを一時停止
    func pauseTimer() {
        guard timerState == .running else { return }

        timerState = .paused
        stopTimerPublisher()
    }

    /// タイマーを再開
    func resumeTimer() {
        guard timerState == .paused else { return }

        timerState = .running
        startTimerPublisher()
    }

    /// タイマーを停止・キャンセル
    func stopTimer() {
        timerState = .cancelled
        cleanup()
    }

    /// タイマー完了時の処理
    func handleTimerCompletion() {
        timerState = .completed
        cleanup()

        // マイルストーン音声再生（完了音声）
        if configuration.enabledMilestones.contains(.zero) &&
           !playedMilestones.contains(.zero) {
            playedMilestones.insert(.zero)
            // 音声再生処理はAudioFeedbackServiceなど外部サービスに委譲
        }
    }

    /// レップ待機状態を開始
    /// タイマー開始前の準備状態
    func startWaitingForRep() {
        timerState = .waitingForRep
        remainingTime = Int(configuration.duration)
        playedMilestones.removeAll()

        // 30秒後に手動開始オプションを提示（手動開始が有効な場合）
        if configuration.startTrigger == .manual || configuration.startTrigger == .waitingForRep {
            manualStartTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    self?.canStartManually = true
                }
            }
        }
    }

    /// タイマーをリセット
    func reset() {
        timerState = .notStarted
        remainingTime = Int(configuration.duration)
        isLastTenSeconds = false
        showStartMessage = false
        playedMilestones.removeAll()

        cleanup()
    }

    /// 設定を更新
    /// - Parameter newConfiguration: 新しい設定
    func updateConfiguration(_ newConfiguration: TimedSessionConfiguration) {
        guard newConfiguration.isValid else { return }

        configuration = newConfiguration

        // タイマーが非アクティブな場合のみ残り時間を更新
        if !timerState.isActive {
            remainingTime = Int(newConfiguration.duration)
        }
    }

    // MARK: - Private Methods

    /// タイマーのカウントダウンを開始
    private func startTimerCountdown() {
        timerState = .running
        timerStartTime = Date()
        startTimerPublisher()
    }

    /// Combineタイマーパブリッシャーを開始
    private func startTimerPublisher() {
        // 既存のタイマーを停止
        stopTimerPublisher()

        // 新しいタイマーを作成（1秒間隔）
        timerPublisher = Timer.publish(every: 1.0, on: .main, in: .common)
        timerCancellable = timerPublisher?
            .autoconnect()
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.updateTimer()
                }
            }
    }

    /// タイマーパブリッシャーを停止
    private func stopTimerPublisher() {
        timerCancellable?.cancel()
        timerCancellable = nil
        timerPublisher = nil
    }

    /// タイマーの更新処理（1秒ごとに呼ばれる）
    private func updateTimer() {
        guard timerState == .running else { return }

        // 残り時間を1秒減少
        remainingTime -= 1

        // 最後の10秒判定を更新
        isLastTenSeconds = remainingTime <= 10

        // マイルストーンのチェックと音声再生
        checkAndPlayMilestone()

        // タイマー完了チェック
        if remainingTime <= 0 {
            handleTimerCompletion()
        }
    }

    /// マイルストーンをチェックして音声を再生
    private func checkAndPlayMilestone() {
        guard configuration.playCountdownAudio else { return }

        // 現在の残り時間に対応するマイルストーンを検索
        for milestone in configuration.enabledMilestones {
            if remainingTime == Int(milestone.remainingTime) &&
               !playedMilestones.contains(milestone) {
                playedMilestones.insert(milestone)

                // 実際の音声再生は外部サービスに委譲
                // AudioFeedbackServiceなどで milestone.audioKey を使用して音声を再生
                print("🔔 マイルストーン音声再生: \(milestone.displayName)")
                break
            }
        }
    }

    /// アプリライフサイクル通知の設定
    private func setupNotifications() {
        #if os(iOS)
        // アプリがバックグラウンドに移行する通知
        NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.handleAppWillResignActive()
                }
            }
            .store(in: &cancellables)

        // アプリがフォアグラウンドに復帰する通知
        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.handleAppWillEnterForeground()
                }
            }
            .store(in: &cancellables)
        #else
        // macOSの場合は異なる通知を使用（AppKitを使用）
        // 現時点ではmacOS非対応とする
        print("[SessionTimerManager] Background/Foreground notifications not configured for macOS")
        #endif
    }

    /// アプリがバックグラウンドに移行する際の処理
    private func handleAppWillResignActive() {
        if timerState == .running {
            backgroundTime = Date()
        }
    }

    /// アプリがフォアグラウンドに復帰する際の処理
    private func handleAppWillEnterForeground() {
        guard let backgroundTime = self.backgroundTime,
              let timerStartTime = self.timerStartTime,
              timerState == .running else {
            return
        }

        // バックグラウンド時間を考慮してタイマーを調整
        _ = Date().timeIntervalSince(backgroundTime)  // バックグラウンド時間（デバッグ用）
        let totalElapsed = Date().timeIntervalSince(timerStartTime)
        let adjustedRemainingTime = Int(configuration.duration - totalElapsed)

        if adjustedRemainingTime <= 0 {
            // バックグラウンド中に完了していた場合
            remainingTime = 0
            handleTimerCompletion()
        } else {
            // 残り時間を調整して継続
            remainingTime = adjustedRemainingTime
            isLastTenSeconds = remainingTime <= 10
        }

        self.backgroundTime = nil
    }

    /// リソースのクリーンアップ
    private func cleanup() {
        stopTimerPublisher()
        startMessageTask?.cancel()
        startMessageTask = nil
        manualStartTimer?.invalidate()
        manualStartTimer = nil
        backgroundTime = nil
        timerStartTime = nil
    }

}

// MARK: - Extensions

extension SessionTimerManager {
    /// 残り時間の表示用フォーマット（MM:SS形式）
    var formattedRemainingTime: String {
        let minutes = remainingTime / 60
        let seconds = remainingTime % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }

    /// 進捗率（0.0 - 1.0）
    var progress: Double {
        let totalDuration = configuration.duration
        let elapsed = totalDuration - Double(remainingTime)
        return min(max(elapsed / totalDuration, 0.0), 1.0)
    }

    /// タイマーがアクティブ状態かどうか
    var isTimerActive: Bool {
        return timerState.isActive
    }

    /// タイマーが一時停止可能かどうか
    var canPause: Bool {
        return timerState == .running
    }

    /// タイマーが再開可能かどうか
    var canResume: Bool {
        return timerState == .paused
    }

    /// 手動開始ボタンを表示するかどうか
    var shouldShowManualStart: Bool {
        return timerState == .waitingForRep &&
               (configuration.startTrigger == .manual || configuration.startTrigger == .waitingForRep)
    }
}

// MARK: - Debug Support

extension SessionTimerManager {
    /// デバッグ情報を取得
    var debugInfo: String {
        return """
        SessionTimerManager Debug Info:
        - State: \(timerState.displayName)
        - Remaining Time: \(formattedRemainingTime)
        - Configuration: \(configuration.displaySummary)
        - Progress: \(String(format: "%.1f", progress * 100))%
        - Is Last Ten Seconds: \(isLastTenSeconds)
        - Show Start Message: \(showStartMessage)
        - Played Milestones: \(playedMilestones.map { $0.displayName }.joined(separator: ", "))
        """
    }
}