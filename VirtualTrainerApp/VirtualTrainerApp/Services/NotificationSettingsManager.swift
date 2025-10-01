import Foundation
import Combine

/// 通知設定管理サービス
/// 要件対応: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
@MainActor
class NotificationSettingsManager: ObservableObject {
    @Published var settings: NotificationSettings
    @Published var isCalendarConnected: Bool = false

    private let userDefaults: UserDefaults
    private let notificationService: OshiTrainerNotificationService
    private let apiClient: GoogleCalendarAPIClient?
    private let analyzer: CalendarEventAnalyzer
    private let scheduler: NotificationScheduler

    init(
        userDefaults: UserDefaults = .standard,
        notificationService: OshiTrainerNotificationService,
        apiClient: GoogleCalendarAPIClient? = nil,
        analyzer: CalendarEventAnalyzer = CalendarEventAnalyzer()
    ) {
        self.userDefaults = userDefaults
        self.notificationService = notificationService
        self.apiClient = apiClient
        self.analyzer = analyzer
        self.scheduler = NotificationScheduler()

        // 設定を読み込み
        if let data = userDefaults.data(forKey: UserDefaultsKeys.notificationSettings),
           let savedSettings = try? JSONDecoder().decode(NotificationSettings.self, from: data) {
            self.settings = savedSettings
        } else {
            self.settings = NotificationSettings()
        }

        // カレンダー連携状態を読み込み
        self.isCalendarConnected = userDefaults.bool(forKey: UserDefaultsKeys.isCalendarConnected)
    }

    // MARK: - Settings Management

    /// 通知設定を保存
    /// - Parameter settings: 新しい設定
    func saveSettings(_ settings: NotificationSettings) async {
        self.settings = settings

        if let data = try? JSONEncoder().encode(settings) {
            userDefaults.set(data, forKey: UserDefaultsKeys.notificationSettings)
        }

        // 設定変更時に通知を再スケジュール
        if settings.enabled {
            do {
                try await applySettingsChange()
            } catch {
                print("[NotificationSettingsManager] Failed to apply settings change: \(error)")
            }
        }
    }

    /// 通知を有効化
    func enableNotifications() async throws {
        // 通知権限リクエスト
        let granted = try await notificationService.requestNotificationPermission()
        guard granted else {
            throw NotificationSchedulingError.notificationPermissionDenied
        }

        settings.enabled = true
        await saveSettings(settings)

        // 通知をスケジュール
        if isCalendarConnected {
            try await scheduleCalendarBasedNotifications()
        } else {
            try await applyDefaultSchedule()
        }
    }

    /// 通知を無効化（全通知キャンセル）
    func disableNotifications() async {
        settings.enabled = false
        await saveSettings(settings)
        await notificationService.cancelAllNotifications()
    }

    /// 設定変更時にスケジュールを再計算
    func applySettingsChange() async throws {
        guard settings.enabled else { return }

        // 既存の通知をキャンセル
        await notificationService.cancelAllNotifications()

        // 新しい設定で再スケジュール
        if isCalendarConnected {
            try await scheduleCalendarBasedNotifications()
        } else {
            try await applyDefaultSchedule()
        }
    }

    /// デフォルト通知スケジュールを適用（カレンダー未連携時）
    func applyDefaultSchedule() async throws {
        let now = Date()
        let calendar = Calendar.current
        var defaultSlots: [AvailableTimeSlot] = []

        // 今日から7日分のデフォルトスロット生成（10:00, 18:00）
        for dayOffset in 0..<7 {
            guard let date = calendar.date(byAdding: .day, value: dayOffset, to: calendar.startOfDay(for: now)) else {
                continue
            }

            let defaultHours = [10, 18]
            for hour in defaultHours {
                guard let startTime = calendar.date(bySettingHour: hour, minute: 0, second: 0, of: date),
                      startTime > now else {
                    continue
                }

                let endTime = startTime.addingTimeInterval(2 * 3600)
                let slot = AvailableTimeSlot(
                    startTime: startTime,
                    endTime: endTime,
                    duration: 2 * 3600,
                    slotType: .freeDay
                )
                defaultSlots.append(slot)
            }
        }

        // 通知候補を生成
        let candidates = scheduler.generateNotificationCandidates(from: defaultSlots, settings: settings)

        // 通知をスケジュール
        _ = try await notificationService.scheduleNotifications(candidates)
    }

    /// カレンダーベースの通知スケジュールを生成
    private func scheduleCalendarBasedNotifications() async throws {
        guard let apiClient = apiClient else {
            throw GoogleCalendarError.authenticationFailed(reason: "API client not initialized")
        }

        let now = Date()
        let endDate = Calendar.current.date(byAdding: .day, value: 7, to: now)!

        // カレンダーイベントを取得
        let events = try await apiClient.fetchEvents(from: now, to: endDate)

        // 空き時間を検出
        let slots = analyzer.analyzeAvailableSlots(events: events)

        // 通知候補を生成
        let candidates = scheduler.generateNotificationCandidates(from: slots, settings: settings)

        // 通知をスケジュール
        _ = try await notificationService.scheduleNotifications(candidates)
    }

    /// 通知時間帯フィルタを適用
    /// - Parameters:
    ///   - candidates: 通知候補
    ///   - timeRange: 許可時間帯
    /// - Returns: フィルタ後の候補
    func filterByTimeRange(_ candidates: [NotificationCandidate], timeRange: NotificationTimeRange) -> [NotificationCandidate] {
        return candidates.filter { timeRange.contains($0.scheduledTime) }
    }
}