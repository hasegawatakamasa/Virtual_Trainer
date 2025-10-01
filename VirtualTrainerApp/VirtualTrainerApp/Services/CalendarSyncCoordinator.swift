import Foundation
import BackgroundTasks

/// カレンダー同期調整サービス
/// 要件対応: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
class CalendarSyncCoordinator {
    private let apiClient: GoogleCalendarAPIClient
    private let analyzer: CalendarEventAnalyzer
    private let scheduler: NotificationScheduler
    private let notificationService: OshiTrainerNotificationService
    private let userDefaults: UserDefaults

    private let backgroundTaskIdentifier = "com.virtualtrainer.calendar.sync"
    private let syncIntervalHours: TimeInterval = 4 * 3600  // 4時間

    init(
        apiClient: GoogleCalendarAPIClient,
        analyzer: CalendarEventAnalyzer = CalendarEventAnalyzer(),
        scheduler: NotificationScheduler = NotificationScheduler(),
        notificationService: OshiTrainerNotificationService,
        userDefaults: UserDefaults = .standard
    ) {
        self.apiClient = apiClient
        self.analyzer = analyzer
        self.scheduler = scheduler
        self.notificationService = notificationService
        self.userDefaults = userDefaults
    }

    // MARK: - Background Task Management

    /// バックグラウンド更新タスクを登録
    func registerBackgroundTask() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: backgroundTaskIdentifier,
            using: nil
        ) { task in
            self.handleAppRefresh(task: task as! BGAppRefreshTask)
        }

        print("[CalendarSyncCoordinator] Background task registered: \(backgroundTaskIdentifier)")
    }

    /// BGAppRefreshTask ハンドラ
    /// - Parameter task: バックグラウンドタスク
    func handleAppRefresh(task: BGAppRefreshTask) {
        // タイムアウトハンドラ設定（30秒）
        task.expirationHandler = {
            print("[CalendarSyncCoordinator] Background task expired")
            task.setTaskCompleted(success: false)
        }

        Task {
            do {
                let success = try await syncCalendar()
                task.setTaskCompleted(success: success)

                // 次回タスクをスケジュール
                scheduleNextBackgroundTask()
            } catch {
                print("[CalendarSyncCoordinator] Sync failed: \(error)")
                task.setTaskCompleted(success: false)
            }
        }
    }

    /// 次回のバックグラウンドタスクをスケジュール
    func scheduleNextBackgroundTask() {
        let request = BGAppRefreshTaskRequest(identifier: backgroundTaskIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: syncIntervalHours)

        do {
            try BGTaskScheduler.shared.submit(request)
            print("[CalendarSyncCoordinator] Next background task scheduled: \(request.earliestBeginDate!)")
        } catch {
            print("[CalendarSyncCoordinator] Failed to schedule background task: \(error)")
        }
    }

    // MARK: - Calendar Sync

    /// カレンダー同期を実行
    /// - Returns: 同期成功フラグ
    @discardableResult
    func syncCalendar() async throws -> Bool {
        print("[CalendarSyncCoordinator] Starting calendar sync...")

        // 前回のイベントを取得
        let oldEvents = getCachedEvents()

        // 新しいイベントを取得
        let now = Date()
        let endDate = Calendar.current.date(byAdding: .day, value: 7, to: now)!
        let newEvents = try await apiClient.fetchEvents(from: now, to: endDate)

        // イベント変更を検出
        let hasChanges = detectChanges(oldEvents: oldEvents, newEvents: newEvents)

        if hasChanges {
            print("[CalendarSyncCoordinator] Calendar changes detected, updating notifications...")

            // キャッシュを更新
            cacheEvents(newEvents)

            // 通知スケジュールを再生成
            await notificationService.cancelAllNotifications()

            // 新しいスロットを検出
            let slots = analyzer.analyzeAvailableSlots(events: newEvents)

            // 通知設定を取得
            if let settingsData = userDefaults.data(forKey: UserDefaultsKeys.notificationSettings),
               let settings = try? JSONDecoder().decode(NotificationSettings.self, from: settingsData),
               settings.enabled {

                // 通知候補を生成
                let candidates = scheduler.generateNotificationCandidates(from: slots, settings: settings)

                // 通知をスケジュール
                _ = try await notificationService.scheduleNotifications(candidates)
            }
        } else {
            print("[CalendarSyncCoordinator] No calendar changes detected")
        }

        // 同期時刻を記録
        recordSyncTime(Date())

        return true
    }

    /// イベント変更を検出
    /// - Parameters:
    ///   - oldEvents: 前回のイベント
    ///   - newEvents: 新しいイベント
    /// - Returns: 変更検出フラグ
    func detectChanges(oldEvents: [CalendarEvent], newEvents: [CalendarEvent]) -> Bool {
        // イベント数が異なる場合は変更あり
        guard oldEvents.count == newEvents.count else {
            return true
        }

        // IDと時刻で比較
        let oldEventMap = Dictionary(uniqueKeysWithValues: oldEvents.map { ($0.id, $0) })
        let newEventMap = Dictionary(uniqueKeysWithValues: newEvents.map { ($0.id, $0) })

        // 新しいイベントが追加されたか
        for newEvent in newEvents {
            if let oldEvent = oldEventMap[newEvent.id] {
                // 時刻が変更されたか
                if oldEvent.startTime != newEvent.startTime || oldEvent.endTime != newEvent.endTime {
                    return true
                }
            } else {
                // 新しいイベント
                return true
            }
        }

        // 削除されたイベントがあるか
        for oldEvent in oldEvents {
            if newEventMap[oldEvent.id] == nil {
                return true
            }
        }

        return false
    }

    /// 最終同期時刻を記録
    /// - Parameter date: 同期時刻
    func recordSyncTime(_ date: Date) {
        userDefaults.set(date, forKey: UserDefaultsKeys.calendarLastSyncTime)
    }

    /// 最終同期時刻を取得
    func getLastSyncTime() -> Date? {
        return userDefaults.object(forKey: UserDefaultsKeys.calendarLastSyncTime) as? Date
    }

    /// フォアグラウンド同期をスケジュール
    func scheduleForegroundSync() {
        Task {
            do {
                try await syncCalendar()
            } catch {
                print("[CalendarSyncCoordinator] Foreground sync failed: \(error)")
            }
        }
    }

    // MARK: - Event Caching

    private let cachedEventsKey = "com.virtualtrainer.cachedCalendarEvents"

    private func cacheEvents(_ events: [CalendarEvent]) {
        if let data = try? JSONEncoder().encode(events) {
            userDefaults.set(data, forKey: cachedEventsKey)
        }
    }

    private func getCachedEvents() -> [CalendarEvent] {
        guard let data = userDefaults.data(forKey: cachedEventsKey),
              let events = try? JSONDecoder().decode([CalendarEvent].self, from: data) else {
            return []
        }
        return events
    }
}