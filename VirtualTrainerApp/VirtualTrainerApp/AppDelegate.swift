import UIKit
import UserNotifications
import BackgroundTasks

/// アプリケーションライフサイクル管理
class AppDelegate: NSObject, UIApplicationDelegate {
    private let analyticsService = NotificationAnalyticsService()
    private var syncCoordinator: CalendarSyncCoordinator?

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        // 通知センターのデリゲートを設定
        UNUserNotificationCenter.current().delegate = self

        // 通知カテゴリを登録
        registerNotificationCategories()

        // バックグラウンドタスクを登録
        registerBackgroundTasks()

        // カレンダー同期コーディネーターを初期化
        initializeSyncCoordinator()

        print("[AppDelegate] Application did finish launching")
        return true
    }

    // MARK: - Notification Categories

    private func registerNotificationCategories() {
        // トレーニング開始アクション
        let startAction = UNNotificationAction(
            identifier: "START_TRAINING",
            title: "今すぐ始める",
            options: [.foreground]
        )

        // 後でリマインドアクション
        let remindLaterAction = UNNotificationAction(
            identifier: "REMIND_LATER",
            title: "後で",
            options: []
        )

        // トレーニング招待カテゴリ
        let trainingCategory = UNNotificationCategory(
            identifier: "TRAINING_INVITATION",
            actions: [startAction, remindLaterAction],
            intentIdentifiers: [],
            options: [.customDismissAction]
        )

        // テスト通知カテゴリ（アクションを追加）
        let testCategory = UNNotificationCategory(
            identifier: "TEST_NOTIFICATION",
            actions: [startAction, remindLaterAction],
            intentIdentifiers: [],
            options: [.customDismissAction]
        )

        UNUserNotificationCenter.current().setNotificationCategories([trainingCategory, testCategory])
        print("[AppDelegate] Notification categories registered")
    }

    // MARK: - Background Tasks

    private func registerBackgroundTasks() {
        // バックグラウンドタスク識別子
        let taskIdentifier = "com.virtualtrainer.calendarSync"

        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: taskIdentifier,
            using: nil
        ) { [weak self] task in
            guard let bgTask = task as? BGAppRefreshTask else {
                task.setTaskCompleted(success: false)
                return
            }
            self?.handleBackgroundSync(task: bgTask)
        }

        print("[AppDelegate] Background task registered: \(taskIdentifier)")
    }

    private func handleBackgroundSync(task: BGAppRefreshTask) {
        print("[AppDelegate] Background sync task started")

        // 30秒のタイムアウトを設定
        task.expirationHandler = {
            print("[AppDelegate] Background sync task expired")
            task.setTaskCompleted(success: false)
        }

        Task {
            do {
                guard let coordinator = syncCoordinator else {
                    throw GoogleCalendarError.authenticationFailed(reason: "Sync coordinator not initialized")
                }

                let hasChanges = try await coordinator.syncCalendar()
                print("[AppDelegate] Background sync completed. Changes detected: \(hasChanges)")
                task.setTaskCompleted(success: true)
            } catch {
                print("[AppDelegate] Background sync failed: \(error)")
                task.setTaskCompleted(success: false)
            }

            // 次回のバックグラウンドタスクをスケジュール
            scheduleBackgroundSync()
        }
    }

    func scheduleBackgroundSync() {
        let request = BGAppRefreshTaskRequest(identifier: "com.virtualtrainer.calendarSync")
        // 4時間後に実行
        request.earliestBeginDate = Date(timeIntervalSinceNow: 4 * 3600)

        do {
            try BGTaskScheduler.shared.submit(request)
            print("[AppDelegate] Next background sync scheduled")
        } catch {
            print("[AppDelegate] Failed to schedule background sync: \(error)")
        }
    }

    // MARK: - Sync Coordinator

    private func initializeSyncCoordinator() {
        // カレンダー連携が有効な場合のみ初期化
        guard UserDefaults.standard.bool(forKey: UserDefaultsKeys.isCalendarConnected) else {
            print("[AppDelegate] Calendar not connected, skip sync coordinator initialization")
            return
        }

        let authService = GoogleCalendarAuthService()
        let apiClient = GoogleCalendarAPIClient(authService: authService)
        let analyzer = CalendarEventAnalyzer()
        let notificationService = OshiTrainerNotificationService()

        syncCoordinator = CalendarSyncCoordinator(
            apiClient: apiClient,
            analyzer: analyzer,
            notificationService: notificationService
        )

        print("[AppDelegate] Sync coordinator initialized")
    }
}

// MARK: - UNUserNotificationCenterDelegate

extension AppDelegate: UNUserNotificationCenterDelegate {
    /// 通知がタップされた時の処理
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let notificationId = response.notification.request.identifier
        let actionIdentifier = response.actionIdentifier

        print("[AppDelegate] Notification action: \(actionIdentifier), ID: \(notificationId)")

        // アクションに応じた処理
        switch actionIdentifier {
        case "START_TRAINING":
            print("[AppDelegate] User wants to start training now")
            // トレーニング開始フラグを保存
            UserDefaults.standard.set(true, forKey: "shouldStartTrainingFromNotification")
            UserDefaults.standard.set(notificationId, forKey: "lastTappedNotificationId")

        case "REMIND_LATER":
            print("[AppDelegate] User chose to be reminded later")
            // 後でリマインド処理（1時間後に再通知など）
            scheduleReminder(originalNotificationId: notificationId)

        case UNNotificationDefaultActionIdentifier:
            // 通知本体をタップ（デフォルトアクション）
            print("[AppDelegate] Notification tapped (default action)")
            UserDefaults.standard.set(notificationId, forKey: "lastTappedNotificationId")

        case UNNotificationDismissActionIdentifier:
            // 通知を閉じた
            print("[AppDelegate] Notification dismissed")

        default:
            break
        }

        // 通知タップを記録
        Task {
            do {
                try await analyticsService.recordNotificationTap(notificationId: notificationId)
                print("[AppDelegate] Notification tap recorded")
            } catch {
                print("[AppDelegate] Failed to record notification tap: \(error)")
            }
        }

        completionHandler()
    }

    /// 後でリマインダーをスケジュール
    private func scheduleReminder(originalNotificationId: String) {
        let content = UNMutableNotificationContent()
        content.title = "トレーニングのリマインダー"
        content.body = "まだ時間ありますか？一緒にトレーニングしましょう！"
        content.sound = .default
        content.categoryIdentifier = "TRAINING_INVITATION"

        // 1時間後にリマインド
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 3600, repeats: false)
        let request = UNNotificationRequest(
            identifier: "reminder-\(originalNotificationId)",
            content: content,
            trigger: trigger
        )

        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("[AppDelegate] Failed to schedule reminder: \(error)")
            } else {
                print("[AppDelegate] Reminder scheduled for 1 hour later")
            }
        }
    }

    /// アプリがフォアグラウンドにある時の通知表示処理
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        // フォアグラウンドでも通知を表示
        completionHandler([.banner, .sound, .badge])
    }
}

// MARK: - Scene Phase Monitoring

extension AppDelegate {
    /// アプリがバックグラウンドに入った時の処理
    func applicationDidEnterBackground(_ application: UIApplication) {
        print("[AppDelegate] App entered background")
        scheduleBackgroundSync()

        // 無効通知をマーク（24時間経過）
        Task {
            do {
                try await analyticsService.markInvalidNotifications()
            } catch {
                print("[AppDelegate] Failed to mark invalid notifications: \(error)")
            }
        }
    }

    /// アプリがフォアグラウンドに入った時の処理
    func applicationWillEnterForeground(_ application: UIApplication) {
        print("[AppDelegate] App entering foreground")

        // フォアグラウンド同期を実行
        Task {
            do {
                guard let coordinator = syncCoordinator else { return }
                _ = try await coordinator.syncCalendar()
                print("[AppDelegate] Foreground sync completed")
            } catch {
                print("[AppDelegate] Foreground sync failed: \(error)")
            }
        }
    }
}