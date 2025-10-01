import XCTest
import UserNotifications
@testable import VirtualTrainerApp

/// 統合テスト：バックグラウンド同期とスケジュール更新フロー
/// 要件対応: タスク20 - 6.1, 6.2, 6.3, 6.4, 6.5
final class BackgroundSyncIntegrationTests: XCTestCase {

    var analyzer: CalendarEventAnalyzer!
    var scheduler: NotificationScheduler!
    var notificationService: OshiTrainerNotificationService!

    @MainActor
    override func setUpWithError() throws {
        try super.setUpWithError()
        analyzer = CalendarEventAnalyzer()
        scheduler = NotificationScheduler()
        notificationService = OshiTrainerNotificationService()
    }

    override func tearDownWithError() throws {
        // テスト後に通知をクリーンアップ（nilにする前に実行）
        if let service = notificationService {
            let expectation = self.expectation(description: "Cancel notifications")
            Task {
                await service.cancelAllNotifications()
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 5.0)
        }

        analyzer = nil
        scheduler = nil
        notificationService = nil

        try super.tearDownWithError()
    }

    // MARK: - Test: Background Sync Updates Schedule

    /// バックグラウンド同期によるスケジュール更新
    func testBackgroundSyncUpdatesSchedule() async throws {
        // 通知権限を確認
        let permissionStatus = await notificationService.checkNotificationPermission()
        guard permissionStatus == .authorized || permissionStatus == .provisional else {
            print("⚠️ Notification permission not granted. Skipping test.")
            throw XCTSkip("Notification permission required for this test")
        }
        // Given: 初回のイベント（3件）
        let now = Date()
        let calendar = Calendar.current

        let event1 = CalendarEvent(
            id: "event1",
            startTime: calendar.date(byAdding: .hour, value: 1, to: now)!,
            endTime: calendar.date(byAdding: .hour, value: 2, to: now)!,
            status: .confirmed
        )

        let event2 = CalendarEvent(
            id: "event2",
            startTime: calendar.date(byAdding: .hour, value: 4, to: now)!,
            endTime: calendar.date(byAdding: .hour, value: 5, to: now)!,
            status: .confirmed
        )

        let event3 = CalendarEvent(
            id: "event3",
            startTime: calendar.date(byAdding: .hour, value: 7, to: now)!,
            endTime: calendar.date(byAdding: .hour, value: 8, to: now)!,
            status: .confirmed
        )

        let oldEvents = [event1, event2, event3]

        // When: 初回スケジュール生成
        let initialSlots = analyzer.analyzeAvailableSlots(events: oldEvents)
        let initialSettings = NotificationSettings()
        let initialCandidates = scheduler.generateNotificationCandidates(
            from: initialSlots,
            settings: initialSettings
        )

        let initialScheduledCount = try await notificationService.scheduleNotifications(initialCandidates)

        // Then: 通知がスケジュールされること
        XCTAssertGreaterThan(initialScheduledCount, 0, "Initial notifications should be scheduled")

        let center = UNUserNotificationCenter.current()
        let initialRequests = await center.pendingNotificationRequests()
        let initialCount = initialRequests.count

        // Given: 更新されたイベント（1件追加、1件削除）
        let event4 = CalendarEvent(
            id: "event4",
            startTime: calendar.date(byAdding: .hour, value: 10, to: now)!,
            endTime: calendar.date(byAdding: .hour, value: 11, to: now)!,
            status: .confirmed
        )

        // event2を削除、event4を追加
        let newEvents = [event1, event3, event4]

        // When: イベント変更を検出
        let hasChanges = detectChanges(oldEvents: oldEvents, newEvents: newEvents)

        // Then: 変更が検出されること
        XCTAssertTrue(hasChanges, "Changes should be detected")

        // When: 通知スケジュールを再生成
        await notificationService.cancelAllNotifications()

        let updatedSlots = analyzer.analyzeAvailableSlots(events: newEvents)
        let updatedCandidates = scheduler.generateNotificationCandidates(
            from: updatedSlots,
            settings: initialSettings
        )

        let updatedScheduledCount = try await notificationService.scheduleNotifications(updatedCandidates)

        // Then: 通知が更新されること
        XCTAssertGreaterThan(updatedScheduledCount, 0, "Updated notifications should be scheduled")

        let updatedRequests = await center.pendingNotificationRequests()
        let updatedCount = updatedRequests.count

        // 通知数が変更されている可能性がある
        print("Initial notification count: \(initialCount)")
        print("Updated notification count: \(updatedCount)")
    }

    // MARK: - Test: Event Change Detection

    /// イベント変更検出のテスト
    func testEventChangeDetection() {
        let now = Date()
        let calendar = Calendar.current

        let event1 = CalendarEvent(
            id: "event1",
            startTime: calendar.date(byAdding: .hour, value: 1, to: now)!,
            endTime: calendar.date(byAdding: .hour, value: 2, to: now)!,
            status: .confirmed
        )

        let event2 = CalendarEvent(
            id: "event2",
            startTime: calendar.date(byAdding: .hour, value: 4, to: now)!,
            endTime: calendar.date(byAdding: .hour, value: 5, to: now)!,
            status: .confirmed
        )

        // Given: 変更なし
        let oldEvents1 = [event1, event2]
        let newEvents1 = [event1, event2]

        // Then: 変更が検出されないこと
        XCTAssertFalse(detectChanges(oldEvents: oldEvents1, newEvents: newEvents1),
                       "No changes should be detected when events are identical")

        // Given: イベント追加
        let event3 = CalendarEvent(
            id: "event3",
            startTime: calendar.date(byAdding: .hour, value: 7, to: now)!,
            endTime: calendar.date(byAdding: .hour, value: 8, to: now)!,
            status: .confirmed
        )

        let oldEvents2 = [event1, event2]
        let newEvents2 = [event1, event2, event3]

        // Then: 変更が検出されること
        XCTAssertTrue(detectChanges(oldEvents: oldEvents2, newEvents: newEvents2),
                      "Changes should be detected when event is added")

        // Given: イベント削除
        let oldEvents3 = [event1, event2, event3]
        let newEvents3 = [event1, event2]

        // Then: 変更が検出されること
        XCTAssertTrue(detectChanges(oldEvents: oldEvents3, newEvents: newEvents3),
                      "Changes should be detected when event is removed")

        // Given: イベント時刻変更
        let event1Modified = CalendarEvent(
            id: "event1",
            startTime: calendar.date(byAdding: .hour, value: 2, to: now)!,  // 時刻変更
            endTime: calendar.date(byAdding: .hour, value: 3, to: now)!,
            status: .confirmed
        )

        let oldEvents4 = [event1, event2]
        let newEvents4 = [event1Modified, event2]

        // Then: 変更が検出されること
        XCTAssertTrue(detectChanges(oldEvents: oldEvents4, newEvents: newEvents4),
                      "Changes should be detected when event time is modified")
    }

    // MARK: - Test: Sync Interval

    /// 同期間隔のテスト
    func testSyncInterval() {
        let lastSyncTime = Date()
        let syncIntervalHours: TimeInterval = 4 * 3600

        // Given: 最後の同期から2時間後
        let twoHoursLater = lastSyncTime.addingTimeInterval(2 * 3600)

        // Then: まだ同期が必要ない
        XCTAssertFalse(shouldSync(lastSyncTime: lastSyncTime, now: twoHoursLater, interval: syncIntervalHours),
                       "Should not sync before interval")

        // Given: 最後の同期から4時間後
        let fourHoursLater = lastSyncTime.addingTimeInterval(4 * 3600)

        // Then: 同期が必要
        XCTAssertTrue(shouldSync(lastSyncTime: lastSyncTime, now: fourHoursLater, interval: syncIntervalHours),
                      "Should sync after interval")

        // Given: 最後の同期から8時間後
        let eightHoursLater = lastSyncTime.addingTimeInterval(8 * 3600)

        // Then: 同期が必要
        XCTAssertTrue(shouldSync(lastSyncTime: lastSyncTime, now: eightHoursLater, interval: syncIntervalHours),
                      "Should sync after long interval")
    }

    // MARK: - Helper Methods

    /// イベント変更検出ロジック（CalendarSyncCoordinatorから抽出）
    private func detectChanges(oldEvents: [CalendarEvent], newEvents: [CalendarEvent]) -> Bool {
        // イベント数が異なる場合は変更あり
        if oldEvents.count != newEvents.count {
            return true
        }

        let oldEventDict = Dictionary(uniqueKeysWithValues: oldEvents.map { ($0.id, $0) })
        let newEventDict = Dictionary(uniqueKeysWithValues: newEvents.map { ($0.id, $0) })

        // IDが異なる場合は変更あり
        if Set(oldEventDict.keys) != Set(newEventDict.keys) {
            return true
        }

        // 各イベントの時刻が変更されているかチェック
        for (id, oldEvent) in oldEventDict {
            guard let newEvent = newEventDict[id] else { continue }

            if oldEvent.startTime != newEvent.startTime || oldEvent.endTime != newEvent.endTime {
                return true
            }
        }

        return false
    }

    /// 同期が必要かどうかを判定
    private func shouldSync(lastSyncTime: Date, now: Date, interval: TimeInterval) -> Bool {
        return now.timeIntervalSince(lastSyncTime) >= interval
    }
}