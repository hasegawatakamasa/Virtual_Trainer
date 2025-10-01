import XCTest
import UserNotifications
@testable import VirtualTrainerApp

/// 統合テスト：カレンダー連携から通知配信までのフロー
/// 要件対応: タスク19 - 全要件のE2E検証
final class CalendarNotificationIntegrationTests: XCTestCase {

    var analyzer: CalendarEventAnalyzer!
    var scheduler: NotificationScheduler!
    var notificationService: OshiTrainerNotificationService!
    var settings: NotificationSettings?

    @MainActor
    override func setUpWithError() throws {
        try super.setUpWithError()
        analyzer = CalendarEventAnalyzer()
        scheduler = NotificationScheduler()
        notificationService = OshiTrainerNotificationService()
        settings = NotificationSettings()
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
        settings = nil

        try super.tearDownWithError()
    }

    // MARK: - Test: Calendar Connection to Notification Scheduling

    /// カレンダー連携から通知配信までの完全フロー
    func testCalendarConnectionToNotificationScheduling() async throws {
        // Given: カレンダーイベントのモックデータ
        let now = Date()
        let calendar = Calendar.current

        let event1Start = calendar.date(byAdding: .hour, value: 1, to: now)!
        let event1End = calendar.date(byAdding: .hour, value: 2, to: now)!

        let event2Start = calendar.date(byAdding: .hour, value: 4, to: now)!
        let event2End = calendar.date(byAdding: .hour, value: 5, to: now)!

        let mockEvents = [
            CalendarEvent(
                id: "event1",
                startTime: event1Start,
                endTime: event1End,
                status: .confirmed
            ),
            CalendarEvent(
                id: "event2",
                startTime: event2Start,
                endTime: event2End,
                status: .confirmed
            )
        ]

        // When: 空き時間を検出
        let availableSlots = analyzer.analyzeAvailableSlots(events: mockEvents)

        // Then: 空き時間が検出されること（イベント間の隙間）
        XCTAssertFalse(availableSlots.isEmpty, "Available slots should be detected")

        // When: 通知候補を生成
        guard let settings = settings else {
            XCTFail("Settings not initialized")
            return
        }
        let candidates = scheduler.generateNotificationCandidates(
            from: availableSlots,
            settings: settings
        )

        // Then: 通知候補が生成されること
        XCTAssertFalse(candidates.isEmpty, "Notification candidates should be generated")

        // 通知候補が未来の時刻であることを確認
        for candidate in candidates {
            XCTAssertGreaterThan(candidate.scheduledTime, now, "Notification should be scheduled in the future")
        }

        // When: 通知権限を確認してスケジュール
        let permissionStatus = await notificationService.checkNotificationPermission()

        // テスト環境では通知権限が拒否されている可能性があるため、
        // 権限が許可されている場合のみ通知スケジュールをテスト
        if permissionStatus == .authorized || permissionStatus == .provisional {
            let scheduledCount = try await notificationService.scheduleNotifications(candidates)

            // Then: 通知がスケジュールされること
            XCTAssertGreaterThan(scheduledCount, 0, "At least one notification should be scheduled")

            // When: スケジュールされた通知を確認
            let center = UNUserNotificationCenter.current()
            let pendingRequests = await center.pendingNotificationRequests()

            // Then: 通知が登録されていること
            XCTAssertGreaterThan(pendingRequests.count, 0, "Pending notifications should exist")

            // 通知内容の検証
            for request in pendingRequests {
                let content = request.content
                XCTAssertFalse(content.title.isEmpty, "Notification title should not be empty")
                XCTAssertFalse(content.body.isEmpty, "Notification body should not be empty")
            }
        } else {
            // 通知権限がない場合はスキップ
            print("⚠️ Notification permission not granted. Skipping notification scheduling test.")
            XCTSkip("Notification permission required for this test")
        }
    }

    // MARK: - Test: Empty Calendar Events

    /// イベントなし（予定なし日）の通知生成
    func testNotificationSchedulingForFreeDay() async throws {
        // Given: イベントなし
        let emptyEvents: [CalendarEvent] = []

        // When: 空き時間を検出
        let availableSlots = analyzer.analyzeAvailableSlots(events: emptyEvents)

        // Then: 予定なし日のデフォルトスロットが生成されること
        XCTAssertFalse(availableSlots.isEmpty, "Default slots should be generated for free days")

        // デフォルト時間帯（10:00, 14:00, 18:00）が含まれることを確認
        let slotHours = availableSlots.map { Calendar.current.component(.hour, from: $0.startTime) }
        XCTAssertTrue(slotHours.contains(10) || slotHours.contains(14) || slotHours.contains(18),
                      "Default time slots (10:00, 14:00, 18:00) should be present")
    }

    // MARK: - Test: Notification Frequency Settings

    /// 通知頻度設定のテスト
    func testNotificationFrequencyFiltering() async throws {
        // Given: 多数の空き時間スロット
        let now = Date()
        let calendar = Calendar.current

        var slots: [AvailableTimeSlot] = []
        for i in 1...10 {
            let startTime = calendar.date(byAdding: .hour, value: i, to: now)!
            let endTime = calendar.date(byAdding: .hour, value: i + 1, to: now)!
            slots.append(AvailableTimeSlot(
                startTime: startTime,
                endTime: endTime,
                duration: 3600,
                slotType: .gapTime
            ))
        }

        guard let baseSettings = settings else {
            XCTFail("Settings not initialized")
            return
        }

        // When: 控えめ頻度（1日1回）で通知候補を生成
        var modestSettings = baseSettings
        modestSettings.frequency = .modest
        let modestCandidates = scheduler.generateNotificationCandidates(
            from: slots,
            settings: modestSettings
        )

        // Then: 1日1回に制限されること
        XCTAssertLessThanOrEqual(modestCandidates.count, 1, "Modest frequency should limit to 1 notification per day")

        // When: 標準頻度（1日2回）で通知候補を生成
        var standardSettings = baseSettings
        standardSettings.frequency = .standard
        let standardCandidates = scheduler.generateNotificationCandidates(
            from: slots,
            settings: standardSettings
        )

        // Then: 1日2回に制限されること
        XCTAssertLessThanOrEqual(standardCandidates.count, 2, "Standard frequency should limit to 2 notifications per day")

        // When: 積極的頻度（1日3回）で通知候補を生成
        var activeSettings = baseSettings
        activeSettings.frequency = .active
        let activeCandidates = scheduler.generateNotificationCandidates(
            from: slots,
            settings: activeSettings
        )

        // Then: 1日3回に制限されること
        XCTAssertLessThanOrEqual(activeCandidates.count, 3, "Active frequency should limit to 3 notifications per day")
    }

    // MARK: - Test: Time Range Filtering

    /// 通知時間帯フィルタのテスト
    func testTimeRangeFiltering() async throws {
        // Given: 様々な時間帯のスロット
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())

        var slots: [AvailableTimeSlot] = []

        // 早朝（6:00）
        let morningSlot = calendar.date(bySettingHour: 6, minute: 0, second: 0, of: today)!
        slots.append(AvailableTimeSlot(
            startTime: morningSlot,
            endTime: calendar.date(byAdding: .hour, value: 1, to: morningSlot)!,
            duration: 3600,
            slotType: .morningSlot
        ))

        // 午後（14:00）
        let afternoonSlot = calendar.date(bySettingHour: 14, minute: 0, second: 0, of: today)!
        slots.append(AvailableTimeSlot(
            startTime: afternoonSlot,
            endTime: calendar.date(byAdding: .hour, value: 1, to: afternoonSlot)!,
            duration: 3600,
            slotType: .gapTime
        ))

        // 夜（22:00）
        let nightSlot = calendar.date(bySettingHour: 22, minute: 0, second: 0, of: today)!
        slots.append(AvailableTimeSlot(
            startTime: nightSlot,
            endTime: calendar.date(byAdding: .hour, value: 1, to: nightSlot)!,
            duration: 3600,
            slotType: .eveningSlot
        ))

        guard let baseSettings = settings else {
            XCTFail("Settings not initialized")
            return
        }

        // When: 9:00-18:00の時間帯で通知候補を生成
        var timeRangeSettings = baseSettings
        timeRangeSettings.timeRangeStart = calendar.date(bySettingHour: 9, minute: 0, second: 0, of: today)!
        timeRangeSettings.timeRangeEnd = calendar.date(bySettingHour: 18, minute: 0, second: 0, of: today)!

        let candidates = scheduler.generateNotificationCandidates(
            from: slots,
            settings: timeRangeSettings
        )

        // Then: 時間帯外の通知は除外されること
        for candidate in candidates {
            let hour = calendar.component(.hour, from: candidate.scheduledTime)
            XCTAssertGreaterThanOrEqual(hour, 9, "Notification should be after 9:00")
            XCTAssertLessThan(hour, 18, "Notification should be before 18:00")
        }
    }
}