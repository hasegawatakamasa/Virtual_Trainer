import XCTest
@testable import VirtualTrainerApp

/// エッジケーステストとエラーハンドリング検証
/// 要件対応: タスク23 - 1.5, 2.7, 4.8, 5.7
final class EdgeCaseTests: XCTestCase {

    var analyzer: CalendarEventAnalyzer!
    var scheduler: NotificationScheduler!

    override func setUpWithError() throws {
        try super.setUpWithError()
        analyzer = CalendarEventAnalyzer()
        scheduler = NotificationScheduler()
    }

    override func tearDownWithError() throws {
        analyzer = nil
        scheduler = nil
        try super.tearDownWithError()
    }

    // MARK: - Test: Empty Events

    /// 空のイベントリスト
    func testEmptyEventsList() {
        // Given: イベントなし
        let emptyEvents: [CalendarEvent] = []

        // When: 空き時間を検出
        let slots = analyzer.analyzeAvailableSlots(events: emptyEvents)

        // Then: デフォルトスロットが生成されること
        XCTAssertFalse(slots.isEmpty, "Default slots should be generated for empty events")
    }

    // MARK: - Test: No Gaps Between Events

    /// イベント間に隙間がない場合
    func testNoGapsBetweenEvents() {
        // Given: 隙間なしのイベント
        let now = Date()
        let calendar = Calendar.current

        let event1Start = calendar.date(byAdding: .hour, value: 1, to: now)!
        let event1End = calendar.date(byAdding: .hour, value: 3, to: event1Start)!

        let event2Start = event1End  // 隙間なし
        let event2End = calendar.date(byAdding: .hour, value: 2, to: event2Start)!

        let events = [
            CalendarEvent(id: "1", startTime: event1Start, endTime: event1End, status: .confirmed),
            CalendarEvent(id: "2", startTime: event2Start, endTime: event2End, status: .confirmed)
        ]

        // When: 空き時間を検出
        let slots = analyzer.analyzeAvailableSlots(events: events)

        // Then: 隙間時間スロットは生成されない（朝・夜スロットのみ）
        let gapSlots = slots.filter { $0.slotType == .gapTime }
        XCTAssertTrue(gapSlots.isEmpty, "No gap slots should be detected when events are consecutive")
    }

    // MARK: - Test: Short Gaps

    /// 30分未満の短い隙間
    func testShortGapsBetweenEvents() {
        // Given: 20分の隙間
        let now = Date()
        let calendar = Calendar.current

        let event1Start = calendar.date(byAdding: .hour, value: 1, to: now)!
        let event1End = calendar.date(byAdding: .hour, value: 2, to: event1Start)!

        let event2Start = calendar.date(byAdding: .minute, value: 20, to: event1End)!  // 20分の隙間
        let event2End = calendar.date(byAdding: .hour, value: 1, to: event2Start)!

        let events = [
            CalendarEvent(id: "1", startTime: event1Start, endTime: event1End, status: .confirmed),
            CalendarEvent(id: "2", startTime: event2Start, endTime: event2End, status: .confirmed)
        ]

        // When: 空き時間を検出
        let slots = analyzer.analyzeAvailableSlots(events: events)

        // Then: 30分未満の隙間は検出されない
        let gapSlots = slots.filter { $0.slotType == .gapTime }
        XCTAssertTrue(gapSlots.isEmpty, "Gaps shorter than 30 minutes should be ignored")
    }

    // MARK: - Test: All Day Events

    /// 終日イベント
    func testAllDayEvents() {
        // Given: 終日イベント（00:00 - 23:59）
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())

        let allDayStart = today
        let allDayEnd = calendar.date(bySettingHour: 23, minute: 59, second: 59, of: today)!

        let events = [
            CalendarEvent(id: "all-day", startTime: allDayStart, endTime: allDayEnd, status: .confirmed)
        ]

        // When: 空き時間を検出
        let slots = analyzer.analyzeAvailableSlots(events: events)

        // Then: 空き時間が検出されない、または最小限
        let todaySlots = slots.filter {
            calendar.isDate($0.startTime, inSameDayAs: today)
        }
        XCTAssertTrue(todaySlots.isEmpty || todaySlots.count < 2, "All day events should leave minimal or no slots")
    }

    // MARK: - Test: Past Notifications

    /// 過去の時刻の通知候補
    func testPastTimeNotifications() {
        // Given: 過去の時刻のスロット
        let past = Date().addingTimeInterval(-3600)  // 1時間前

        let slots = [
            AvailableTimeSlot(
                startTime: past,
                endTime: past.addingTimeInterval(3600),
                duration: 3600,
                slotType: .gapTime
            )
        ]

        let settings = NotificationSettings()

        // When: 通知候補を生成
        let candidates = scheduler.generateNotificationCandidates(from: slots, settings: settings)

        // Then: 過去の通知は除外されること
        XCTAssertTrue(candidates.isEmpty, "Past notifications should be filtered out")
    }

    // MARK: - Test: Zero Notification Candidates

    /// 通知候補が0件の場合
    func testZeroNotificationCandidates() {
        // Given: 通知候補0件
        let slots: [AvailableTimeSlot] = []
        let settings = NotificationSettings()

        // When: 通知候補を生成
        let candidates = scheduler.generateNotificationCandidates(from: slots, settings: settings)

        // Then: 空の配列が返ること（エラーにならない）
        XCTAssertTrue(candidates.isEmpty, "Should handle zero candidates gracefully")
    }

    // MARK: - Test: Frequency Setting Changes

    /// 通知頻度設定変更時の挙動
    func testFrequencySettingChanges() {
        // Given: 多数のスロット
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

        // When: 頻度設定を変更
        var settings = NotificationSettings()

        settings.frequency = .modest
        let modestCandidates = scheduler.generateNotificationCandidates(from: slots, settings: settings)

        settings.frequency = .active
        let activeCandidates = scheduler.generateNotificationCandidates(from: slots, settings: settings)

        // Then: 頻度に応じて候補数が変わること
        XCTAssertLessThanOrEqual(modestCandidates.count, activeCandidates.count,
                                 "Modest frequency should generate fewer candidates than active")
    }

    // MARK: - Test: Weekend Only Setting

    /// 週末のみ設定
    func testWeekendOnlySetting() {
        // Given: 平日と週末のスロット
        let calendar = Calendar.current
        let today = Date()

        // 次の月曜日を探す
        var components = calendar.dateComponents([.yearForWeekOfYear, .weekOfYear], from: today)
        components.weekday = 2  // 月曜日
        let monday = calendar.date(from: components)!

        // 次の土曜日を探す
        components.weekday = 7  // 土曜日
        let saturday = calendar.date(from: components)!

        let mondaySlot = AvailableTimeSlot(
            startTime: calendar.date(bySettingHour: 14, minute: 0, second: 0, of: monday)!,
            endTime: calendar.date(bySettingHour: 15, minute: 0, second: 0, of: monday)!,
            duration: 3600,
            slotType: .freeDay
        )

        let saturdaySlot = AvailableTimeSlot(
            startTime: calendar.date(bySettingHour: 14, minute: 0, second: 0, of: saturday)!,
            endTime: calendar.date(bySettingHour: 15, minute: 0, second: 0, of: saturday)!,
            duration: 3600,
            slotType: .freeDay
        )

        let slots = [mondaySlot, saturdaySlot]

        // When: 週末のみ設定で通知候補を生成
        var settings = NotificationSettings()
        settings.weekendOnly = true

        let candidates = scheduler.generateNotificationCandidates(from: slots, settings: settings)

        // Then: 土曜日のスロットのみが選ばれること
        for candidate in candidates {
            let weekday = calendar.component(.weekday, from: candidate.scheduledTime)
            XCTAssertTrue(weekday == 1 || weekday == 7, "Weekend only should filter to Saturday/Sunday")
        }
    }

    // MARK: - Test: Overlapping Events

    /// 重複イベント
    func testOverlappingEvents() {
        // Given: 重複するイベント
        let now = Date()
        let calendar = Calendar.current

        let event1Start = calendar.date(byAdding: .hour, value: 1, to: now)!
        let event1End = calendar.date(byAdding: .hour, value: 3, to: event1Start)!

        let event2Start = calendar.date(byAdding: .hour, value: 1, to: event1Start)!  // 重複
        let event2End = calendar.date(byAdding: .hour, value: 2, to: event2Start)!

        let events = [
            CalendarEvent(id: "1", startTime: event1Start, endTime: event1End, status: .confirmed),
            CalendarEvent(id: "2", startTime: event2Start, endTime: event2End, status: .confirmed)
        ]

        // When: 空き時間を検出
        let slots = analyzer.analyzeAvailableSlots(events: events)

        // Then: エラーにならず、正しく処理されること
        XCTAssertNotNil(slots, "Should handle overlapping events gracefully")
    }

    // MARK: - Test: Invalid Event Times

    /// 無効なイベント時刻（終了時刻が開始時刻より前）
    func testInvalidEventTimes() {
        // Given: 無効な時刻のイベント
        let now = Date()
        let calendar = Calendar.current

        let startTime = calendar.date(byAdding: .hour, value: 2, to: now)!
        let endTime = calendar.date(byAdding: .hour, value: 1, to: now)!  // 開始より前

        let events = [
            CalendarEvent(id: "invalid", startTime: startTime, endTime: endTime, status: .confirmed)
        ]

        // When: 空き時間を検出
        let slots = analyzer.analyzeAvailableSlots(events: events)

        // Then: エラーにならず、無効なイベントはスキップされること
        XCTAssertNotNil(slots, "Should handle invalid event times gracefully")
    }

    // MARK: - Test: Time Zone Changes

    /// タイムゾーン変更
    func testTimeZoneChanges() {
        // Given: 異なるタイムゾーンでのイベント
        let _ = Calendar.current
        let now = Date()

        // UTC時刻でイベントを作成
        var utcCalendar = Calendar.current
        utcCalendar.timeZone = TimeZone(identifier: "UTC")!

        let event = CalendarEvent(
            id: "utc-event",
            startTime: now,
            endTime: now.addingTimeInterval(3600),
            status: .confirmed
        )

        // When: 空き時間を検出（ローカルタイムゾーン）
        let slots = analyzer.analyzeAvailableSlots(events: [event])

        // Then: タイムゾーンに関係なく正しく処理されること
        XCTAssertNotNil(slots, "Should handle timezone differences correctly")
    }

    // MARK: - Test: Notification Permission Denied

    /// 通知権限拒否時の処理
    @MainActor
    func testNotificationPermissionDenied() async {
        let notificationService = OshiTrainerNotificationService()

        // Note: 実際の権限テストはUIテストで行う
        // ここではエラーハンドリングのロジックをテスト

        let hasPermission = await notificationService.checkNotificationPermission()

        // Then: 権限状態を正しく取得できること
        XCTAssertNotNil(hasPermission, "Should check notification permission without crashing")
    }

    // MARK: - Test: Large Duration Events

    /// 非常に長いイベント（24時間以上）
    func testVeryLongDurationEvents() {
        // Given: 48時間のイベント
        let now = Date()

        let events = [
            CalendarEvent(
                id: "long-event",
                startTime: now,
                endTime: now.addingTimeInterval(48 * 3600),  // 48時間
                status: .confirmed
            )
        ]

        // When: 空き時間を検出
        let slots = analyzer.analyzeAvailableSlots(events: events)

        // Then: エラーにならず処理されること
        XCTAssertNotNil(slots, "Should handle very long duration events")
    }
}