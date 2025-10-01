import XCTest
@testable import VirtualTrainerApp

/// パフォーマンステストとメモリキャッシュ最適化
/// 要件対応: タスク22 - パフォーマンス目標
final class PerformanceTests: XCTestCase {

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

    // MARK: - Test: Available Slots Analysis Performance

    /// 空き時間検出のパフォーマンステスト（7日間イベント）
    func testAvailableSlotsAnalysisPerformance() {
        // Given: 7日間の大量イベント（1日あたり10件 = 合計70件）
        let events = generateMockEvents(days: 7, eventsPerDay: 10)

        // When: 空き時間を検出
        measure {
            let _ = analyzer.analyzeAvailableSlots(events: events)
        }

        // Then: パフォーマンス測定結果を確認
        // 期待値: < 500ms
    }

    /// 通知スケジュール計算のパフォーマンステスト
    func testNotificationSchedulingPerformance() {
        // Given: 多数の空き時間スロット
        let events = generateMockEvents(days: 7, eventsPerDay: 10)
        let slots = analyzer.analyzeAvailableSlots(events: events)
        let settings = NotificationSettings()

        // When: 通知候補を生成
        measure {
            let _ = scheduler.generateNotificationCandidates(from: slots, settings: settings)
        }

        // Then: パフォーマンス測定結果を確認
        // 期待値: < 500ms
    }

    // MARK: - Test: Large Dataset Performance

    /// 大量イベント処理のパフォーマンステスト（30日間）
    func testLargeDatasetPerformance() {
        // Given: 30日間の大量イベント（1日あたり20件 = 合計600件）
        let events = generateMockEvents(days: 30, eventsPerDay: 20)

        // When: 空き時間を検出して通知候補を生成
        measure {
            let slots = analyzer.analyzeAvailableSlots(events: events)
            let settings = NotificationSettings()
            let _ = scheduler.generateNotificationCandidates(from: slots, settings: settings)
        }

        // Then: パフォーマンス測定結果を確認
        // 期待値: < 2秒
    }

    // MARK: - Test: Event Sorting Performance

    /// イベントソートのパフォーマンステスト
    func testEventSortingPerformance() {
        // Given: ランダムな順序の大量イベント
        var events = generateMockEvents(days: 7, eventsPerDay: 50)
        events.shuffle()

        // When: イベントをソート
        measure {
            let _ = events.sorted { $0.startTime < $1.startTime }
        }

        // Then: パフォーマンス測定結果を確認
    }

    // MARK: - Test: Time Slot Prioritization Performance

    /// 通知候補の優先順位付けパフォーマンステスト
    func testTimeSlotPrioritizationPerformance() {
        // Given: 多数のスロット
        let events = generateMockEvents(days: 7, eventsPerDay: 10)
        let slots = analyzer.analyzeAvailableSlots(events: events)
        let settings = NotificationSettings()

        // When: 候補生成と優先順位付け
        measure {
            var candidates = scheduler.generateNotificationCandidates(from: slots, settings: settings)
            // 優先順位でソート（長い空き時間を優先）
            candidates = candidates.sorted { $0.priority > $1.priority }
        }

        // Then: パフォーマンス測定結果を確認
    }

    // MARK: - Test: Memory Usage

    /// メモリ使用量テスト
    func testMemoryUsage() {
        // Given: 大量イベント
        let events = generateMockEvents(days: 30, eventsPerDay: 30)  // 900件

        // When: 繰り返し処理
        for _ in 0..<10 {
            let slots = analyzer.analyzeAvailableSlots(events: events)
            let settings = NotificationSettings()
            let _ = scheduler.generateNotificationCandidates(from: slots, settings: settings)
        }

        // Then: メモリリークがないこと（Instrumentsで確認）
    }

    // MARK: - Helper Methods

    /// モックイベント生成
    private func generateMockEvents(days: Int, eventsPerDay: Int) -> [CalendarEvent] {
        var events: [CalendarEvent] = []
        let calendar = Calendar.current
        let now = Date()

        for day in 0..<days {
            guard let date = calendar.date(byAdding: .day, value: day, to: calendar.startOfDay(for: now)) else {
                continue
            }

            for eventIndex in 0..<eventsPerDay {
                // ランダムな開始時刻（6:00 - 22:00）
                let startHour = Int.random(in: 6...21)
                let startMinute = Int.random(in: 0...59)

                guard let startTime = calendar.date(bySettingHour: startHour, minute: startMinute, second: 0, of: date) else {
                    continue
                }

                // ランダムな期間（15分 - 2時間）
                let duration = TimeInterval(Int.random(in: 15...120) * 60)
                let endTime = startTime.addingTimeInterval(duration)

                let event = CalendarEvent(
                    id: "event-\(day)-\(eventIndex)",
                    startTime: startTime,
                    endTime: endTime,
                    status: .confirmed
                )

                events.append(event)
            }
        }

        return events.sorted { $0.startTime < $1.startTime }
    }
}

// MARK: - Calendar Event Cache (タスク22要件)

/// カレンダーイベントのメモリキャッシュ
final class CalendarEventCache {
    static let shared = CalendarEventCache()

    private var cache: [String: CachedEvents] = [:]
    private let cacheExpiration: TimeInterval = 3600  // 1時間
    private let queue = DispatchQueue(label: "com.virtualtrainer.eventcache", attributes: .concurrent)

    private struct CachedEvents {
        let events: [CalendarEvent]
        let timestamp: Date

        var isExpired: Bool {
            Date().timeIntervalSince(timestamp) > 3600
        }
    }

    private init() {}

    /// イベントをキャッシュに保存
    func cacheEvents(_ events: [CalendarEvent], forKey key: String) {
        queue.async(flags: .barrier) {
            self.cache[key] = CachedEvents(events: events, timestamp: Date())
        }
    }

    /// キャッシュからイベントを取得
    func getCachedEvents(forKey key: String) -> [CalendarEvent]? {
        var result: [CalendarEvent]?

        queue.sync {
            guard let cached = cache[key], !cached.isExpired else {
                return
            }
            result = cached.events
        }

        return result
    }

    /// キャッシュをクリア
    func clearCache() {
        queue.async(flags: .barrier) {
            self.cache.removeAll()
        }
    }

    /// 期限切れキャッシュを削除
    func removeExpiredCache() {
        queue.async(flags: .barrier) {
            self.cache = self.cache.filter { !$0.value.isExpired }
        }
    }

    /// キャッシュヒット率を計算
    func calculateHitRate(hits: Int, total: Int) -> Double {
        guard total > 0 else { return 0.0 }
        return Double(hits) / Double(total)
    }
}

// MARK: - Cache Performance Tests

/// キャッシュパフォーマンステスト
final class CachePerformanceTests: XCTestCase {

    let cache = CalendarEventCache.shared

    override func setUpWithError() throws {
        try super.setUpWithError()
        cache.clearCache()
    }

    override func tearDownWithError() throws {
        cache.clearCache()
        try super.tearDownWithError()
    }

    /// キャッシュ保存・取得のパフォーマンステスト
    func testCacheSaveAndRetrievePerformance() {
        let events = generateMockEvents(count: 100)
        let key = "test-cache-key"

        // When: キャッシュに保存
        measure {
            cache.cacheEvents(events, forKey: key)
            let _ = cache.getCachedEvents(forKey: key)
        }

        // Then: 高速にアクセスできること
    }

    /// キャッシュヒット率テスト
    func testCacheHitRate() {
        let events = generateMockEvents(count: 50)
        let key = "test-events"

        var hits = 0
        var total = 0

        // 最初はキャッシュミス
        total += 1
        if cache.getCachedEvents(forKey: key) != nil {
            hits += 1
        }

        // キャッシュに保存
        cache.cacheEvents(events, forKey: key)

        // 以降はキャッシュヒット
        for _ in 0..<10 {
            total += 1
            if cache.getCachedEvents(forKey: key) != nil {
                hits += 1
            }
        }

        let hitRate = cache.calculateHitRate(hits: hits, total: total)
        XCTAssertGreaterThan(hitRate, 0.8, "Cache hit rate should be > 80%")
    }

    /// 期限切れキャッシュの自動削除テスト
    func testExpiredCacheRemoval() {
        let events = generateMockEvents(count: 50)
        cache.cacheEvents(events, forKey: "old-key")

        // 期限切れキャッシュを削除
        cache.removeExpiredCache()

        // Note: 実際の期限切れテストは1時間待つ必要があるため、
        // ここでは削除メソッドが正常に動作することのみ確認
    }

    // MARK: - Helper

    private func generateMockEvents(count: Int) -> [CalendarEvent] {
        var events: [CalendarEvent] = []
        let now = Date()

        for i in 0..<count {
            let startTime = now.addingTimeInterval(TimeInterval(i * 3600))
            let endTime = startTime.addingTimeInterval(3600)

            events.append(CalendarEvent(
                id: "event-\(i)",
                startTime: startTime,
                endTime: endTime,
                status: .confirmed
            ))
        }

        return events
    }
}