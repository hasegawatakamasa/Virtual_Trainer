// DEBUG: Gap time data provider for debugging - remove in production

import Foundation
import SwiftUI

/// 隙間時間デバッグデータ提供サービス
/// 要件対応: 5.1, 5.4, 5.6
@MainActor
class GapTimeDebugProvider: ObservableObject {
    @Published var gapTimes: [AvailableTimeSlot] = []
    @Published var excludedSlots: [AvailableTimeSlot] = []
    @Published var allEvents: [CalendarEvent] = []

    private let analyzer: CalendarEventAnalyzer
    private let apiClient: GoogleCalendarAPIClient?
    private let minimumGapDuration: TimeInterval = 30 * 60  // 30分

    init() {
        self.analyzer = CalendarEventAnalyzer()

        // カレンダー連携状態を確認
        let isConnected = UserDefaults.standard.bool(forKey: UserDefaultsKeys.isCalendarConnected)
        if isConnected {
            let authService = GoogleCalendarAuthService()
            self.apiClient = GoogleCalendarAPIClient(authService: authService)
        } else {
            self.apiClient = nil
        }
    }

    /// 隙間時間検出結果を読み込み
    func loadGapTimes() async {
        guard let apiClient = apiClient else {
            gapTimes = []
            excludedSlots = []
            return
        }

        do {
            let calendar = Calendar.current
            let now = Date()
            guard let endDate = calendar.date(byAdding: .day, value: 7, to: now) else {
                return
            }

            // APIからイベントを取得
            let events = try await apiClient.fetchEvents(from: now, to: endDate)
            allEvents = events

            // 隙間時間を解析
            let allSlots = analyzer.analyzeAvailableSlots(events: events)

            // 30分以上と30分未満に分類
            gapTimes = allSlots.filter { $0.duration >= minimumGapDuration }
            excludedSlots = allSlots.filter { $0.duration < minimumGapDuration }
        } catch {
            print("[GapTimeDebugProvider] Failed to load gap times: \(error)")
            gapTimes = []
            excludedSlots = []
        }
    }

    /// 特定の隙間時間の前後イベントを取得
    /// - Parameter slot: 対象の隙間時間
    /// - Returns: (前のイベント, 後のイベント)
    func getSurroundingEvents(for slot: AvailableTimeSlot) -> (before: CalendarEvent?, after: CalendarEvent?) {
        let sortedEvents = allEvents.sorted { $0.startTime < $1.startTime }

        var beforeEvent: CalendarEvent?
        var afterEvent: CalendarEvent?

        for event in sortedEvents {
            // 隙間時間の開始時刻の直前のイベント
            if event.endTime <= slot.startTime {
                beforeEvent = event
            }
            // 隙間時間の終了時刻の直後のイベント
            if event.startTime >= slot.endTime && afterEvent == nil {
                afterEvent = event
                break
            }
        }

        return (beforeEvent, afterEvent)
    }
}
