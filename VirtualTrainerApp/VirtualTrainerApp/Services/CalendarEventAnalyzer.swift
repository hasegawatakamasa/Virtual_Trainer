import Foundation

/// カレンダーイベント解析サービス
/// 要件対応: 2.3, 2.4, 2.5, 2.6
class CalendarEventAnalyzer {
    private let minimumGapDuration: TimeInterval = 30 * 60  // 30分
    private let morningStartHour = 6
    private let eveningEndHour = 21

    // MARK: - Public Methods

    /// カレンダーイベントから空き時間を検出
    /// - Parameter events: 解析対象のイベント配列
    /// - Returns: 空き時間スロット配列
    func analyzeAvailableSlots(events: [CalendarEvent]) -> [AvailableTimeSlot] {
        var allSlots: [AvailableTimeSlot] = []
        let calendar = Calendar.current
        let now = Date()

        // イベントを日別にグループ化
        let groupedEvents = Dictionary(grouping: events) { event -> Date in
            calendar.startOfDay(for: event.startTime)
        }

        // 今日から7日分の日付を生成
        for dayOffset in 0..<7 {
            guard let date = calendar.date(byAdding: .day, value: dayOffset, to: calendar.startOfDay(for: now)) else {
                continue
            }

            let dayEvents = groupedEvents[date] ?? []

            if dayEvents.isEmpty {
                // 予定なし日のスロットを生成
                allSlots.append(contentsOf: detectFreeDay(date: date, events: []))
            } else {
                // 隙間時間を検出
                allSlots.append(contentsOf: detectGapTimes(in: dayEvents))

                // 朝の空き時間を検出
                if let morningSlot = detectMorningSlot(in: dayEvents, currentTime: now) {
                    allSlots.append(morningSlot)
                }

                // 夜の空き時間を検出
                if let eveningSlot = detectEveningSlot(in: dayEvents) {
                    allSlots.append(eveningSlot)
                }
            }
        }

        return allSlots
    }

    /// 隙間時間を検出（30分以上の空き）
    /// - Parameter events: 日次イベント配列
    /// - Returns: 隙間時間スロット配列
    func detectGapTimes(in events: [CalendarEvent]) -> [AvailableTimeSlot] {
        guard events.count > 1 else { return [] }

        // イベントを開始時刻でソート
        let sortedEvents = events.sorted { $0.startTime < $1.startTime }
        var gaps: [AvailableTimeSlot] = []

        for i in 0..<(sortedEvents.count - 1) {
            let currentEvent = sortedEvents[i]
            let nextEvent = sortedEvents[i + 1]

            let gapStart = currentEvent.endTime
            let gapEnd = nextEvent.startTime
            let duration = gapEnd.timeIntervalSince(gapStart)

            // 30分以上の隙間のみを対象
            if duration >= minimumGapDuration {
                let slot = AvailableTimeSlot(
                    startTime: gapStart,
                    endTime: gapEnd,
                    duration: duration,
                    slotType: .gapTime
                )
                gaps.append(slot)
            }
        }

        return gaps
    }

    /// 朝の空き時間を検出
    /// - Parameter events: 日次イベント配列
    /// - Returns: 朝の空き時間スロット（あれば）
    func detectMorningSlot(in events: [CalendarEvent], currentTime: Date) -> AvailableTimeSlot? {
        guard !events.isEmpty else { return nil }

        let calendar = Calendar.current
        let sortedEvents = events.sorted { $0.startTime < $1.startTime }
        let firstEvent = sortedEvents[0]

        // 最初のイベントの開始時刻
        let firstEventHour = calendar.component(.hour, from: firstEvent.startTime)

        // 朝6時以降で、最初のイベントが9時以降の場合
        guard firstEventHour >= 9 else { return nil }

        // 現在時刻が6時以降かチェック
        let currentHour = calendar.component(.hour, from: currentTime)
        guard currentHour >= morningStartHour else { return nil }

        // 空き時間の開始は現在時刻 or 6:00のいずれか遅い方
        let morningStart = calendar.date(bySettingHour: morningStartHour, minute: 0, second: 0, of: firstEvent.startTime)!
        let slotStart = max(morningStart, currentTime)

        guard slotStart < firstEvent.startTime else { return nil }

        let duration = firstEvent.startTime.timeIntervalSince(slotStart)
        guard duration >= minimumGapDuration else { return nil }

        return AvailableTimeSlot(
            startTime: slotStart,
            endTime: firstEvent.startTime,
            duration: duration,
            slotType: .morningSlot
        )
    }

    /// 夜の空き時間を検出
    /// - Parameter events: 日次イベント配列
    /// - Returns: 夜の空き時間スロット（あれば）
    func detectEveningSlot(in events: [CalendarEvent]) -> AvailableTimeSlot? {
        guard !events.isEmpty else { return nil }

        let calendar = Calendar.current
        let sortedEvents = events.sorted { $0.startTime < $1.startTime }
        let lastEvent = sortedEvents[sortedEvents.count - 1]

        // 最後のイベントの終了時刻
        let lastEventHour = calendar.component(.hour, from: lastEvent.endTime)

        // 21時以前に終わっている場合
        guard lastEventHour < eveningEndHour else { return nil }

        // 21:00までを空き時間として設定
        guard let eveningEnd = calendar.date(bySettingHour: eveningEndHour, minute: 0, second: 0, of: lastEvent.endTime) else {
            return nil
        }

        let duration = eveningEnd.timeIntervalSince(lastEvent.endTime)
        guard duration >= minimumGapDuration else { return nil }

        return AvailableTimeSlot(
            startTime: lastEvent.endTime,
            endTime: eveningEnd,
            duration: duration,
            slotType: .eveningSlot
        )
    }

    /// 予定なし日を検出
    /// - Parameter events: 日次イベント配列
    /// - Returns: 予定なし日のデフォルトスロット配列
    func detectFreeDay(date: Date, events: [CalendarEvent]) -> [AvailableTimeSlot] {
        guard events.isEmpty else { return [] }

        let calendar = Calendar.current
        var slots: [AvailableTimeSlot] = []

        // デフォルトの時刻: 10:00, 14:00, 18:00
        let defaultHours = [10, 14, 18]

        for hour in defaultHours {
            guard let startTime = calendar.date(bySettingHour: hour, minute: 0, second: 0, of: date),
                  let endTime = calendar.date(byAdding: .hour, value: 2, to: startTime) else {
                continue
            }

            // 過去の時刻はスキップ
            guard startTime > Date() else { continue }

            let slot = AvailableTimeSlot(
                startTime: startTime,
                endTime: endTime,
                duration: 2 * 3600,  // 2時間
                slotType: .freeDay
            )
            slots.append(slot)
        }

        return slots
    }
}