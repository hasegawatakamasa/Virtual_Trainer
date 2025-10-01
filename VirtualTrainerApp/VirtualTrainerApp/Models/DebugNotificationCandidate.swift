// DEBUG: Debug-only model for notification candidate inspection

import Foundation

/// 通知候補データモデル
/// 要件対応: 6.1, 6.2, 7.1, 7.2, 7.3
struct DebugNotificationCandidate: Identifiable {
    let id: UUID
    let scheduledTime: Date
    let priorityScore: Double
    var selectionStatus: SelectionStatus
    var exclusionReason: ExclusionReason?
    let slotType: String // AvailableTimeSlotTypeの文字列表現

    enum SelectionStatus {
        case selected
        case excluded
    }

    enum ExclusionReason {
        case frequencyLimit(maxCount: Int)
        case timeRangeFilter(allowedRange: String)
        case pastTime
        case weekendOnlyFilter

        var description: String {
            switch self {
            case .frequencyLimit(let maxCount):
                return "1日の上限\(maxCount)件に達したため"
            case .timeRangeFilter(let range):
                return "設定時間帯外（\(range)）"
            case .pastTime:
                return "過去の時刻"
            case .weekendOnlyFilter:
                return "週末のみ設定"
            }
        }
    }
}
