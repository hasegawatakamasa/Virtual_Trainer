import Foundation
import UserNotifications

// MARK: - CalendarEvent (時刻のみ、プライバシー保護)

struct CalendarEvent: Identifiable, Codable {
    let id: String
    let startTime: Date
    let endTime: Date
    let status: EventStatus

    enum EventStatus: String, Codable {
        case confirmed
        case tentative
    }
}

// MARK: - AvailableTimeSlot

struct AvailableTimeSlot: Identifiable {
    let id = UUID()
    let startTime: Date
    let endTime: Date
    let duration: TimeInterval
    let slotType: TimeSlotType

    enum TimeSlotType: String, Codable {
        case gapTime        // 隙間時間
        case morningSlot    // 朝の空き
        case eveningSlot    // 夜の空き
        case freeDay        // 予定なし日
    }
}

// MARK: - NotificationCandidate

struct NotificationCandidate: Identifiable {
    let id = UUID()
    let scheduledTime: Date
    let slot: AvailableTimeSlot
    let priority: Int
    let trainerId: String
    let message: String
    let imageAttachment: UNNotificationAttachment?
}

// MARK: - NotificationSettings

struct NotificationSettings: Codable {
    var enabled: Bool = false
    var frequency: NotificationFrequency = .standard
    var timeRangeStart: Date = Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: Date())!
    var timeRangeEnd: Date = Calendar.current.date(bySettingHour: 21, minute: 0, second: 0, of: Date())!
    var weekendOnly: Bool = false

    enum NotificationFrequency: String, Codable, CaseIterable {
        case modest = "modest"        // 控えめ: 1日1回
        case standard = "standard"    // 標準: 1日2回
        case active = "active"        // 積極的: 1日3回

        var maxDailyNotifications: Int {
            switch self {
            case .modest: return 1
            case .standard: return 2
            case .active: return 3
            }
        }

        var displayName: String {
            switch self {
            case .modest: return "控えめ（1日1回）"
            case .standard: return "標準（1日2回）"
            case .active: return "積極的（1日3回）"
            }
        }
    }
}

// MARK: - NotificationAnalytics

struct NotificationAnalytics {
    let period: DateInterval
    let totalDelivered: Int
    let totalTapped: Int
    let totalLinkedToSession: Int
    let tapRate: Double
    let conversionRate: Double
    let optimalTimeSlots: [TimeSlot: Double]

    struct TimeSlot: Hashable {
        let hour: Int  // 0-23
    }
}

// MARK: - NotificationEffectiveness

struct NotificationEffectiveness {
    let weekStart: Date
    let weekEnd: Date
    let deliveredCount: Int
    let tappedCount: Int
    let sessionCount: Int
    let effectiveness: Double  // セッション実施率
}

// MARK: - GoogleUser

struct GoogleUser {
    let userId: String
    let email: String
    let fullName: String?
    let profileImageURL: URL?
}

// MARK: - NotificationTimeRange

struct NotificationTimeRange: Codable {
    var start: Date
    var end: Date

    func contains(_ date: Date) -> Bool {
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: date)
        let startHour = calendar.component(.hour, from: start)
        let endHour = calendar.component(.hour, from: end)
        return hour >= startHour && hour < endHour
    }
}

// MARK: - CalendarInfo

struct CalendarInfo: Codable {
    let id: String
    let summary: String
    let timeZone: String
}

// MARK: - FrequencyAdjustment

struct FrequencyAdjustment {
    let suggestedFrequency: NotificationSettings.NotificationFrequency
    let reason: String
}