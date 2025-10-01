// DEBUG: Notification detail inspector - remove in production

import SwiftUI
import UserNotifications

/// 通知詳細表示
/// 要件対応: 2.6
struct NotificationDetailView: View {
    let notification: UNNotificationRequest

    var body: some View {
        List {
            Section("基本情報") {
                DebugDetailRow(label: "通知ID", value: notification.identifier)

                if let triggerDate = getTriggerDate() {
                    DebugDetailRow(label: "配信予定時刻", value: formatDate(triggerDate))
                }

                DebugDetailRow(label: "カテゴリ", value: notification.content.categoryIdentifier)
            }

            Section("コンテンツ") {
                DebugDetailRow(label: "タイトル", value: notification.content.title)
                DebugDetailRow(label: "本文", value: notification.content.body)
            }

            if let trigger = notification.trigger as? UNCalendarNotificationTrigger {
                Section("トリガー条件") {
                    if let components = trigger.dateComponents.date {
                        DebugDetailRow(label: "日時", value: formatDate(components))
                    }
                    DebugDetailRow(label: "繰り返し", value: trigger.repeats ? "あり" : "なし")
                }
            }

            if !notification.content.userInfo.isEmpty {
                Section("追加情報") {
                    ForEach(Array(notification.content.userInfo.keys), id: \.self) { key in
                        if let value = notification.content.userInfo[key] {
                            DebugDetailRow(label: "\(key)", value: "\(value)")
                        }
                    }
                }
            }
        }
        .navigationTitle("通知詳細")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func getTriggerDate() -> Date? {
        guard let trigger = notification.trigger as? UNCalendarNotificationTrigger else {
            return nil
        }
        return trigger.nextTriggerDate()
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }
}

struct DebugDetailRow: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.body)
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    let content = UNMutableNotificationContent()
    content.title = "推乃 藍"
    content.body = "テスト通知です"
    content.categoryIdentifier = "TEST"

    let components = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: Date())
    let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: false)

    let request = UNNotificationRequest(identifier: "test-123", content: content, trigger: trigger)

    return NavigationStack {
        NotificationDetailView(notification: request)
    }
}
