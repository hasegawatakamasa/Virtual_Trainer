// DEBUG: Scheduled notifications inspector - remove in production

import SwiftUI
import UserNotifications

/// 予約通知一覧表示
/// 要件対応: 2.1, 2.2, 2.3, 2.4, 2.7, 3.1, 3.2, 3.5, 3.7
struct ScheduledNotificationsView: View {
    @State private var notifications: [UNNotificationRequest] = []
    @State private var isLoading = false
    @State private var showCancelConfirm = false
    @State private var showCancelAllConfirm = false
    @State private var selectedNotification: UNNotificationRequest?
    private let cancellationManager = NotificationCancellationManager()

    var body: some View {
        List {
            if isLoading {
                ProgressView("読み込み中...")
            } else if notifications.isEmpty {
                Text("現在スケジュールされている通知はありません")
                    .foregroundColor(.secondary)
                    .padding()
            } else {
                ForEach(notifications, id: \.identifier) { notification in
                    NavigationLink(destination: NotificationDetailView(notification: notification)) {
                        NotificationRow(notification: notification)
                    }
                    .swipeActions {
                        Button(role: .destructive) {
                            selectedNotification = notification
                            showCancelConfirm = true
                        } label: {
                            Label("キャンセル", systemImage: "trash")
                        }
                    }
                }
            }
        }
        .navigationTitle("予約通知一覧")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                if !notifications.isEmpty {
                    Button(role: .destructive) {
                        showCancelAllConfirm = true
                    } label: {
                        Text("全てキャンセル")
                    }
                }
            }
        }
        .onAppear {
            loadNotifications()
        }
        .refreshable {
            loadNotifications()
        }
        .confirmationDialog("通知をキャンセルしますか？", isPresented: $showCancelConfirm, presenting: selectedNotification) { notification in
            Button("キャンセル", role: .destructive) {
                cancelNotification(notification)
            }
        }
        .confirmationDialog("全ての通知をキャンセルしますか？", isPresented: $showCancelAllConfirm) {
            Button("全てキャンセル", role: .destructive) {
                cancelAllNotifications()
            }
        }
    }

    private func loadNotifications() {
        isLoading = true
        Task {
            let center = UNUserNotificationCenter.current()
            let requests = await center.pendingNotificationRequests()

            // 配信予定時刻でソート
            let sorted = requests.sorted { req1, req2 in
                guard let trigger1 = req1.trigger as? UNCalendarNotificationTrigger,
                      let trigger2 = req2.trigger as? UNCalendarNotificationTrigger,
                      let date1 = trigger1.nextTriggerDate(),
                      let date2 = trigger2.nextTriggerDate() else {
                    return false
                }
                return date1 < date2
            }

            notifications = sorted
            isLoading = false
        }
    }

    private func cancelNotification(_ notification: UNNotificationRequest) {
        cancellationManager.cancelNotification(identifier: notification.identifier)
        notifications.removeAll { $0.identifier == notification.identifier }
    }

    private func cancelAllNotifications() {
        cancellationManager.cancelAllNotifications()
        notifications.removeAll()
    }
}

struct NotificationRow: View {
    let notification: UNNotificationRequest

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let triggerDate = getTriggerDate() {
                Text(triggerDate, style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(triggerDate, style: .time)
                    .font(.headline)
            }

            Text(notification.content.title)
                .font(.subheadline)
                .fontWeight(.medium)

            Text(notification.content.body)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
        }
        .padding(.vertical, 4)
    }

    private func getTriggerDate() -> Date? {
        guard let trigger = notification.trigger as? UNCalendarNotificationTrigger else {
            return nil
        }
        return trigger.nextTriggerDate()
    }
}

#Preview {
    NavigationStack {
        ScheduledNotificationsView()
    }
}
