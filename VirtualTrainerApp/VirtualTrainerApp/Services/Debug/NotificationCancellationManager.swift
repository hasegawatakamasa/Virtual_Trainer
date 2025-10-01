// DEBUG: Notification cancellation manager for testing - remove in production

import Foundation
import UserNotifications

/// 通知キャンセル管理サービス
/// 要件対応: 3.3, 3.4, 3.6
class NotificationCancellationManager {
    private let notificationCenter: UNUserNotificationCenter

    init(notificationCenter: UNUserNotificationCenter = .current()) {
        self.notificationCenter = notificationCenter
    }

    /// 個別通知をキャンセル
    /// - Parameter identifier: 通知ID
    func cancelNotification(identifier: String) {
        notificationCenter.removePendingNotificationRequests(withIdentifiers: [identifier])
    }

    /// 複数通知を一括キャンセル
    /// - Parameter identifiers: 通知ID配列
    func cancelNotifications(identifiers: [String]) {
        notificationCenter.removePendingNotificationRequests(withIdentifiers: identifiers)
    }

    /// 全ての予約通知をキャンセル
    func cancelAllNotifications() {
        notificationCenter.removeAllPendingNotificationRequests()
    }
}
