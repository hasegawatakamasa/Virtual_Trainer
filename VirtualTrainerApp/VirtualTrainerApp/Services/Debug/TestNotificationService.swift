// DEBUG: Test notification service for debugging - remove in production

import Foundation
import UserNotifications
import SwiftUI

/// テスト通知送信サービス
/// 要件対応: 1.2, 1.3, 1.4, 1.5, 1.6
@MainActor
class TestNotificationService: ObservableObject {
    @Published var sendingStatus: NotificationSendStatus = .idle

    private let notificationCenter: UNUserNotificationCenter
    private let oshiSettings: OshiTrainerSettings

    init(
        notificationCenter: UNUserNotificationCenter = .current(),
        oshiSettings: OshiTrainerSettings = .shared
    ) {
        self.notificationCenter = notificationCenter
        self.oshiSettings = oshiSettings
    }

    /// テスト通知を即座に送信
    /// - Throws: NotificationError - 通知権限なし、システムエラー
    func sendTestNotification() async throws {
        sendingStatus = .sending

        do {
            // 通知権限チェック
            let settings = await notificationCenter.notificationSettings()
            guard settings.authorizationStatus == .authorized else {
                sendingStatus = .failure(TestNotificationError.permissionDenied)
                return
            }

            // 通知コンテンツを作成
            let content = UNMutableNotificationContent()
            let trainer = oshiSettings.selectedTrainer
            content.title = trainer.displayName
            content.body = "[テスト通知] 通知機能のテストです。このメッセージが表示されれば正常に動作しています。"
            content.sound = .default
            content.categoryIdentifier = "TEST_NOTIFICATION"
            content.userInfo = [
                "isTest": true,
                "trainerId": trainer.id,
                "timestamp": Date().timeIntervalSince1970
            ]

            // 画像添付は一旦無効化
            // if let attachment = try? await createTrainerImageAttachment(trainer: trainer) {
            //     content.attachments = [attachment]
            // }

            // 即座に配信（5秒後）
            let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 5, repeats: false)

            // リクエスト作成
            let request = UNNotificationRequest(
                identifier: "test-notification-\(UUID().uuidString)",
                content: content,
                trigger: trigger
            )

            // 通知をスケジュール
            try await notificationCenter.add(request)

            sendingStatus = .success
        } catch {
            sendingStatus = .failure(error)
            throw error
        }
    }

    /// 通知アタッチメント（トレーナー画像）を作成
    /// - Parameter trainer: トレーナー情報
    /// - Returns: UNNotificationAttachment
    private func createTrainerImageAttachment(trainer: OshiTrainer) async throws -> UNNotificationAttachment? {
        guard let imageURL = trainer.imageFileURL() else {
            return nil
        }

        // 一時ディレクトリにコピー
        let tempDirectory = FileManager.default.temporaryDirectory
        let tempFileURL = tempDirectory.appendingPathComponent("\(trainer.id)-\(UUID().uuidString).png")

        try? FileManager.default.copyItem(at: imageURL, to: tempFileURL)

        // サムネイル画像を全体表示（アイコン風）
        // LINEのようにアイコンとして表示させるため、画像全体を使用
        let options: [String: Any] = [
            UNNotificationAttachmentOptionsTypeHintKey: "public.png",
            // サムネイルの非表示領域を指定（0.0 = 表示、1.0 = 非表示）
            // x, y, width, height すべて 0-1 の範囲で、画像のどの部分を表示するか指定
            // 全体を表示する場合: x=0, y=0, width=1, height=1
            UNNotificationAttachmentOptionsThumbnailClippingRectKey: CGRect(x: 0.0, y: 0.0, width: 1.0, height: 1.0).dictionaryRepresentation as Any,
            // サムネイルを非表示にしない
            UNNotificationAttachmentOptionsThumbnailHiddenKey: false
        ]

        let attachment = try UNNotificationAttachment(identifier: "trainer-image", url: tempFileURL, options: options)
        return attachment
    }

    /// 現在のステータスをリセット
    func resetStatus() {
        sendingStatus = .idle
    }
}

enum NotificationSendStatus {
    case idle
    case sending
    case success
    case failure(Error)
}

enum TestNotificationError: LocalizedError {
    case permissionDenied
    case systemError

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "通知の権限が許可されていません。設定アプリで通知を有効にしてください。"
        case .systemError:
            return "通知の送信中にエラーが発生しました。"
        }
    }
}
