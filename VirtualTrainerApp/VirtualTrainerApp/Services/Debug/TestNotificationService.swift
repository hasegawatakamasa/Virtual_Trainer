// DEBUG: Test notification service for debugging - remove in production

import Foundation
import UserNotifications
import SwiftUI
import Intents

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

            let trainer = oshiSettings.selectedTrainer
            let message = "[テスト通知] 通知機能のテストです。このメッセージが表示されれば正常に動作しています。"

            // Communication Notifications形式で通知コンテンツを作成
            let content: UNNotificationContent
            if #available(iOS 15.0, *) {
                print("[TestNotificationService] Communication Notification形式でテスト通知作成")
                content = try createCommunicationNotification(trainer: trainer, message: message)
            } else {
                print("[TestNotificationService] 通常通知形式でテスト通知作成")
                content = createStandardNotification(trainer: trainer, message: message)
            }

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
            print("[TestNotificationService] テスト通知スケジュール完了")

            sendingStatus = .success
        } catch {
            print("[TestNotificationService] テスト通知送信失敗: \(error)")
            sendingStatus = .failure(error)
            throw error
        }
    }

    /// Communication Notifications形式の通知コンテンツを作成（iOS 15+）
    @available(iOS 15.0, *)
    private func createCommunicationNotification(trainer: OshiTrainer, message: String) throws -> UNNotificationContent {
        print("[TestNotificationService] Communication Notification作成開始")

        // INPersonを作成
        let sender = createINPerson(from: trainer)
        print("[TestNotificationService] INPerson作成完了: \(sender.displayName)")

        // INSendMessageIntentを作成
        let intent = INSendMessageIntent(
            recipients: nil,
            outgoingMessageType: .outgoingMessageText,
            content: message,
            speakableGroupName: nil,
            conversationIdentifier: trainer.id,
            serviceName: nil,
            sender: sender,
            attachments: nil
        )
        print("[TestNotificationService] INSendMessageIntent作成完了")

        // 送信者画像を設定
        if let image = createINImage(from: trainer) {
            intent.setImage(image, forParameterNamed: \.sender)
            print("[TestNotificationService] 送信者画像設定完了")
        } else {
            print("[TestNotificationService] ⚠️ 送信者画像の設定に失敗")
        }

        // Intentから通知コンテンツを生成
        let content = UNMutableNotificationContent()
        content.title = trainer.displayName
        content.body = message
        content.sound = .default
        content.categoryIdentifier = "TEST_NOTIFICATION"
        content.userInfo = [
            "isTest": true,
            "trainerId": trainer.id,
            "timestamp": Date().timeIntervalSince1970
        ]

        // INInteractionをdonateしてからupdating
        let interaction = INInteraction(intent: intent, response: nil)
        interaction.direction = .incoming
        interaction.donate(completion: { error in
            if let error = error {
                print("[TestNotificationService] ⚠️ Interaction donate失敗: \(error)")
            } else {
                print("[TestNotificationService] Interaction donate成功")
            }
        })

        // Intentを使ってコンテンツを更新
        do {
            let updatedContent = try content.updating(from: intent)
            print("[TestNotificationService] ✅ Communication Notification変換成功")
            return updatedContent
        } catch {
            print("[TestNotificationService] ⚠️ Communication Notification変換失敗: \(error)")
            throw error
        }
    }

    /// 通常通知コンテンツを作成（iOS 15未満用）
    private func createStandardNotification(trainer: OshiTrainer, message: String) -> UNNotificationContent {
        let content = UNMutableNotificationContent()
        content.title = trainer.displayName
        content.body = message
        content.sound = .default
        content.categoryIdentifier = "TEST_NOTIFICATION"
        content.userInfo = [
            "isTest": true,
            "trainerId": trainer.id,
            "timestamp": Date().timeIntervalSince1970
        ]
        return content
    }

    /// INPersonを作成
    private func createINPerson(from trainer: OshiTrainer) -> INPerson {
        let personHandle = INPersonHandle(value: trainer.id, type: .unknown)

        var nameComponents = PersonNameComponents()
        nameComponents.nickname = trainer.displayName

        let image = createINImage(from: trainer)

        return INPerson(
            personHandle: personHandle,
            nameComponents: nameComponents,
            displayName: trainer.displayName,
            image: image,
            contactIdentifier: nil,
            customIdentifier: trainer.id
        )
    }

    /// INImageを作成
    private func createINImage(from trainer: OshiTrainer) -> INImage? {
        guard let imageURL = trainer.imageFileURL() else {
            print("[TestNotificationService] トレーナー画像URL取得失敗: \(trainer.displayName)")
            return nil
        }

        guard let imageData = try? Data(contentsOf: imageURL) else {
            print("[TestNotificationService] 画像データ読み込み失敗: \(imageURL.path)")
            return nil
        }

        print("[TestNotificationService] 画像読み込み成功: \(trainer.displayName)")
        return INImage(imageData: imageData)
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
