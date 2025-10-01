//
//  NotificationService.swift
//  NotificationServiceExtension
//
//  Created by 正留慎也 on 2025/10/01.
//

import UserNotifications
import Intents

class NotificationService: UNNotificationServiceExtension {

    var contentHandler: ((UNNotificationContent) -> Void)?
    var bestAttemptContent: UNMutableNotificationContent?

    override func didReceive(_ request: UNNotificationRequest, withContentHandler contentHandler: @escaping (UNNotificationContent) -> Void) {
        self.contentHandler = contentHandler
        bestAttemptContent = (request.content.mutableCopy() as? UNMutableNotificationContent)

        guard let bestAttemptContent = bestAttemptContent else {
            contentHandler(request.content)
            return
        }

        // iOS 15未満の場合は通常通知として配信
        guard isCommunicationNotificationsSupported() else {
            print("[NotificationService] iOS 15未満のため通常通知として配信")
            contentHandler(bestAttemptContent)
            return
        }

        // App Group共有UserDefaultsからトレーナー情報取得
        guard let trainer = getSelectedTrainer() else {
            print("[NotificationService] トレーナー情報取得失敗、通常通知として配信")
            contentHandler(bestAttemptContent)
            return
        }

        print("[NotificationService] トレーナー情報取得成功: \(trainer.displayName)")

        // Communication Notifications形式に変換
        let communicationContent = convertToCommunicationNotification(
            originalContent: request.content,
            trainer: trainer
        )

        contentHandler(communicationContent)
    }

    override func serviceExtensionTimeWillExpire() {
        // 時間切れの場合は元の通知を配信
        if let contentHandler = contentHandler,
           let bestAttemptContent = bestAttemptContent {
            print("[NotificationService] Extension時間超過、フォールバック処理を実行")
            contentHandler(bestAttemptContent)
        }
    }

    // MARK: - iOS Version Check

    /// iOS 15以上でCommunication Notificationsがサポートされているか確認
    private func isCommunicationNotificationsSupported() -> Bool {
        if #available(iOS 15.0, *) {
            return true
        } else {
            return false
        }
    }

    // MARK: - Trainer Information

    /// App Group共有UserDefaultsから選択されたトレーナーを取得
    private func getSelectedTrainer() -> OshiTrainer? {
        guard let sharedDefaults = UserDefaults(suiteName: "group.com.yourcompany.VirtualTrainer") else {
            print("[NotificationService] App Group UserDefaults取得失敗")
            return .oshinoAi // デフォルトトレーナー
        }

        guard let trainerId = sharedDefaults.string(forKey: "selectedOshiTrainerId") else {
            print("[NotificationService] トレーナーID未設定、デフォルトトレーナーを使用")
            return .oshinoAi
        }

        let trainer = OshiTrainer.allTrainers.first(where: { $0.id == trainerId }) ?? .oshinoAi
        print("[NotificationService] トレーナー取得: \(trainer.displayName) (ID: \(trainerId))")
        return trainer
    }

    // MARK: - INImage Creation

    /// トレーナー画像からINImageを作成
    private func createINImage(from trainer: OshiTrainer) -> INImage? {
        guard let imageURL = trainer.imageFileURL() else {
            print("[NotificationService] トレーナー画像URL取得失敗: \(trainer.displayName)")
            return nil
        }

        guard let imageData = try? Data(contentsOf: imageURL) else {
            print("[NotificationService] 画像データ読み込み失敗: \(imageURL.path)")
            return nil
        }

        print("[NotificationService] 画像読み込み成功: \(trainer.displayName)")
        return INImage(imageData: imageData)
    }

    // MARK: - INPerson Creation

    /// トレーナー情報からINPersonを作成
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

    // MARK: - Communication Notification Conversion

    /// 通常通知をCommunication Notifications形式に変換
    @available(iOS 15.0, *)
    private func convertToCommunicationNotification(
        originalContent: UNNotificationContent,
        trainer: OshiTrainer
    ) -> UNNotificationContent {
        let sender = createINPerson(from: trainer)

        // INSendMessageIntentの作成
        let intent = INSendMessageIntent(
            recipients: nil,
            outgoingMessageType: .outgoingMessageText,
            content: originalContent.body,
            speakableGroupName: nil,
            conversationIdentifier: trainer.id,
            serviceName: nil,
            sender: sender,
            attachments: nil
        )

        // 送信者画像を明示的に設定
        if let image = createINImage(from: trainer) {
            intent.setImage(image, forParameterNamed: \.sender)
        }

        // INInteractionを作成してdonate
        let interaction = INInteraction(intent: intent, response: nil)
        interaction.direction = .incoming
        interaction.donate(completion: { error in
            if let error = error {
                print("[NotificationService] Interaction donate失敗: \(error.localizedDescription)")
            } else {
                print("[NotificationService] Interaction donate成功")
            }
        })

        // 通知コンテンツを変換
        do {
            let updatedContent = try originalContent.updating(from: intent)
            print("[NotificationService] Communication Notification変換成功")
            return updatedContent
        } catch {
            print("[NotificationService] Communication Notification変換失敗: \(error.localizedDescription)")
            return originalContent
        }
    }
}
