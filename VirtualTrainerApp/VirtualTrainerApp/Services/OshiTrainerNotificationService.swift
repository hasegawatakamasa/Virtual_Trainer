import Foundation
import UserNotifications
import Intents

/// 推しトレーナー通知サービス
/// 要件対応: 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
@MainActor
class OshiTrainerNotificationService: ObservableObject {
    private let notificationCenter: UNUserNotificationCenter
    private let oshiTrainerSettings: OshiTrainerSettings
    private let analyticsService: NotificationAnalyticsService?

    init(
        notificationCenter: UNUserNotificationCenter = .current(),
        oshiTrainerSettings: OshiTrainerSettings = .shared,
        analyticsService: NotificationAnalyticsService? = nil
    ) {
        self.notificationCenter = notificationCenter
        self.oshiTrainerSettings = oshiTrainerSettings
        self.analyticsService = analyticsService
    }

    // MARK: - Public Methods

    /// 通知候補から実際の通知を作成・スケジュール
    /// - Parameter candidates: 通知候補配列
    /// - Returns: スケジュール成功数
    func scheduleNotifications(_ candidates: [NotificationCandidate]) async throws -> Int {
        // 通知権限チェック
        let authStatus = await checkNotificationPermission()
        guard authStatus == .authorized else {
            throw NotificationSchedulingError.notificationPermissionDenied
        }

        var successCount = 0

        for candidate in candidates {
            do {
                try await scheduleNotification(candidate)
                successCount += 1

                // アナリティクスに記録
                try? await analyticsService?.recordNotificationDelivery(
                    notificationId: candidate.id.uuidString,
                    scheduledTime: candidate.scheduledTime,
                    trainerId: candidate.trainerId
                )
            } catch {
                print("[OshiTrainerNotificationService] Failed to schedule notification: \(error)")
            }
        }

        return successCount
    }

    /// 単一の通知をスケジュール
    private func scheduleNotification(_ candidate: NotificationCandidate) async throws {
        let trainer = oshiTrainerSettings.selectedTrainer

        // Communication Notifications形式の通知コンテンツを作成
        let content: UNNotificationContent
        if #available(iOS 15.0, *) {
            content = try createCommunicationNotification(trainer: trainer, message: candidate.message, candidate: candidate)
        } else {
            // iOS 15未満の場合は通常通知
            content = createStandardNotification(trainer: trainer, message: candidate.message, candidate: candidate)
        }

        // トリガー作成（指定時刻）
        let calendar = Calendar.current
        let components = calendar.dateComponents([.year, .month, .day, .hour, .minute], from: candidate.scheduledTime)
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: false)

        // リクエスト作成
        let request = UNNotificationRequest(
            identifier: candidate.id.uuidString,
            content: content,
            trigger: trigger
        )

        // 通知をスケジュール
        try await notificationCenter.add(request)
    }

    /// Communication Notifications形式の通知コンテンツを作成（iOS 15+）
    @available(iOS 15.0, *)
    private func createCommunicationNotification(trainer: OshiTrainer, message: String, candidate: NotificationCandidate) throws -> UNNotificationContent {
        print("[OshiTrainerNotificationService] Communication Notification作成開始")

        // INPersonを作成
        let sender = createINPerson(from: trainer)
        print("[OshiTrainerNotificationService] INPerson作成完了: \(sender.displayName)")

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
        print("[OshiTrainerNotificationService] INSendMessageIntent作成完了")

        // 送信者画像を設定
        if let image = createINImage(from: trainer) {
            intent.setImage(image, forParameterNamed: \.sender)
            print("[OshiTrainerNotificationService] 送信者画像設定完了")
        } else {
            print("[OshiTrainerNotificationService] ⚠️ 送信者画像の設定に失敗")
        }

        // Intentから通知コンテンツを生成
        let content = UNMutableNotificationContent()
        content.title = trainer.displayName
        content.body = message
        content.sound = .default
        content.categoryIdentifier = "TRAINING_INVITATION"
        content.userInfo = [
            "notificationId": candidate.id.uuidString,
            "trainerId": trainer.id,
            "slotType": candidate.slot.slotType.rawValue
        ]

        // INInteractionをdonateしてからupdating
        let interaction = INInteraction(intent: intent, response: nil)
        interaction.direction = .incoming
        interaction.donate(completion: { error in
            if let error = error {
                print("[OshiTrainerNotificationService] ⚠️ Interaction donate失敗: \(error)")
            } else {
                print("[OshiTrainerNotificationService] Interaction donate成功")
            }
        })

        // Intentを使ってコンテンツを更新
        do {
            let updatedContent = try content.updating(from: intent)
            print("[OshiTrainerNotificationService] ✅ Communication Notification変換成功")
            return updatedContent
        } catch {
            print("[OshiTrainerNotificationService] ⚠️ Communication Notification変換失敗: \(error)")
            throw error
        }
    }

    /// 通常通知コンテンツを作成（iOS 15未満用）
    private func createStandardNotification(trainer: OshiTrainer, message: String, candidate: NotificationCandidate) -> UNNotificationContent {
        let content = UNMutableNotificationContent()
        content.title = trainer.displayName
        content.body = message
        content.sound = .default
        content.categoryIdentifier = "TRAINING_INVITATION"
        content.userInfo = [
            "notificationId": candidate.id.uuidString,
            "trainerId": trainer.id,
            "slotType": candidate.slot.slotType.rawValue
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
            print("[OshiTrainerNotificationService] トレーナー画像URL取得失敗: \(trainer.displayName)")
            return nil
        }

        guard let imageData = try? Data(contentsOf: imageURL) else {
            print("[OshiTrainerNotificationService] 画像データ読み込み失敗: \(imageURL.path)")
            return nil
        }

        print("[OshiTrainerNotificationService] 画像読み込み成功: \(trainer.displayName)")
        return INImage(imageData: imageData)
    }

    /// 推しトレーナーに応じたメッセージを生成
    /// - Parameters:
    ///   - trainer: トレーナー情報
    ///   - slotType: 時間帯タイプ
    /// - Returns: パーソナライズドメッセージ
    func generateTrainerMessage(trainer: OshiTrainer, slotType: AvailableTimeSlot.TimeSlotType) -> String {
        switch slotType {
        case .gapTime:
            return "ちょっと時間できたんやない？軽く体動かさへん？"
        case .morningSlot:
            return "おはよ！朝の時間、トレーニングにどう？"
        case .eveningSlot:
            return "夜の空き時間やん。軽くトレーニングどう？"
        case .freeDay:
            return "今日は時間あるみたいやん。一緒にトレーニングしよ？"
        }
    }

    /// 通知アタッチメント（トレーナー画像）を作成
    /// - Parameter trainer: トレーナー情報
    /// - Returns: UNNotificationAttachment
    func createTrainerImageAttachment(trainer: OshiTrainer) async throws -> UNNotificationAttachment? {
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

    /// 全ての pending 通知をキャンセル
    func cancelAllNotifications() async {
        notificationCenter.removeAllPendingNotificationRequests()
    }

    /// 特定の通知IDをキャンセル
    func cancelNotification(withId id: String) async {
        notificationCenter.removePendingNotificationRequests(withIdentifiers: [id])
    }

    /// 通知権限をリクエスト
    func requestNotificationPermission() async throws -> Bool {
        let granted = try await notificationCenter.requestAuthorization(options: [.alert, .sound, .badge])
        return granted
    }

    /// 通知権限ステータスを確認
    func checkNotificationPermission() async -> UNAuthorizationStatus {
        let settings = await notificationCenter.notificationSettings()
        return settings.authorizationStatus
    }

    /// 次回の通知を取得（プレビュー用）
    func getNextNotifications(limit: Int = 3) async -> [NotificationPreview] {
        let requests = await notificationCenter.pendingNotificationRequests()

        let previews = requests.prefix(limit).compactMap { request -> NotificationPreview? in
            guard let trigger = request.trigger as? UNCalendarNotificationTrigger,
                  let nextTriggerDate = trigger.nextTriggerDate() else {
                return nil
            }

            return NotificationPreview(
                id: request.identifier,
                scheduledTime: nextTriggerDate,
                title: request.content.title,
                body: request.content.body
            )
        }

        return previews.sorted { $0.scheduledTime < $1.scheduledTime }
    }
}

// MARK: - NotificationPreview

struct NotificationPreview: Identifiable {
    let id: String
    let scheduledTime: Date
    let title: String
    let body: String
}