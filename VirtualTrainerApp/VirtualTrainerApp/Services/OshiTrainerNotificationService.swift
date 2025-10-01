import Foundation
import UserNotifications

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

        // 通知コンテンツを作成
        let content = UNMutableNotificationContent()
        content.title = trainer.displayName
        content.body = candidate.message
        content.sound = .default
        content.categoryIdentifier = "TRAINING_INVITATION"
        content.userInfo = [
            "notificationId": candidate.id.uuidString,
            "trainerId": trainer.id,
            "slotType": candidate.slot.slotType.rawValue
        ]

        // トレーナー画像を添付
        if let attachment = try? await createTrainerImageAttachment(trainer: trainer) {
            content.attachments = [attachment]
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

        let attachment = try UNNotificationAttachment(identifier: "trainer-image", url: tempFileURL, options: nil)
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