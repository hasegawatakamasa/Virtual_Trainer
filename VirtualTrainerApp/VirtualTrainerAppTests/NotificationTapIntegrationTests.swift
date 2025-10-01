import XCTest
import CoreData
@testable import VirtualTrainerApp

/// 統合テスト：通知タップからトレーニング開始までのフロー
/// 要件対応: タスク21 - 4.7, 8.2, 8.3
final class NotificationTapIntegrationTests: XCTestCase {

    var analyticsService: NotificationAnalyticsService!
    var coreDataManager: CoreDataManager!

    override func setUpWithError() throws {
        try super.setUpWithError()
        coreDataManager = CoreDataManager.shared
        analyticsService = NotificationAnalyticsService(coreDataManager: coreDataManager)
    }

    override func tearDownWithError() throws {
        // テストデータをクリーンアップ
        cleanupTestData()

        analyticsService = nil
        coreDataManager = nil
        try super.tearDownWithError()
    }

    // MARK: - Test: Notification Tap Recording

    /// 通知タップの記録テスト
    func testNotificationTapRecording() async throws {
        // Given: 通知IDとスケジュール時刻
        let notificationId = "test-notification-\(UUID().uuidString)"
        let scheduledTime = Date()
        let trainerId = "oshi-ai"

        // When: 通知配信を記録
        try await analyticsService.recordNotificationDelivery(
            notificationId: notificationId,
            scheduledTime: scheduledTime,
            trainerId: trainerId
        )

        // Then: 通知レコードが作成されること
        let context = coreDataManager.backgroundContext
        let record = try await context.perform {
            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            let records = try context.fetch(request)
            return records.first
        }

        XCTAssertNotNil(record, "Notification record should be created")
        XCTAssertEqual(record?.id, notificationId)
        XCTAssertEqual(record?.trainerId, trainerId)
        XCTAssertFalse(record?.wasTapped ?? true, "wasTapped should be false initially")
        XCTAssertNil(record?.linkedSessionId, "linkedSessionId should be nil initially")

        // When: 通知タップを記録
        try await analyticsService.recordNotificationTap(notificationId: notificationId)

        // 少し待機してCoreDataの変更を確実に保存
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1秒

        // Then: wasTappedがtrueになること
        let tappedRecord = try await context.perform {
            // コンテキストをリフレッシュ
            context.refreshAllObjects()

            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            let records = try context.fetch(request)
            return records.first
        }

        XCTAssertTrue(tappedRecord?.wasTapped ?? false, "wasTapped should be true after tap")
    }

    // MARK: - Test: Notification to Session Linking

    /// 通知からセッションへの紐付けテスト
    func testNotificationToSessionLinking() async throws {
        // Given: 通知レコード
        let notificationId = "test-notification-\(UUID().uuidString)"
        let scheduledTime = Date()
        let trainerId = "oshi-ai"

        try await analyticsService.recordNotificationDelivery(
            notificationId: notificationId,
            scheduledTime: scheduledTime,
            trainerId: trainerId
        )

        try await analyticsService.recordNotificationTap(notificationId: notificationId)

        // When: セッションIDを紐付け
        let sessionId = "session-\(UUID().uuidString)"
        try await analyticsService.linkNotificationToSession(
            notificationId: notificationId,
            sessionId: sessionId
        )

        // 少し待機してCoreDataの変更を確実に保存
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1秒

        // Then: linkedSessionIdが設定されること
        let context = coreDataManager.backgroundContext
        let linkedRecord = try await context.perform {
            context.refreshAllObjects()

            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            let records = try context.fetch(request)
            return records.first
        }

        XCTAssertEqual(linkedRecord?.linkedSessionId, sessionId, "Session should be linked to notification")
        XCTAssertTrue(linkedRecord?.wasTapped ?? false, "Tapped flag should remain true")
    }

    // MARK: - Test: Complete Flow

    /// 通知タップからセッション開始までの完全フロー
    func testCompleteNotificationToSessionFlow() async throws {
        // Given: 通知配信
        let notificationId = "test-notification-\(UUID().uuidString)"
        let scheduledTime = Date()
        let trainerId = "oshi-ai"

        try await analyticsService.recordNotificationDelivery(
            notificationId: notificationId,
            scheduledTime: scheduledTime,
            trainerId: trainerId
        )

        // When: ユーザーが通知をタップ
        try await analyticsService.recordNotificationTap(notificationId: notificationId)

        // 少し待機
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: wasTappedがtrueになること
        let context = coreDataManager.backgroundContext
        var record = try await context.perform {
            context.refreshAllObjects()

            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            let records = try context.fetch(request)
            return records.first
        }

        XCTAssertTrue(record?.wasTapped ?? false, "Notification should be marked as tapped")

        // When: ユーザーがトレーニングを開始
        let sessionId = "session-\(UUID().uuidString)"
        try await analyticsService.linkNotificationToSession(
            notificationId: notificationId,
            sessionId: sessionId
        )

        // 少し待機
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: セッションが紐付けられること
        record = try await context.perform {
            context.refreshAllObjects()

            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            let records = try context.fetch(request)
            return records.first
        }

        XCTAssertEqual(record?.linkedSessionId, sessionId, "Session should be linked")
        XCTAssertFalse(record?.isInvalid ?? true, "Valid notification should not be marked invalid")
    }

    // MARK: - Test: Invalid Notification Marking

    /// 無効通知のマーキングテスト（24時間経過）
    func testInvalidNotificationMarking() async throws {
        // Given: 24時間以上前の通知（トレーニング未実施）
        let notificationId = "test-notification-\(UUID().uuidString)"
        let oldScheduledTime = Date().addingTimeInterval(-25 * 3600)  // 25時間前
        let trainerId = "oshi-ai"

        try await analyticsService.recordNotificationDelivery(
            notificationId: notificationId,
            scheduledTime: oldScheduledTime,
            trainerId: trainerId
        )

        // When: 無効通知をマーク
        try await analyticsService.markInvalidNotifications()

        // Then: isInvalidがtrueになること
        let context = coreDataManager.backgroundContext
        let record = try await context.perform {
            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            let records = try context.fetch(request)
            return records.first
        }

        XCTAssertTrue(record?.isInvalid ?? false, "Old notification without session should be marked invalid")
    }

    // MARK: - Test: Valid Notification Not Marked Invalid

    /// 有効通知が無効とマークされないことを確認
    func testValidNotificationNotMarkedInvalid() async throws {
        // Given: 24時間以上前の通知（トレーニング実施済み）
        let notificationId = "test-notification-\(UUID().uuidString)"
        let oldScheduledTime = Date().addingTimeInterval(-25 * 3600)  // 25時間前
        let trainerId = "oshi-ai"
        let sessionId = "session-\(UUID().uuidString)"

        try await analyticsService.recordNotificationDelivery(
            notificationId: notificationId,
            scheduledTime: oldScheduledTime,
            trainerId: trainerId
        )

        try await analyticsService.linkNotificationToSession(
            notificationId: notificationId,
            sessionId: sessionId
        )

        // When: 無効通知をマーク
        try await analyticsService.markInvalidNotifications()

        // Then: isInvalidはfalseのまま（セッション紐付け済み）
        let context = coreDataManager.backgroundContext
        let record = try await context.perform {
            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            let records = try context.fetch(request)
            return records.first
        }

        XCTAssertFalse(record?.isInvalid ?? true, "Notification with linked session should not be marked invalid")
    }

    // MARK: - Helper Methods

    /// テストデータのクリーンアップ
    private func cleanupTestData() {
        let context = coreDataManager.backgroundContext
        context.performAndWait {
            let request: NSFetchRequest<NSFetchRequestResult> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id BEGINSWITH %@", "test-notification-")

            let deleteRequest = NSBatchDeleteRequest(fetchRequest: request)

            do {
                try context.execute(deleteRequest)
                try context.save()
            } catch {
                print("Failed to cleanup test data: \(error)")
            }
        }
    }
}