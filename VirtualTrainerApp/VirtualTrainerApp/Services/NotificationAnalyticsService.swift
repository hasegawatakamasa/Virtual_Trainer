import Foundation
import CoreData

/// 通知分析・学習サービス
/// 要件対応: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
final class NotificationAnalyticsService: @unchecked Sendable {
    private let coreDataManager: CoreDataManager

    init(coreDataManager: CoreDataManager = .shared) {
        self.coreDataManager = coreDataManager
    }

    // MARK: - Notification Recording

    /// 通知配信を記録
    /// - Parameters:
    ///   - notificationId: 通知ID
    ///   - scheduledTime: 配信時刻
    ///   - trainerId: トレーナーID
    func recordNotificationDelivery(notificationId: String, scheduledTime: Date, trainerId: String) async throws {
        let context = coreDataManager.backgroundContext

        await context.perform {
            let record = NotificationRecord(context: context)
            record.id = notificationId
            record.scheduledTime = scheduledTime
            record.deliveryTime = Date()
            record.trainerId = trainerId
            record.wasTapped = false
            record.linkedSessionId = nil
            record.slotType = "unknown"
            record.createdAt = Date()
            record.isInvalid = false

            do {
                try context.save()
                print("[NotificationAnalyticsService] Notification delivery recorded: \(notificationId)")
            } catch {
                print("[NotificationAnalyticsService] Failed to record notification delivery: \(error)")
            }
        }
    }

    /// 通知タップを記録
    /// - Parameter notificationId: 通知ID
    func recordNotificationTap(notificationId: String) async throws {
        let context = coreDataManager.backgroundContext

        await context.perform {
            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            do {
                if let record = try context.fetch(request).first {
                    record.wasTapped = true
                    try context.save()
                    print("[NotificationAnalyticsService] Notification tap recorded: \(notificationId)")
                }
            } catch {
                print("[NotificationAnalyticsService] Failed to record notification tap: \(error)")
            }
        }
    }

    /// 通知とセッションを紐付け
    /// - Parameters:
    ///   - notificationId: 通知ID
    ///   - sessionId: セッションID
    func linkNotificationToSession(notificationId: String, sessionId: String) async throws {
        let context = coreDataManager.backgroundContext

        await context.perform {
            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", notificationId)
            request.fetchLimit = 1

            do {
                if let record = try context.fetch(request).first {
                    record.linkedSessionId = sessionId
                    try context.save()
                    print("[NotificationAnalyticsService] Notification linked to session: \(notificationId) -> \(sessionId)")
                }
            } catch {
                print("[NotificationAnalyticsService] Failed to link notification to session: \(error)")
            }
        }
    }

    /// 無効通知をマーク（24時間経過後）
    func markInvalidNotifications() async throws {
        let context = coreDataManager.backgroundContext

        await context.perform {
            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            let oneDayAgo = Date().addingTimeInterval(-24 * 3600)

            request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: [
                NSPredicate(format: "scheduledTime < %@", oneDayAgo as NSDate),
                NSPredicate(format: "linkedSessionId == nil"),
                NSPredicate(format: "isInvalid == NO")
            ])

            do {
                let records = try context.fetch(request)
                for record in records {
                    record.isInvalid = true
                }
                if !records.isEmpty {
                    try context.save()
                    print("[NotificationAnalyticsService] Marked \(records.count) notifications as invalid")
                }
            } catch {
                print("[NotificationAnalyticsService] Failed to mark invalid notifications: \(error)")
            }
        }
    }

    // MARK: - Analytics

    /// 週次分析を実行
    /// - Returns: 分析結果
    func performWeeklyAnalysis() async throws -> NotificationAnalytics {
        let context = coreDataManager.backgroundContext

        return try await context.perform {
            let calendar = Calendar.current
            let now = Date()
            let weekStart = calendar.date(byAdding: .day, value: -7, to: now)!

            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "scheduledTime >= %@", weekStart as NSDate)

            let records = try context.fetch(request)

            let totalDelivered = records.count
            let totalTapped = records.filter { $0.wasTapped }.count
            let totalLinkedToSession = records.filter { $0.linkedSessionId != nil }.count

            let tapRate = totalDelivered > 0 ? Double(totalTapped) / Double(totalDelivered) : 0
            let conversionRate = totalDelivered > 0 ? Double(totalLinkedToSession) / Double(totalDelivered) : 0

            let optimalTimeSlots = self.calculateOptimalTimeSlotsSync(records: records, in: context)

            return NotificationAnalytics(
                period: DateInterval(start: weekStart, end: now),
                totalDelivered: totalDelivered,
                totalTapped: totalTapped,
                totalLinkedToSession: totalLinkedToSession,
                tapRate: tapRate,
                conversionRate: conversionRate,
                optimalTimeSlots: optimalTimeSlots
            )
        }
    }

    /// 最適時間帯を計算
    /// - Returns: 時間帯別スコア
    func calculateOptimalTimeSlots() async throws -> [NotificationAnalytics.TimeSlot: Double] {
        let context = coreDataManager.backgroundContext

        return try await context.perform {
            let calendar = Calendar.current
            let weekStart = calendar.date(byAdding: .day, value: -7, to: Date())!

            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "scheduledTime >= %@", weekStart as NSDate)

            let records = try context.fetch(request)
            return self.calculateOptimalTimeSlotsSync(records: records, in: context)
        }
    }

    nonisolated private func calculateOptimalTimeSlotsSync(records: [NotificationRecord], in context: NSManagedObjectContext) -> [NotificationAnalytics.TimeSlot: Double] {
        let calendar = Calendar.current
        var timeSlotScores: [NotificationAnalytics.TimeSlot: (success: Int, total: Int)] = [:]

        for record in records {
            let hour = calendar.component(.hour, from: record.scheduledTime ?? Date())
            let timeSlot = NotificationAnalytics.TimeSlot(hour: hour)

            if timeSlotScores[timeSlot] == nil {
                timeSlotScores[timeSlot] = (success: 0, total: 0)
            }

            let isSuccess = record.linkedSessionId != nil
            timeSlotScores[timeSlot]!.total += 1
            if isSuccess {
                timeSlotScores[timeSlot]!.success += 1
            }
        }

        // スコア計算（実施率）
        var result: [NotificationAnalytics.TimeSlot: Double] = [:]
        for (timeSlot, counts) in timeSlotScores {
            let score = counts.total > 0 ? Double(counts.success) / Double(counts.total) : 0
            result[timeSlot] = score
        }

        return result
    }

    /// 通知頻度を調整（学習ベース）
    /// - Parameter analytics: 分析結果
    /// - Returns: 推奨頻度調整
    func suggestFrequencyAdjustment(analytics: NotificationAnalytics) -> FrequencyAdjustment? {
        // タップ率が20%未満の場合、頻度を減らす
        if analytics.tapRate < 0.2 {
            return FrequencyAdjustment(
                suggestedFrequency: .modest,
                reason: "通知のタップ率が低いため、頻度を控えめに調整することをおすすめします。"
            )
        }

        // 実施率が50%以上の場合、頻度を増やす
        if analytics.conversionRate >= 0.5 {
            return FrequencyAdjustment(
                suggestedFrequency: .active,
                reason: "通知からのトレーニング実施率が高いため、頻度を増やすことをおすすめします。"
            )
        }

        return nil
    }

    /// 今週の通知効果を取得
    /// - Returns: 通知効果サマリー
    func getWeeklyEffectiveness() async throws -> NotificationEffectiveness {
        let context = coreDataManager.backgroundContext

        return try await context.perform {
            let calendar = Calendar.current
            let now = Date()
            let weekStart = calendar.date(byAdding: .day, value: -7, to: now)!

            let request: NSFetchRequest<NotificationRecord> = NotificationRecord.fetchRequest()
            request.predicate = NSPredicate(format: "scheduledTime >= %@", weekStart as NSDate)

            let records = try context.fetch(request)

            let deliveredCount = records.count
            let tappedCount = records.filter { $0.wasTapped }.count
            let sessionCount = records.filter { $0.linkedSessionId != nil }.count

            let effectiveness = deliveredCount > 0 ? Double(sessionCount) / Double(deliveredCount) : 0

            return NotificationEffectiveness(
                weekStart: weekStart,
                weekEnd: now,
                deliveredCount: deliveredCount,
                tappedCount: tappedCount,
                sessionCount: sessionCount,
                effectiveness: effectiveness
            )
        }
    }
}