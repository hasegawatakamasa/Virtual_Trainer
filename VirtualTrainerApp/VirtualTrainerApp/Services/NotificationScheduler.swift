import Foundation
import UIKit

/// 通知スケジュール生成サービス
/// 要件対応: 3.1, 3.2, 3.3, 3.4, 3.6
class NotificationScheduler {
    private let oshiTrainerSettings: OshiTrainerSettings

    init(oshiTrainerSettings: OshiTrainerSettings = .shared) {
        self.oshiTrainerSettings = oshiTrainerSettings
    }

    // MARK: - Public Methods

    /// 空き時間スロットから通知候補を生成
    /// - Parameters:
    ///   - slots: 空き時間スロット配列
    ///   - settings: 通知設定
    /// - Returns: 通知候補配列
    func generateNotificationCandidates(from slots: [AvailableTimeSlot], settings: NotificationSettings) -> [NotificationCandidate] {
        var candidates: [NotificationCandidate] = []

        for slot in slots {
            // 通知時刻を計算（隙間時間の場合は開始10分後、それ以外はそのまま）
            let notificationTime: Date
            switch slot.slotType {
            case .gapTime:
                notificationTime = slot.startTime.addingTimeInterval(10 * 60)  // 10分後
            case .morningSlot, .eveningSlot, .freeDay:
                notificationTime = slot.startTime
            }

            // 過去の時刻はスキップ
            guard notificationTime > Date() else { continue }

            // 時間帯フィルタ（NotificationTimeRangeを使用）
            let timeRange = NotificationTimeRange(start: settings.timeRangeStart, end: settings.timeRangeEnd)
            guard timeRange.contains(notificationTime) else {
                continue
            }

            // 週末フィルタ
            if settings.weekendOnly {
                let calendar = Calendar.current
                let weekday = calendar.component(.weekday, from: notificationTime)
                guard weekday == 1 || weekday == 7 else { continue }  // 日曜=1, 土曜=7
            }

            // メッセージ生成
            let message = generateMessage(for: slot.slotType)

            // 優先順位計算（長い空き時間ほど高優先度）
            let priority = Int(slot.duration / 60)  // 分単位

            let candidate = NotificationCandidate(
                scheduledTime: notificationTime,
                slot: slot,
                priority: priority,
                trainerId: oshiTrainerSettings.selectedTrainer.id,
                message: message,
                imageAttachment: nil  // 画像は後で添付
            )

            candidates.append(candidate)
        }

        // 候補を処理
        var filteredCandidates = excludePastNotifications(candidates)
        filteredCandidates = prioritizeCandidates(filteredCandidates)
        filteredCandidates = filterByFrequency(filteredCandidates, frequency: settings.frequency)

        return filteredCandidates
    }

    /// 通知候補に優先順位を付与
    /// - Parameter candidates: 通知候補配列
    /// - Returns: 優先順位付き候補配列
    func prioritizeCandidates(_ candidates: [NotificationCandidate]) -> [NotificationCandidate] {
        // 優先順位でソート（高い順）
        return candidates.sorted { $0.priority > $1.priority }
    }

    /// ユーザー設定に応じて候補を絞り込み
    /// - Parameters:
    ///   - candidates: 通知候補配列
    ///   - frequency: 通知頻度設定
    /// - Returns: 絞り込み後の候補配列
    func filterByFrequency(_ candidates: [NotificationCandidate], frequency: NotificationSettings.NotificationFrequency) -> [NotificationCandidate] {
        let maxDaily = frequency.maxDailyNotifications
        var result: [NotificationCandidate] = []
        let calendar = Calendar.current

        // 日別にグループ化
        let groupedByDay = Dictionary(grouping: candidates) { candidate -> Date in
            calendar.startOfDay(for: candidate.scheduledTime)
        }

        // 各日の通知を制限
        for (_, dayCandidates) in groupedByDay.sorted(by: { $0.key < $1.key }) {
            let limitedCandidates = Array(dayCandidates.prefix(maxDaily))
            result.append(contentsOf: limitedCandidates)
        }

        return result
    }

    /// 過去の通知時刻を除外
    /// - Parameter candidates: 通知候補配列
    /// - Returns: 有効な候補のみの配列
    func excludePastNotifications(_ candidates: [NotificationCandidate]) -> [NotificationCandidate] {
        let now = Date()
        return candidates.filter { $0.scheduledTime > now }
    }

    /// アプリアクティブ中かチェック
    func isAppActive() -> Bool {
        return UIApplication.shared.applicationState == .active
    }

    // MARK: - Private Methods

    /// スロットタイプに応じたメッセージ生成
    private func generateMessage(for slotType: AvailableTimeSlot.TimeSlotType) -> String {
        let trainer = oshiTrainerSettings.selectedTrainer

        switch slotType {
        case .gapTime:
            return generateGapTimeMessage(trainer: trainer)
        case .morningSlot:
            return generateMorningMessage(trainer: trainer)
        case .eveningSlot:
            return generateEveningMessage(trainer: trainer)
        case .freeDay:
            return generateFreeDayMessage(trainer: trainer)
        }
    }

    private func generateGapTimeMessage(trainer: OshiTrainer) -> String {
        let messages = [
            "ちょっと時間できたんやない？軽く体動かさへん？",
            "隙間時間発見！トレーニングのチャンスやで〜",
            "今なら時間あるやん。一緒にトレーニングしよ？"
        ]
        return messages.randomElement() ?? messages[0]
    }

    private func generateMorningMessage(trainer: OshiTrainer) -> String {
        let messages = [
            "おはよ！朝の時間、トレーニングにどう？",
            "朝の空き時間やん。体動かしたらスッキリするで！",
            "今日も頑張っていこな！朝トレしよ？"
        ]
        return messages.randomElement() ?? messages[0]
    }

    private func generateEveningMessage(trainer: OshiTrainer) -> String {
        let messages = [
            "夜の空き時間やん。軽くトレーニングどう？",
            "今日も1日お疲れ様！最後にトレーニングしとく？",
            "夜やけど、少し体動かさへん？"
        ]
        return messages.randomElement() ?? messages[0]
    }

    private func generateFreeDayMessage(trainer: OshiTrainer) -> String {
        let messages = [
            "今日は時間あるみたいやん。一緒にトレーニングしよ？",
            "予定なし？なら、ゆっくりトレーニングできるで！",
            "今日は余裕ありそうやな。しっかりトレーニングしよか！"
        ]
        return messages.randomElement() ?? messages[0]
    }
}