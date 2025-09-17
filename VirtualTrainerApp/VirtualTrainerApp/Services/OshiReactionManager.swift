import Foundation
import Combine
import AVFoundation

@MainActor
class OshiReactionManager: ObservableObject {
    static let shared = OshiReactionManager()
    
    @Published var recentReactions: [OshiReaction] = []
    @Published var reactionCooldowns: [String: Date] = [:]
    
    private let audioFeedbackService = AudioFeedbackService()
    private let achievementSystem = AchievementSystem.shared
    private let trainingSessionService = TrainingSessionService.shared
    private var cancellables = Set<AnyCancellable>()
    
    // クールダウン設定（秒）
    private let cooldownDurations: [ReactionType: TimeInterval] = [
        .newRecord: 300,        // 5分
        .streak: 1800,          // 30分
        .milestone: 600,        // 10分
        .encouragement: 3600,   // 1時間
        .levelUp: 60            // 1分
    ]
    
    private init() {
        setupReactionSubscriptions()
    }
    
    // MARK: - Reaction Triggers
    func checkNewRecord(session: SessionSummary) {
        guard canTriggerReaction(.newRecord) else { return }
        
        let personalBests = trainingSessionService.calculatePersonalBests()
        let newRecords = personalBests.filter { $0.isNewRecord }
        
        for record in newRecords {
            let reaction = OshiReaction(
                type: .newRecord,
                characterName: session.characterName,
                message: generateNewRecordMessage(record),
                audioFileName: getNewRecordAudioFile(session.characterName),
                triggeredAt: Date(),
                metadata: ["recordType": record.recordType.rawValue, "value": "\(record.value)"]
            )
            
            triggerReaction(reaction)
            
            // 絆ポイント付与
            achievementSystem.awardBondPoints(
                calculateNewRecordBondPoints(record),
                to: session.characterName,
                reason: "新記録達成: \(record.recordType.displayName)"
            )
        }
    }
    
    func checkStreak() {
        guard canTriggerReaction(.streak) else { return }
        
        let currentStreak = calculateCurrentStreak()
        let streakMilestones = [3, 7, 14, 30]
        let characterName = getCurrentCharacter()
        
        for milestone in streakMilestones.reversed() {
            if currentStreak == milestone {
                let reaction = OshiReaction(
                    type: .streak,
                    characterName: characterName,
                    message: generateStreakMessage(days: milestone),
                    audioFileName: getStreakAudioFile(characterName, days: milestone),
                    triggeredAt: Date(),
                    metadata: ["streakDays": "\(milestone)"]
                )
                
                triggerReaction(reaction)
                
                // 絆ポイント付与
                achievementSystem.awardBondPoints(
                    milestone * 5,
                    to: characterName,
                    reason: "\(milestone)日連続ストリーク"
                )
                break
            }
        }
    }
    
    func checkMilestone(session: SessionSummary) {
        guard canTriggerReaction(.milestone) else { return }
        
        let lifetimeReps = calculateLifetimeReps()
        let milestones = [100, 500, 1000, 2500, 5000]
        
        for milestone in milestones.reversed() {
            if lifetimeReps >= milestone && !hasMilestoneBeenCelebrated(milestone, session.characterName) {
                let reaction = OshiReaction(
                    type: .milestone,
                    characterName: session.characterName,
                    message: generateMilestoneMessage(reps: milestone),
                    audioFileName: getMilestoneAudioFile(session.characterName, reps: milestone),
                    triggeredAt: Date(),
                    metadata: ["milestone": "\(milestone)"]
                )
                
                triggerReaction(reaction)
                markMilestoneCelebrated(milestone, session.characterName)
                
                // 絆ポイント付与
                achievementSystem.awardBondPoints(
                    milestone / 10,
                    to: session.characterName,
                    reason: "累計\(milestone)レップ達成"
                )
                break
            }
        }
    }
    
    func triggerEncouragement() {
        guard canTriggerReaction(.encouragement) else { return }
        
        let daysSinceLastSession = daysSinceLastTraining()
        let characterName = getCurrentCharacter()
        
        if daysSinceLastSession >= 3 {
            let reaction = OshiReaction(
                type: .encouragement,
                characterName: characterName,
                message: generateEncouragementMessage(daysSinceLastSession),
                audioFileName: getEncouragementAudioFile(characterName),
                triggeredAt: Date(),
                metadata: ["daysSinceLastSession": "\(daysSinceLastSession)"]
            )
            
            triggerReaction(reaction)
            
            // 復帰ボーナス
            achievementSystem.awardBondPoints(
                10,
                to: characterName,
                reason: "トレーニング復帰"
            )
        }
    }
    
    func triggerLevelUpReaction(characterName: String, newLevel: Int, oldLevel: Int) {
        let reaction = OshiReaction(
            type: .levelUp,
            characterName: characterName,
            message: generateLevelUpMessage(newLevel: newLevel),
            audioFileName: getLevelUpAudioFile(characterName, level: newLevel),
            triggeredAt: Date(),
            metadata: ["newLevel": "\(newLevel)", "oldLevel": "\(oldLevel)"]
        )
        
        triggerReaction(reaction)
    }
    
    // MARK: - Reaction Processing
    private func triggerReaction(_ reaction: OshiReaction) {
        // 反応をリストに追加
        recentReactions.append(reaction)
        
        // 最新10個のみ保持
        if recentReactions.count > 10 {
            recentReactions.removeFirst(recentReactions.count - 10)
        }
        
        // クールダウン設定
        setCooldown(for: reaction.type)
        
        // 音声再生
        playReactionAudio(reaction)
        
        print("💕 推し反応: \(reaction.characterName) - \(reaction.message)")
    }
    
    private func playReactionAudio(_ reaction: OshiReaction) {
        if let audioFileName = reaction.audioFileName {
            // 既存のAudioFeedbackServiceを活用
            audioFeedbackService.playCustomAudio(fileName: audioFileName)
        }
    }
    
    // MARK: - Cooldown Management
    private func canTriggerReaction(_ type: ReactionType) -> Bool {
        let key = type.rawValue
        
        guard let lastTrigger = reactionCooldowns[key],
              let cooldownDuration = cooldownDurations[type] else {
            return true
        }
        
        return Date().timeIntervalSince(lastTrigger) >= cooldownDuration
    }
    
    private func setCooldown(for type: ReactionType) {
        reactionCooldowns[type.rawValue] = Date()
    }
    
    // MARK: - Message Generation
    private func generateNewRecordMessage(_ record: PersonalRecord) -> String {
        let messages = [
            "すごい！\(record.recordType.displayName)で新記録だよ！",
            "やったね！\(record.value)回の新記録達成！",
            "新記録おめでとう！どんどん強くなってるね！",
            "すばらしい！君の努力が実を結んだよ！"
        ]
        return messages.randomElement() ?? "新記録達成！"
    }
    
    private func generateStreakMessage(days: Int) -> String {
        switch days {
        case 3:
            return "3日連続！いい調子だよ！"
        case 7:
            return "1週間連続！素晴らしい継続力だね！"
        case 14:
            return "2週間連続！もう習慣になってるね！"
        case 30:
            return "1ヶ月連続！本当に尊敬するよ！"
        default:
            return "\(days)日連続！すごい継続力だ！"
        }
    }
    
    private func generateMilestoneMessage(reps: Int) -> String {
        switch reps {
        case 100:
            return "100回達成！最初の大台だよ！"
        case 500:
            return "500回達成！本格的になってきたね！"
        case 1000:
            return "1000回達成！もう立派な筋トレマスターだ！"
        case 2500:
            return "2500回達成！驚異的な数字だよ！"
        case 5000:
            return "5000回達成！伝説級の努力だね！"
        default:
            return "\(reps)回達成！すごい記録だよ！"
        }
    }
    
    private func generateEncouragementMessage(_ daysSince: Int) -> String {
        switch daysSince {
        case 3...5:
            return "久しぶり！また一緒にがんばろう！"
        case 6...10:
            return "待ってたよ！今日から再スタートだね！"
        default:
            return "おかえり！また会えて嬉しいよ！"
        }
    }
    
    private func generateLevelUpMessage(newLevel: Int) -> String {
        switch newLevel {
        case 1...5:
            return "レベル\(newLevel)になったよ！一緒に成長してるね！"
        case 6...10:
            return "レベル\(newLevel)達成！絆が深まってきたね！"
        case 11...20:
            return "レベル\(newLevel)！もうベストパートナーだよ！"
        default:
            return "レベル\(newLevel)！君との絆は無限大だ！"
        }
    }
    
    // MARK: - Audio File Selection
    private func getNewRecordAudioFile(_ characterName: String) -> String? {
        switch characterName {
        case "ずんだもん":
            return "zundamon_new_record.wav"
        case "四国めたん":
            return "shikoku_new_record.wav"
        default:
            return nil
        }
    }
    
    private func getStreakAudioFile(_ characterName: String, days: Int) -> String? {
        switch characterName {
        case "ずんだもん":
            return "zundamon_streak_\(days)days.wav"
        case "四国めたん":
            return "shikoku_streak_\(days)days.wav"
        default:
            return nil
        }
    }
    
    private func getMilestoneAudioFile(_ characterName: String, reps: Int) -> String? {
        switch characterName {
        case "ずんだもん":
            return "zundamon_milestone_\(reps).wav"
        case "四国めたん":
            return "shikoku_milestone_\(reps).wav"
        default:
            return nil
        }
    }
    
    private func getEncouragementAudioFile(_ characterName: String) -> String? {
        switch characterName {
        case "ずんだもん":
            return "zundamon_encouragement.wav"
        case "四国めたん":
            return "shikoku_encouragement.wav"
        default:
            return nil
        }
    }
    
    private func getLevelUpAudioFile(_ characterName: String, level: Int) -> String? {
        switch characterName {
        case "ずんだもん":
            return "zundamon_levelup_\(level).wav"
        case "四国めたん":
            return "shikoku_levelup_\(level).wav"
        default:
            return nil
        }
    }
    
    // MARK: - Helper Methods
    private func calculateCurrentStreak() -> Int {
        let calendar = Calendar.current
        let sessions = trainingSessionService.fetchSessionsForDateRange(
            from: calendar.date(byAdding: .month, value: -2, to: Date()) ?? Date(),
            to: Date()
        )
        
        guard !sessions.isEmpty else { return 0 }
        
        let sessionDates = Set(sessions.compactMap { session -> Date? in
            guard let startTime = session.startTime else { return nil }
            return calendar.startOfDay(for: startTime)
        })
        
        var streak = 0
        var checkDate = calendar.startOfDay(for: Date())
        
        while sessionDates.contains(checkDate) {
            streak += 1
            checkDate = calendar.date(byAdding: .day, value: -1, to: checkDate) ?? checkDate
        }
        
        return streak
    }
    
    private func calculateLifetimeReps() -> Int {
        let sessions = trainingSessionService.fetchSessionsForDateRange(
            from: Date.distantPast,
            to: Date()
        )
        return sessions.reduce(0) { $0 + Int($1.totalReps) }
    }
    
    private func daysSinceLastTraining() -> Int {
        let sessions = trainingSessionService.fetchSessionsForDateRange(
            from: Calendar.current.date(byAdding: .month, value: -1, to: Date()) ?? Date(),
            to: Date()
        )
        
        guard let lastSession = sessions.first,
              let lastDate = lastSession.startTime else { return 999 }
        
        let calendar = Calendar.current
        return calendar.dateComponents([.day], from: lastDate, to: Date()).day ?? 999
    }
    
    private func getCurrentCharacter() -> String {
        return UserDefaults.standard.string(forKey: "selectedCharacter") ?? "ずんだもん"
    }
    
    private func calculateNewRecordBondPoints(_ record: PersonalRecord) -> Int {
        switch record.recordType {
        case .maxReps:
            return min(100, record.value * 2)
        case .perfectForm:
            return 50
        case .longestStreak:
            return record.value * 5
        case .totalLifetime:
            return record.value / 100
        }
    }
    
    // MARK: - Milestone Tracking
    private func hasMilestoneBeenCelebrated(_ milestone: Int, _ characterName: String) -> Bool {
        let key = "milestone_\(milestone)_\(characterName)"
        return UserDefaults.standard.bool(forKey: key)
    }
    
    private func markMilestoneCelebrated(_ milestone: Int, _ characterName: String) {
        let key = "milestone_\(milestone)_\(characterName)"
        UserDefaults.standard.set(true, forKey: key)
    }
    
    // MARK: - Subscription Setup
    private func setupReactionSubscriptions() {
        // セッション完了時の自動反応チェック
        trainingSessionService.$currentSession
            .compactMap { $0 }
            .sink { _ in
                // セッション開始時の処理（現在は未実装）
            }
            .store(in: &cancellables)
    }
}

// MARK: - Data Models
struct OshiReaction: Identifiable {
    let id = UUID()
    let type: ReactionType
    let characterName: String
    let message: String
    let audioFileName: String?
    let triggeredAt: Date
    let metadata: [String: String]
}

enum ReactionType: String, CaseIterable {
    case newRecord = "new_record"
    case streak = "streak"
    case milestone = "milestone"
    case encouragement = "encouragement"
    case levelUp = "level_up"
    
    var displayName: String {
        switch self {
        case .newRecord: return "新記録"
        case .streak: return "連続記録"
        case .milestone: return "マイルストーン"
        case .encouragement: return "応援"
        case .levelUp: return "レベルアップ"
        }
    }
    
    var iconName: String {
        switch self {
        case .newRecord: return "trophy.fill"
        case .streak: return "flame.fill"
        case .milestone: return "star.fill"
        case .encouragement: return "heart.fill"
        case .levelUp: return "arrow.up.circle.fill"
        }
    }
}

// MARK: - AudioFeedbackService Extension
extension AudioFeedbackService {
    func playCustomAudio(fileName: String) {
        // カスタム音声ファイル再生の実装
        // 既存のplayAudio methodを拡張
        guard let url = Bundle.main.url(forResource: fileName, withExtension: nil) else {
            print("⚠️ 音声ファイルが見つかりません: \(fileName)")
            return
        }
        
        do {
            let audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer.play()
            print("🔊 再生中: \(fileName)")
        } catch {
            print("⚠️ 音声再生エラー: \(error)")
        }
    }
}