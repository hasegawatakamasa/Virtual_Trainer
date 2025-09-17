import Foundation
import Combine
import CoreData

class AchievementSystem: ObservableObject {
    static let shared = AchievementSystem()
    
    @Published var achievements: [AchievementDefinition] = []
    @Published var unlockedAchievements: [Achievement] = []
    @Published var oshiBondLevels: [OshiBondLevel] = []
    @Published var recentUnlocks: [Achievement] = []
    
    private let coreDataManager = CoreDataManager.shared
    private let userDefaults = UserDefaults.standard
    private var cancellables = Set<AnyCancellable>()
    
    private init() {
        loadAchievementDefinitions()
        loadOshiBondLevels()
        loadUnlockedAchievements()
    }
    
    // MARK: - Achievement Evaluation
    func evaluateAchievements(for session: SessionSummary) {
        var newAchievements: [Achievement] = []
        
        // 連続日数アチーブメント
        if let streakAchievement = evaluateStreakAchievements() {
            newAchievements.append(streakAchievement)
        }
        
        // 累計レップアチーブメント
        if let repsAchievement = evaluateLifetimeRepsAchievements() {
            newAchievements.append(repsAchievement)
        }
        
        // 完璧フォーム達成アチーブメント
        if let formAchievement = evaluatePerfectFormAchievements(for: session) {
            newAchievements.append(formAchievement)
        }
        
        // セッション特化アチーブメント
        if let sessionAchievement = evaluateSessionAchievements(for: session) {
            newAchievements.append(sessionAchievement)
        }
        
        // 新しいアチーブメントがあれば処理
        for achievement in newAchievements {
            unlockAchievement(achievement)
        }
    }
    
    private func evaluateStreakAchievements() -> Achievement? {
        let streakDays = calculateCurrentStreak()
        let characterName = getCurrentCharacter()
        
        let streakMilestones = [3, 7, 14, 30, 60, 100]
        
        for milestone in streakMilestones.reversed() {
            if streakDays >= milestone {
                let achievementType = "streak_\(milestone)_days"
                
                // 既に獲得済みかチェック
                if !isAchievementUnlocked(type: achievementType, characterName: characterName) {
                    return Achievement(
                        context: coreDataManager.viewContext,
                        type: achievementType,
                        characterName: characterName,
                        bondPointsAwarded: calculateStreakBondPoints(milestone),
                        unlockedAt: Date()
                    )
                }
                break
            }
        }
        
        return nil
    }
    
    private func evaluateLifetimeRepsAchievements() -> Achievement? {
        let lifetimeReps = calculateLifetimeReps()
        let characterName = getCurrentCharacter()
        
        let repMilestones = [100, 500, 1000, 2500, 5000, 10000]
        
        for milestone in repMilestones.reversed() {
            if lifetimeReps >= milestone {
                let achievementType = "lifetime_reps_\(milestone)"
                
                if !isAchievementUnlocked(type: achievementType, characterName: characterName) {
                    return Achievement(
                        context: coreDataManager.viewContext,
                        type: achievementType,
                        characterName: characterName,
                        bondPointsAwarded: calculateRepsBondPoints(milestone),
                        unlockedAt: Date()
                    )
                }
                break
            }
        }
        
        return nil
    }
    
    private func evaluatePerfectFormAchievements(for session: SessionSummary) -> Achievement? {
        let characterName = session.characterName
        
        // 完璧フォーム条件: フォーム正確率 >= 95% かつ レップ数 >= 10
        if session.formAccuracy >= 0.95 && session.totalReps >= 10 {
            let achievementType = "perfect_form_session"
            
            // 同日の重複を防ぐため、日付チェック
            let today = Calendar.current.startOfDay(for: Date())
            let existingToday = unlockedAchievements.filter { 
                $0.type == achievementType && 
                $0.characterName == characterName &&
                Calendar.current.startOfDay(for: $0.unlockedAt ?? Date()) == today
            }
            
            if existingToday.isEmpty {
                return Achievement(
                    context: coreDataManager.viewContext,
                    type: achievementType,
                    characterName: characterName,
                    bondPointsAwarded: 50,
                    unlockedAt: Date()
                )
            }
        }
        
        return nil
    }
    
    private func evaluateSessionAchievements(for session: SessionSummary) -> Achievement? {
        let characterName = session.characterName
        
        // 長時間セッション: 20分以上
        if session.sessionDuration >= 1200 { // 20 minutes
            let achievementType = "long_session"
            
            if !isAchievementUnlocked(type: achievementType, characterName: characterName) {
                return Achievement(
                    context: coreDataManager.viewContext,
                    type: achievementType,
                    characterName: characterName,
                    bondPointsAwarded: 30,
                    unlockedAt: Date()
                )
            }
        }
        
        // 高強度セッション: 50レップ以上
        if session.totalReps >= 50 {
            let achievementType = "high_intensity_session"
            
            if !isAchievementUnlocked(type: achievementType, characterName: characterName) {
                return Achievement(
                    context: coreDataManager.viewContext,
                    type: achievementType,
                    characterName: characterName,
                    bondPointsAwarded: 40,
                    unlockedAt: Date()
                )
            }
        }
        
        return nil
    }
    
    // MARK: - Bond Points System
    func awardBondPoints(_ points: Int, to characterName: String, reason: String, fromAchievement: Bool = false) {
        var bondLevel = getOshiBondLevel(for: characterName)
        bondLevel.addBondPoints(points)
        
        // レベルアップチェック（アチーブメントからの呼び出しの場合はスキップ）
        if !fromAchievement {
            let oldLevel = oshiBondLevels.first { $0.characterName == characterName }?.currentLevel ?? 0
            if bondLevel.currentLevel > oldLevel {
                let levelUpType = "level_up_\(bondLevel.currentLevel)"
                
                // 既に同じレベルアップアチーブメントが存在しないかチェック
                if !isAchievementUnlocked(type: levelUpType, characterName: characterName) {
                    // レベルアップアチーブメント
                    let levelUpAchievement = Achievement(
                        context: coreDataManager.viewContext,
                        type: levelUpType,
                        characterName: characterName,
                        bondPointsAwarded: Int32(bondLevel.currentLevel * 10),
                        unlockedAt: Date()
                    )
                    unlockAchievement(levelUpAchievement)
                    
                    // コンテンツアンロック
                    unlockContent(for: characterName, level: bondLevel.currentLevel)
                }
            }
        }
        
        updateOshiBondLevel(bondLevel)
        print("💎 \(characterName) に \(points) ポイント付与: \(reason)")
    }
    
    private func unlockContent(for characterName: String, level: Int) {
        // レベルに応じた新規コンテンツのアンロック
        let unlockedContent = getUnlockedContentForLevel(level)
        
        for content in unlockedContent {
            print("🎉 \(characterName) レベル\(level)で \(content) がアンロックされました！")
        }
    }
    
    private func getUnlockedContentForLevel(_ level: Int) -> [String] {
        switch level {
        case 1:
            return ["基本応援音声"]
        case 3:
            return ["特別励ましボイス"]
        case 5:
            return ["新記録祝福ボイス"]
        case 10:
            return ["専用BGM", "特別エフェクト"]
        case 15:
            return ["プレミアムボイスパック"]
        case 20:
            return ["限定キャラクターコスチューム"]
        default:
            return level % 5 == 0 ? ["ボーナスボイス\(level)"] : []
        }
    }
    
    // MARK: - Achievement Management
    private func unlockAchievement(_ achievement: Achievement) {
        Task { @MainActor in
            // Core Dataに保存
            let savedAchievement = coreDataManager.createAchievement(
                type: achievement.type ?? "",
                characterName: achievement.characterName ?? "",
                bondPointsAwarded: achievement.bondPointsAwarded,
                unlockedAt: achievement.unlockedAt ?? Date()
            )
            
            // ローカル配列に追加（保存されたオブジェクトを使用）
            unlockedAchievements.append(savedAchievement)
            recentUnlocks.append(savedAchievement)
            
            // 絆ポイント付与（アチーブメントからの呼び出しフラグを立てる）
            awardBondPoints(
                Int(achievement.bondPointsAwarded), 
                to: achievement.characterName ?? "", 
                reason: getAchievementDisplayName(achievement.type ?? ""),
                fromAchievement: true
            )
            
            print("🏆 アチーブメント獲得: \(getAchievementDisplayName(achievement.type ?? ""))")
            
            // 最新の解除のみ保持（UI表示用）
            if recentUnlocks.count > 5 {
                recentUnlocks.removeFirst(recentUnlocks.count - 5)
            }
        }
    }
    
    private func isAchievementUnlocked(type: String, characterName: String) -> Bool {
        return unlockedAchievements.contains { 
            $0.type == type && $0.characterName == characterName 
        }
    }
    
    // MARK: - Progress Calculation
    func getAchievementProgress() -> [AchievementProgress] {
        var progressList: [AchievementProgress] = []
        let characterName = getCurrentCharacter()
        
        // 連続日数進捗
        let currentStreak = calculateCurrentStreak()
        let streakMilestones = [3, 7, 14, 30, 60, 100]
        
        for milestone in streakMilestones {
            let isUnlocked = isAchievementUnlocked(type: "streak_\(milestone)_days", characterName: characterName)
            let progress = min(1.0, Double(currentStreak) / Double(milestone))
            
            progressList.append(AchievementProgress(
                type: "streak_\(milestone)_days",
                title: "\(milestone)日連続",
                description: "連続でトレーニングを実行",
                progress: progress,
                isUnlocked: isUnlocked,
                iconName: "flame.fill"
            ))
        }
        
        // 累計レップ進捗
        let lifetimeReps = calculateLifetimeReps()
        let repsMilestones = [100, 500, 1000, 2500, 5000, 10000]
        
        for milestone in repsMilestones {
            let isUnlocked = isAchievementUnlocked(type: "lifetime_reps_\(milestone)", characterName: characterName)
            let progress = min(1.0, Double(lifetimeReps) / Double(milestone))
            
            progressList.append(AchievementProgress(
                type: "lifetime_reps_\(milestone)",
                title: "累計\(milestone)レップ",
                description: "生涯通算レップ数達成",
                progress: progress,
                isUnlocked: isUnlocked,
                iconName: "number.circle.fill"
            ))
        }
        
        return progressList.sorted { $0.progress > $1.progress }
    }
    
    // MARK: - Helper Methods
    private func calculateCurrentStreak() -> Int {
        let calendar = Calendar.current
        let sessions = coreDataManager.fetchTrainingSessions()
        
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
        let sessions = coreDataManager.fetchTrainingSessions()
        return sessions.reduce(0) { $0 + Int($1.totalReps) }
    }
    
    private func getCurrentCharacter() -> String {
        return userDefaults.string(forKey: "selectedCharacter") ?? "ずんだもん"
    }
    
    private func getOshiBondLevel(for characterName: String) -> OshiBondLevel {
        if let existing = oshiBondLevels.first(where: { $0.characterName == characterName }) {
            return existing
        } else {
            let newLevel = OshiBondLevel(characterName: characterName)
            oshiBondLevels.append(newLevel)
            return newLevel
        }
    }
    
    private func updateOshiBondLevel(_ level: OshiBondLevel) {
        if let index = oshiBondLevels.firstIndex(where: { $0.characterName == level.characterName }) {
            oshiBondLevels[index] = level
        }
        userDefaults.setOshiBondLevels(oshiBondLevels)
    }
    
    // MARK: - Bond Points Calculation
    private func calculateStreakBondPoints(_ days: Int) -> Int32 {
        switch days {
        case 3: return 20
        case 7: return 50
        case 14: return 100
        case 30: return 200
        case 60: return 400
        case 100: return 800
        default: return Int32(days * 2)
        }
    }
    
    private func calculateRepsBondPoints(_ reps: Int) -> Int32 {
        switch reps {
        case 100: return 30
        case 500: return 100
        case 1000: return 250
        case 2500: return 500
        case 5000: return 1000
        case 10000: return 2000
        default: return Int32(reps / 10)
        }
    }
    
    // MARK: - Data Loading
    private func loadAchievementDefinitions() {
        // アチーブメント定義をロード（今後JSONファイルから読み込み可能）
        achievements = createDefaultAchievements()
    }
    
    private func loadOshiBondLevels() {
        oshiBondLevels = userDefaults.getOshiBondLevels()
    }
    
    private func loadUnlockedAchievements() {
        unlockedAchievements = coreDataManager.fetchAchievements()
    }
    
    private func createDefaultAchievements() -> [AchievementDefinition] {
        return [
            AchievementDefinition(type: "streak_3_days", title: "3日連続", description: "3日連続でトレーニング", iconName: "flame.fill"),
            AchievementDefinition(type: "streak_7_days", title: "1週間連続", description: "7日連続でトレーニング", iconName: "flame.fill"),
            AchievementDefinition(type: "lifetime_reps_100", title: "最初の100", description: "累計100レップ達成", iconName: "number.circle.fill"),
            AchievementDefinition(type: "lifetime_reps_1000", title: "1000の大台", description: "累計1000レップ達成", iconName: "number.circle.fill"),
            AchievementDefinition(type: "perfect_form_session", title: "完璧フォーム", description: "95%以上の正確率", iconName: "checkmark.seal.fill")
        ]
    }
    
    private func getAchievementDisplayName(_ type: String) -> String {
        return achievements.first { $0.type == type }?.title ?? type
    }
}

// MARK: - Achievement Definition
struct AchievementDefinition: Identifiable {
    let id = UUID()
    let type: String
    let title: String
    let description: String
    let iconName: String
}

// MARK: - Achievement Progress
struct AchievementProgress: Identifiable {
    let id = UUID()
    let type: String
    let title: String
    let description: String
    let progress: Double
    let isUnlocked: Bool
    let iconName: String
}

// MARK: - Achievement Extension
extension Achievement {
    convenience init(context: NSManagedObjectContext, type: String, characterName: String, bondPointsAwarded: Int32, unlockedAt: Date) {
        self.init(context: context)
        self.id = UUID()
        self.type = type
        self.characterName = characterName
        self.bondPointsAwarded = bondPointsAwarded
        self.unlockedAt = unlockedAt
    }
}