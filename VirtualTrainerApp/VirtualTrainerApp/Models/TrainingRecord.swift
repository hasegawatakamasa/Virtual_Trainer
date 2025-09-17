import Foundation

// MARK: - Training Record Types
struct PersonalRecord: Codable, Identifiable {
    let id: UUID
    let recordType: RecordType
    let value: Int
    let exerciseType: String
    let characterName: String
    let achievedAt: Date
    let previousBest: Int?
    
    init(recordType: RecordType, value: Int, exerciseType: String, characterName: String, achievedAt: Date, previousBest: Int?) {
        self.id = UUID()
        self.recordType = recordType
        self.value = value
        self.exerciseType = exerciseType
        self.characterName = characterName
        self.achievedAt = achievedAt
        self.previousBest = previousBest
    }
    
    var isNewRecord: Bool {
        guard let previous = previousBest else { return true }
        return value > previous
    }
    
    var improvement: Int? {
        guard let previous = previousBest else { return nil }
        return value - previous
    }
}

enum RecordType: String, CaseIterable, Codable {
    case maxReps = "max_reps"
    case longestStreak = "longest_streak"
    case perfectForm = "perfect_form"
    case totalLifetime = "total_lifetime"
    
    var displayName: String {
        switch self {
        case .maxReps:
            return "最大レップ数"
        case .longestStreak:
            return "最長連続日数"
        case .perfectForm:
            return "完璧フォーム達成"
        case .totalLifetime:
            return "生涯累計レップ"
        }
    }
    
    var iconName: String {
        switch self {
        case .maxReps:
            return "trophy.fill"
        case .longestStreak:
            return "flame.fill"
        case .perfectForm:
            return "checkmark.seal.fill"
        case .totalLifetime:
            return "infinity.circle.fill"
        }
    }
}

// MARK: - Oshi Bond Level System
struct OshiBondLevel: Codable, Identifiable {
    let id: UUID
    let characterName: String
    var currentLevel: Int
    var bondPoints: Int
    var totalBondPoints: Int
    let lastUpdated: Date
    
    init(characterName: String, currentLevel: Int = 0, bondPoints: Int = 0, totalBondPoints: Int = 0, lastUpdated: Date = Date()) {
        self.id = UUID()
        self.characterName = characterName
        self.currentLevel = currentLevel
        self.bondPoints = bondPoints
        self.totalBondPoints = totalBondPoints
        self.lastUpdated = lastUpdated
    }
    
    var nextLevelPoints: Int {
        return calculatePointsForLevel(currentLevel + 1)
    }
    
    var currentLevelMinPoints: Int {
        return calculatePointsForLevel(currentLevel)
    }
    
    var progressToNextLevel: Double {
        let currentPoints = bondPoints - currentLevelMinPoints
        let pointsNeeded = nextLevelPoints - currentLevelMinPoints
        return pointsNeeded > 0 ? Double(currentPoints) / Double(pointsNeeded) : 1.0
    }
    
    private func calculatePointsForLevel(_ level: Int) -> Int {
        // レベルに応じた必要ポイント計算（指数的増加）
        switch level {
        case 0...5:
            return level * 100
        case 6...10:
            return 500 + (level - 5) * 200
        case 11...20:
            return 1500 + (level - 10) * 300
        default:
            return 4500 + (level - 20) * 500
        }
    }
    
    mutating func addBondPoints(_ points: Int) {
        bondPoints += points
        totalBondPoints += points
        checkLevelUp()
    }
    
    private mutating func checkLevelUp() {
        while bondPoints >= nextLevelPoints {
            currentLevel += 1
        }
    }
}

// MARK: - Share Template
struct ShareTemplate: Codable, Identifiable {
    let id: UUID
    let templateType: ShareTemplateType
    let backgroundImageName: String
    let characterImageName: String
    let textStyle: ShareTextStyle
    let layout: ShareLayout
    
    init(templateType: ShareTemplateType, backgroundImageName: String, characterImageName: String, textStyle: ShareTextStyle, layout: ShareLayout) {
        self.id = UUID()
        self.templateType = templateType
        self.backgroundImageName = backgroundImageName
        self.characterImageName = characterImageName
        self.textStyle = textStyle
        self.layout = layout
    }
}

enum ShareTemplateType: String, CaseIterable, Codable {
    case newRecord = "new_record"
    case dailyCompletion = "daily_completion"
    case weeklyMilestone = "weekly_milestone"
    case levelUp = "level_up"
    
    var displayName: String {
        switch self {
        case .newRecord:
            return "新記録達成"
        case .dailyCompletion:
            return "今日のトレーニング完了"
        case .weeklyMilestone:
            return "週間目標達成"
        case .levelUp:
            return "レベルアップ"
        }
    }
}

struct ShareTextStyle: Codable {
    let fontName: String
    let fontSize: CGFloat
    let textColor: String // Hex color
    let shadowColor: String
    let shadowOffset: CGSize
}

struct ShareLayout: Codable {
    let characterPosition: CGPoint
    let characterSize: CGSize
    let titlePosition: CGPoint
    let subtitlePosition: CGPoint
    let statsPosition: CGPoint
}

// MARK: - Training Session Extensions
extension TrainingSession {
    func toSessionSummary() -> SessionSummary {
        let repsArray = (reps?.allObjects as? [SessionRep]) ?? []
        
        return SessionSummary(
            id: id ?? UUID(),
            exerciseType: exerciseType ?? "",
            characterName: characterName ?? "",
            startTime: startTime ?? Date(),
            endTime: endTime,
            totalReps: Int(totalReps),
            formErrors: Int(formErrors),
            speedWarnings: Int(speedWarnings),
            sessionDuration: sessionDuration,
            reps: repsArray.map { $0.toRepSummary() }
        )
    }
}

extension SessionRep {
    func toRepSummary() -> RepSummary {
        return RepSummary(
            id: id ?? UUID(),
            timestamp: timestamp ?? Date(),
            formQuality: formQuality ?? "Unknown",
            keypointConfidence: keypointConfidence
        )
    }
}

// MARK: - Session Summary Models
struct SessionSummary: Identifiable, Codable {
    let id: UUID
    let exerciseType: String
    let characterName: String
    let startTime: Date
    let endTime: Date?
    let totalReps: Int
    let formErrors: Int
    let speedWarnings: Int
    let sessionDuration: Double
    let reps: [RepSummary]
    
    var formAccuracy: Double {
        guard totalReps > 0 else { return 0.0 }
        return Double(totalReps - formErrors) / Double(totalReps)
    }
    
    var averageRepDuration: Double {
        guard totalReps > 0, sessionDuration > 0 else { return 0.0 }
        return sessionDuration / Double(totalReps)
    }
    
    var completionRate: Double {
        // Calculate based on expected reps vs actual reps
        // For now, assume target is 10 reps per set
        let targetReps = 10
        return min(1.0, Double(totalReps) / Double(targetReps))
    }
}

struct RepSummary: Identifiable, Codable {
    let id: UUID
    let timestamp: Date
    let formQuality: String
    let keypointConfidence: Double
    
    var isGoodForm: Bool {
        return formQuality == "Normal" && keypointConfidence > 0.8
    }
}

// MARK: - UserDefaults Extensions for Persistence
extension UserDefaults {
    private enum Keys {
        static let personalRecords = "personal_records"
        static let oshiBondLevels = "oshi_bond_levels"
        static let shareTemplates = "share_templates"
    }
    
    // Personal Records
    func setPersonalRecords(_ records: [PersonalRecord]) {
        if let encoded = try? JSONEncoder().encode(records) {
            set(encoded, forKey: Keys.personalRecords)
        }
    }
    
    func getPersonalRecords() -> [PersonalRecord] {
        guard let data = data(forKey: Keys.personalRecords),
              let records = try? JSONDecoder().decode([PersonalRecord].self, from: data) else {
            return []
        }
        return records
    }
    
    // Oshi Bond Levels
    func setOshiBondLevels(_ levels: [OshiBondLevel]) {
        if let encoded = try? JSONEncoder().encode(levels) {
            set(encoded, forKey: Keys.oshiBondLevels)
        }
    }
    
    func getOshiBondLevels() -> [OshiBondLevel] {
        guard let data = data(forKey: Keys.oshiBondLevels),
              let levels = try? JSONDecoder().decode([OshiBondLevel].self, from: data) else {
            return []
        }
        return levels
    }
    
    func getOshiBondLevel(for characterName: String) -> OshiBondLevel? {
        return getOshiBondLevels().first { $0.characterName == characterName }
    }
    
    func updateOshiBondLevel(_ level: OshiBondLevel) {
        var levels = getOshiBondLevels()
        if let index = levels.firstIndex(where: { $0.characterName == level.characterName }) {
            levels[index] = level
        } else {
            levels.append(level)
        }
        setOshiBondLevels(levels)
    }
    
    // Share Templates
    func setShareTemplates(_ templates: [ShareTemplate]) {
        if let encoded = try? JSONEncoder().encode(templates) {
            set(encoded, forKey: Keys.shareTemplates)
        }
    }
    
    func getShareTemplates() -> [ShareTemplate] {
        guard let data = data(forKey: Keys.shareTemplates),
              let templates = try? JSONDecoder().decode([ShareTemplate].self, from: data) else {
            return defaultShareTemplates()
        }
        return templates
    }
    
    private func defaultShareTemplates() -> [ShareTemplate] {
        return [
            ShareTemplate(
                templateType: .newRecord,
                backgroundImageName: "new_record_bg",
                characterImageName: "character_celebration",
                textStyle: ShareTextStyle(
                    fontName: "HiraginoSans-W6",
                    fontSize: 24,
                    textColor: "#FFFFFF",
                    shadowColor: "#000000",
                    shadowOffset: CGSize(width: 2, height: 2)
                ),
                layout: ShareLayout(
                    characterPosition: CGPoint(x: 100, y: 200),
                    characterSize: CGSize(width: 150, height: 150),
                    titlePosition: CGPoint(x: 200, y: 100),
                    subtitlePosition: CGPoint(x: 200, y: 150),
                    statsPosition: CGPoint(x: 200, y: 300)
                )
            )
        ]
    }
}

// MARK: - Integration with Existing Models
extension ExerciseSession {
    func toTrainingSessionData() -> (exerciseType: String, characterName: String) {
        return (
            exerciseType: self.exerciseType.displayName,
            characterName: "ずんだもん" // Default character, could be made configurable
        )
    }
}