import Foundation

/// エクササイズの種目を表す列挙型
enum ExerciseType: String, CaseIterable, Codable, Identifiable {
    case overheadPress = "overhead_press"
    case squat = "squat"
    case plank = "plank"
    case pushUp = "push_up"
    case lunge = "lunge"
    case burpee = "burpee"
    
    /// Identifiable準拠のためのID
    var id: String { rawValue }
    
    /// 表示用の日本語名
    var displayName: String {
        switch self {
        case .overheadPress:
            return "オーバーヘッドプレス"
        case .squat:
            return "スクワット"
        case .plank:
            return "プランク"
        case .pushUp:
            return "腕立て伏せ"
        case .lunge:
            return "ランジ"
        case .burpee:
            return "バーピー"
        }
    }
    
    /// エクササイズの説明
    var description: String {
        switch self {
        case .overheadPress:
            return "肩と腕を鍛える基本的なウェイトトレーニング"
        case .squat:
            return "下半身全体を鍛える王道トレーニング"
        case .plank:
            return "体幹を鍛える静的トレーニング"
        case .pushUp:
            return "胸と腕を鍛える自重トレーニング"
        case .lunge:
            return "下半身とバランス感覚を鍛える"
        case .burpee:
            return "全身を使った高強度トレーニング"
        }
    }
    
    /// 難易度（1-5の5段階）
    var difficulty: Int {
        switch self {
        case .overheadPress:
            return 2
        case .squat:
            return 2
        case .plank:
            return 1
        case .pushUp:
            return 2
        case .lunge:
            return 3
        case .burpee:
            return 5
        }
    }
    
    /// 推定カロリー消費量（10分あたり）
    var estimatedCalories: Int {
        switch self {
        case .overheadPress:
            return 60
        case .squat:
            return 80
        case .plank:
            return 30
        case .pushUp:
            return 70
        case .lunge:
            return 75
        case .burpee:
            return 120
        }
    }
    
    /// 現在利用可能かどうか
    var isAvailable: Bool {
        switch self {
        case .overheadPress:
            return true
        default:
            return false // 将来実装予定
        }
    }
    
    /// SF Symbols のアイコン名
    var iconSystemName: String {
        switch self {
        case .overheadPress:
            return "figure.strengthtraining.traditional"
        case .squat:
            return "figure.strengthtraining.functional"
        case .plank:
            return "figure.core.training"
        case .pushUp:
            return "figure.arms.open"
        case .lunge:
            return "figure.walk"
        case .burpee:
            return "figure.jumprope"
        }
    }
    
    /// 利用できない場合の表示ラベル
    var comingSoonLabel: String? {
        return isAvailable ? nil : "近日公開"
    }
    
    /// 難易度を星で表示
    var difficultyStars: String {
        return String(repeating: "★", count: difficulty) + String(repeating: "☆", count: 5 - difficulty)
    }
}

// MARK: - 便利なメソッド
extension ExerciseType {
    /// 利用可能な種目のみを取得
    static var availableExercises: [ExerciseType] {
        return ExerciseType.allCases.filter { $0.isAvailable }
    }
    
    /// 利用不可能な種目のみを取得
    static var comingSoonExercises: [ExerciseType] {
        return ExerciseType.allCases.filter { !$0.isAvailable }
    }
}