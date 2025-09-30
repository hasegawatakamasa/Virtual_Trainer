import Foundation

/// エクササイズの種目を表す列挙型
enum ExerciseType: String, CaseIterable, Codable, Identifiable {
    case overheadPress = "overhead_press"
    case sideRaise = "side_raise"
    case squat = "squat"
    case pushUp = "push_up"
    
    /// Identifiable準拠のためのID
    var id: String { rawValue }
    
    /// 表示用の日本語名
    var displayName: String {
        switch self {
        case .overheadPress:
            return "オーバーヘッドプレス"
        case .sideRaise:
            return "サイドレイズ"
        case .squat:
            return "スクワット"
        case .pushUp:
            return "腕立て伏せ"
        }
    }
    
    /// エクササイズの説明
    var description: String {
        switch self {
        case .overheadPress:
            return "肩と腕を鍛える基本的なウェイトトレーニング"
        case .sideRaise:
            return "肩の側面を集中的に鍛えるトレーニング"
        case .squat:
            return "下半身全体を鍛える王道トレーニング"
        case .pushUp:
            return "胸と腕を鍛える自重トレーニング"
        }
    }
    
    /// 難易度（1-5の5段階）
    var difficulty: Int {
        switch self {
        case .overheadPress:
            return 2
        case .sideRaise:
            return 2
        case .squat:
            return 2
        case .pushUp:
            return 2
        }
    }
    
    /// 推定カロリー消費量（10分あたり）
    var estimatedCalories: Int {
        switch self {
        case .overheadPress:
            return 60
        case .sideRaise:
            return 50
        case .squat:
            return 80
        case .pushUp:
            return 70
        }
    }
    
    /// 現在利用可能かどうか
    var isAvailable: Bool {
        switch self {
        case .overheadPress:
            return true
        case .sideRaise, .squat, .pushUp:
            return false // 将来実装予定
        }
    }
    
    /// SF Symbols のアイコン名
    var iconSystemName: String {
        switch self {
        case .overheadPress:
            return "figure.strengthtraining.traditional"
        case .sideRaise:
            return "figure.arms.open"
        case .squat:
            return "figure.strengthtraining.functional"
        case .pushUp:
            return "figure.walk"
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

    // MARK: - 目標情報プロパティ

    /// 目標時間（秒単位）- 1分 = 60秒
    var targetDuration: Int {
        return 60  // 全種目1分推奨
    }

    /// 目標回数（オプショナル）- 回数ベースの種目のみ
    var targetReps: Int? {
        switch self {
        case .overheadPress:
            return 10
        case .sideRaise:
            return 15
        case .squat:
            return 15
        case .pushUp:
            return 12
        }
    }

    /// 実践的なガイダンステキスト
    var guidanceText: String {
        switch self {
        case .overheadPress:
            return "1分間で10回を目標に、フォームを意識して丁寧に行いましょう"
        case .sideRaise:
            return "1分間で15回を目標に、肘を軽く曲げて肩の高さまで上げます"
        case .squat:
            return "1分間で15回を目標に、膝がつま先より前に出ないように注意"
        case .pushUp:
            return "1分間で12回を目標に、肘を90度まで曲げてください"
        }
    }

    /// 目標タイプ（時間ベース or 回数ベース）
    var targetType: TargetType {
        targetReps == nil ? .duration : .reps
    }

    /// 目標表示用テキスト（カード表示用）
    var targetDisplayText: String {
        if let reps = targetReps {
            return "目標: \(reps)回 / 1分"
        } else {
            return "目標: 1分キープ"
        }
    }
}

/// 目標タイプ列挙型
enum TargetType {
    case duration  // 時間ベース
    case reps      // 回数ベース
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