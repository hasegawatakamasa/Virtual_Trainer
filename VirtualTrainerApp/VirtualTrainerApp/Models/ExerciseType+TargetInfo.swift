import Foundation

// MARK: - ExerciseType+TargetInfo Extension

extension ExerciseType {
    /// 目標情報を構造化したデータ
    struct TargetInfo {
        let duration: Int
        let reps: Int?
        let type: TargetType
        let displayText: String
        let guidanceText: String
    }

    /// 目標情報を取得
    var targetInfo: TargetInfo {
        TargetInfo(
            duration: targetDuration,
            reps: targetReps,
            type: targetType,
            displayText: targetDisplayText,
            guidanceText: guidanceText
        )
    }
}