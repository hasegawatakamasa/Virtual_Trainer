import Foundation
import CoreGraphics
import Combine

/// フォーム分析サービス
@MainActor
class FormAnalyzer: ObservableObject {
    // MARK: - Published Properties
    @Published var currentAngle: Double = 0.0
    @Published var isInExerciseZone: Bool = false
    @Published var lastAnalysisTime: Date = Date()
    @Published var currentExerciseType: ExerciseType = .overheadPress
    @Published var errorCount: Int = 0

    // MARK: - Private Properties
    private let config: RepCounterConfig
    private var exerciseConfig: ExerciseFormConfig
    
    // MARK: - Initialization
    init(exerciseType: ExerciseType = .overheadPress, config: RepCounterConfig = AppSettings.shared.createRepCounterConfig()) {
        self.currentExerciseType = exerciseType
        self.config = config
        self.exerciseConfig = ExerciseFormConfig.forExercise(exerciseType)
    }
    
    /// エクササイズタイプを変更
    func setExerciseType(_ exerciseType: ExerciseType) {
        currentExerciseType = exerciseType
        exerciseConfig = ExerciseFormConfig.forExercise(exerciseType)
    }
    
    // MARK: - Public Methods
    
    /// キーポイントから肘の角度を計算
    func calculateElbowAngle(keypoints: FilteredKeypoints) -> Double {
        let leftAngle = calculateAngle(
            point1: keypoints.leftShoulder,
            point2: keypoints.leftElbow,
            point3: keypoints.leftWrist
        )
        
        let rightAngle = calculateAngle(
            point1: keypoints.rightShoulder,
            point2: keypoints.rightElbow,
            point3: keypoints.rightWrist
        )
        
        // 両腕の平均角度を返す
        let averageAngle = (leftAngle + rightAngle) / 2.0
        
        // デバッグ情報の出力
        if config.debugMode && averageAngle > 0 {
            print("📐 Angle calculation: left=\(String(format: "%.1f", leftAngle))° right=\(String(format: "%.1f", rightAngle))° avg=\(String(format: "%.1f", averageAngle))°")
        } else if config.debugMode && averageAngle == 0 {
            print("⚠️ Zero angle detected - keypoints: left_shoulder=\(keypoints.leftShoulder) left_elbow=\(keypoints.leftElbow) left_wrist=\(keypoints.leftWrist)")
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.currentAngle = averageAngle
            self?.lastAnalysisTime = Date()
        }
        
        return averageAngle
    }
    
    /// エクササイズゾーン内かどうか判定（手首が肩より上）
    func isInExerciseZone(keypoints: FilteredKeypoints) -> Bool {
        // 両手首の平均Y座標
        let avgWristY = (keypoints.leftWrist.y + keypoints.rightWrist.y) / 2.0
        
        // 両肩の平均Y座標
        let avgShoulderY = (keypoints.leftShoulder.y + keypoints.rightShoulder.y) / 2.0
        
        // Y座標系では上が小さい値なので、手首Y < 肩Y の場合に「上にある」
        // 種目別のマージンを適用
        let margin = exerciseConfig.exerciseZoneMargin
        let inZone = avgWristY < (avgShoulderY + margin)
        
        
        DispatchQueue.main.async { [weak self] in
            self?.isInExerciseZone = inZone
        }
        
        return inZone
    }
    
    /// 正規化されたキーポイント特徴量を生成
    func normalizeKeypoints(_ keypoints: FilteredKeypoints) -> [Float]? {
        return keypoints.generateFeatures()
    }
    
    /// 包括的な分析を実行
    func analyzeForm(keypoints: FilteredKeypoints) -> FormAnalysisResult {
        let angle = calculateElbowAngle(keypoints: keypoints)
        let inZone = isInExerciseZone(keypoints: keypoints)
        let features = normalizeKeypoints(keypoints)
        
        return FormAnalysisResult(
            elbowAngle: angle,
            isInExerciseZone: inZone,
            normalizedFeatures: features,
            timestamp: Date(),
            keypoints: keypoints
        )
    }
    
    // MARK: - Private Methods
    
    /// 3点間の角度を計算（既存Python実装と同じアルゴリズム）
    private func calculateAngle(point1: CGPoint, point2: CGPoint, point3: CGPoint) -> Double {
        // 無効な点（0,0）をチェック
        guard point1 != .zero && point2 != .zero && point3 != .zero else {
            return 0.0
        }
        
        // ベクトル計算
        let vector1 = CGPoint(x: point1.x - point2.x, y: point1.y - point2.y)
        let vector2 = CGPoint(x: point3.x - point2.x, y: point3.y - point2.y)
        
        // 角度計算
        let angle1 = atan2(vector1.y, vector1.x)
        let angle2 = atan2(vector2.y, vector2.x)
        
        // ラジアンから度に変換
        var angleDifference = abs(angle1 - angle2) * 180.0 / .pi
        
        // 0-180度の範囲に正規化
        if angleDifference > 180.0 {
            angleDifference = 360.0 - angleDifference
        }
        
        return Double(angleDifference)
    }
}

/// フォーム分析の結果
struct FormAnalysisResult: Equatable {
    let elbowAngle: Double
    let isInExerciseZone: Bool
    let normalizedFeatures: [Float]?
    let timestamp: Date
    let keypoints: FilteredKeypoints
    
    /// 分析が有効かどうか
    var isValid: Bool {
        return normalizedFeatures != nil && elbowAngle > 0
    }
    
    /// デバッグ用の情報
    var debugDescription: String {
        return """
        FormAnalysisResult:
        - Elbow Angle: \(String(format: "%.1f", elbowAngle))°
        - In Exercise Zone: \(isInExerciseZone)
        - Features Count: \(normalizedFeatures?.count ?? 0)
        - Timestamp: \(timestamp)
        """
    }
}

/// フォーム分析の設定
struct FormAnalysisConfig {
    /// 角度計算の精度
    let anglePrecision: Double
    
    /// エクササイズゾーンの閾値（肩からの相対距離）
    let exerciseZoneThreshold: Double
    
    /// 特徴量正規化の設定
    let normalizationEnabled: Bool
    
    static let `default` = FormAnalysisConfig(
        anglePrecision: 0.1,
        exerciseZoneThreshold: 0.0,
        normalizationEnabled: true
    )
}

/// フォーム品質評価
enum FormQuality: String, CaseIterable {
    case excellent = "excellent"
    case good = "good"
    case fair = "fair"
    case poor = "poor"
    
    var displayName: String {
        switch self {
        case .excellent:
            return "優秀"
        case .good:
            return "良好"
        case .fair:
            return "普通"
        case .poor:
            return "要改善"
        }
    }
    
    var color: (red: Double, green: Double, blue: Double) {
        switch self {
        case .excellent:
            return (red: 0.0, green: 1.0, blue: 0.0) // 緑
        case .good:
            return (red: 0.5, green: 1.0, blue: 0.0) // 黄緑
        case .fair:
            return (red: 1.0, green: 0.8, blue: 0.0) // オレンジ
        case .poor:
            return (red: 1.0, green: 0.0, blue: 0.0) // 赤
        }
    }
    
    /// 角度に基づいてフォーム品質を評価
    static func evaluate(angle: Double, inZone: Bool, config: ExerciseFormConfig = .default) -> FormQuality {
        guard inZone else { return .poor }
        
        // 種目別の理想的な角度範囲での評価
        switch angle {
        case config.excellentAngleRange:
            return .excellent
        case config.goodAngleRange:
            return .good
        case config.fairAngleRange:
            return .fair
        default:
            return .poor
        }
    }
}

/// 種目別のフォーム設定
struct ExerciseFormConfig {
    let exerciseType: ExerciseType
    let excellentAngleRange: ClosedRange<Double>
    let goodAngleRange: ClosedRange<Double>
    let fairAngleRange: ClosedRange<Double>
    let exerciseZoneMargin: Double
    let primaryMuscleGroups: [String]
    let formCheckpoints: [String]
    
    static let `default` = ExerciseFormConfig.forExercise(.overheadPress)
    
    /// 種目に応じたフォーム設定を生成
    static func forExercise(_ exerciseType: ExerciseType) -> ExerciseFormConfig {
        switch exerciseType {
        case .overheadPress:
            return ExerciseFormConfig(
                exerciseType: .overheadPress,
                excellentAngleRange: 80...120,
                goodAngleRange: 70...130,
                fairAngleRange: 60...140,
                exerciseZoneMargin: 20.0,
                primaryMuscleGroups: ["三角筋", "上腕三頭筋", "体幹"],
                formCheckpoints: ["肘が体の前に出過ぎない", "腰を反らさない", "肩甲骨を安定"]
            )
        case .squat:
            return ExerciseFormConfig(
                exerciseType: .squat,
                excellentAngleRange: 85...115,
                goodAngleRange: 75...125,
                fairAngleRange: 65...135,
                exerciseZoneMargin: 30.0,
                primaryMuscleGroups: ["大腿四頭筋", "臀筋", "ハムストリング"],
                formCheckpoints: ["膝がつま先より前に出ない", "背中をまっすぐ", "太ももが床と平行"]
            )
        case .pushUp:
            return ExerciseFormConfig(
                exerciseType: .pushUp,
                excellentAngleRange: 75...115,
                goodAngleRange: 65...125,
                fairAngleRange: 55...135,
                exerciseZoneMargin: 15.0,
                primaryMuscleGroups: ["胸筋", "上腕三頭筋", "体幹"],
                formCheckpoints: ["体を一直線に保つ", "胸が床に近づく", "手は肩幅より少し広め"]
            )
        case .sideRaise:
            return ExerciseFormConfig(
                exerciseType: .sideRaise,
                excellentAngleRange: 85...95,
                goodAngleRange: 75...105,
                fairAngleRange: 65...115,
                exerciseZoneMargin: 20.0,
                primaryMuscleGroups: ["三角筋（側部）", "僧帽筋", "上腕"],
                formCheckpoints: ["肘を軽く曲げる", "肩の高さまで上げる", "ゆっくりと制御して下ろす"]
            )
        }
    }
}