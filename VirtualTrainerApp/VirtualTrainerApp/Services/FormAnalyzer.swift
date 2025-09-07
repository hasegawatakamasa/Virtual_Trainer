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
    
    // MARK: - Private Properties
    private let config: RepCounterConfig
    
    // MARK: - Initialization
    init(config: RepCounterConfig = AppSettings.shared.createRepCounterConfig()) {
        self.config = config
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
        // 少しマージンを持たせて判定を緩くする（肩より少し下でもOK）
        let margin = 20.0  // ピクセル単位のマージン
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
    static func evaluate(angle: Double, inZone: Bool) -> FormQuality {
        guard inZone else { return .poor }
        
        // 理想的な角度範囲での評価
        switch angle {
        case 80...120:
            return .excellent
        case 70...130:
            return .good
        case 60...140:
            return .fair
        default:
            return .poor
        }
    }
}