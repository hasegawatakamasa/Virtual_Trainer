import Foundation
import CoreGraphics

/// 姿勢検出で得られる17個のキーポイント情報
struct PoseKeypoints: Codable, Equatable {
    /// 17個のキーポイント座標
    let points: [CGPoint]
    
    /// 各ポイントの信頼度 (0.0-1.0)
    let confidence: [Float]
    
    /// 検出された時刻
    let timestamp: Date
    
    /// キーポイントのインデックス定義
    enum KeypointIndex: Int, CaseIterable {
        case nose = 0
        case leftEye = 1
        case rightEye = 2
        case leftEar = 3
        case rightEar = 4
        case leftShoulder = 5
        case rightShoulder = 6
        case leftElbow = 7
        case rightElbow = 8
        case leftWrist = 9
        case rightWrist = 10
        case leftHip = 11
        case rightHip = 12
        case leftKnee = 13
        case rightKnee = 14
        case leftAnkle = 15
        case rightAnkle = 16
    }
    
    init(points: [CGPoint], confidence: [Float], timestamp: Date = Date()) {
        self.points = points
        self.confidence = confidence
        self.timestamp = timestamp
    }
    
    /// 有効性をチェック
    var isValid: Bool {
        guard points.count == 17 && confidence.count == 17 else { return false }
        
        // 重要なキーポイント（肩、肘、手首）が有効かチェック
        let requiredPoints: [KeypointIndex] = [
            .leftShoulder, .rightShoulder,
            .leftElbow, .rightElbow,
            .leftWrist, .rightWrist
        ]
        
        for keypoint in requiredPoints {
            let point = points[keypoint.rawValue]
            let conf = confidence[keypoint.rawValue]
            
            // 座標が0でない、かつ信頼度が閾値以上
            if point == .zero || conf < 0.3 {
                return false
            }
        }
        
        return true
    }
    
    /// 特定のキーポイントを取得
    func point(for keypoint: KeypointIndex) -> CGPoint? {
        guard keypoint.rawValue < points.count,
              confidence[keypoint.rawValue] > 0.3 else {
            return nil
        }
        
        let point = points[keypoint.rawValue]
        return point == .zero ? nil : point
    }
    
    /// 特定のキーポイントの信頼度を取得
    func confidence(for keypoint: KeypointIndex) -> Float? {
        guard keypoint.rawValue < confidence.count else { return nil }
        return confidence[keypoint.rawValue]
    }
}

/// フィルタリング済みの重要なキーポイント
struct FilteredKeypoints: Equatable {
    let leftShoulder: CGPoint
    let rightShoulder: CGPoint
    let leftElbow: CGPoint
    let rightElbow: CGPoint
    let leftWrist: CGPoint
    let rightWrist: CGPoint
    let timestamp: Date
    
    init?(from poseKeypoints: PoseKeypoints) {
        guard poseKeypoints.isValid else { return nil }
        
        guard let leftShoulder = poseKeypoints.point(for: .leftShoulder),
              let rightShoulder = poseKeypoints.point(for: .rightShoulder),
              let leftElbow = poseKeypoints.point(for: .leftElbow),
              let rightElbow = poseKeypoints.point(for: .rightElbow),
              let leftWrist = poseKeypoints.point(for: .leftWrist),
              let rightWrist = poseKeypoints.point(for: .rightWrist) else {
            return nil
        }
        
        self.leftShoulder = leftShoulder
        self.rightShoulder = rightShoulder
        self.leftElbow = leftElbow
        self.rightElbow = rightElbow
        self.leftWrist = leftWrist
        self.rightWrist = rightWrist
        self.timestamp = poseKeypoints.timestamp
    }
    
    /// 正規化された特徴量を生成（既存Python実装と同じロジック）
    func generateFeatures() -> [Float]? {
        // 肩の中心点を計算
        let centerPoint = CGPoint(
            x: (leftShoulder.x + rightShoulder.x) / 2.0,
            y: (leftShoulder.y + rightShoulder.y) / 2.0
        )
        
        // 肩間の距離を計算
        let shoulderDistance = sqrt(
            pow(leftShoulder.x - rightShoulder.x, 2) +
            pow(leftShoulder.y - rightShoulder.y, 2)
        )
        
        guard shoulderDistance > 0 else { return nil }
        
        // すべてのポイントを中心点を基準に相対座標に変換
        let points = [leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist]
        let relativePoints = points.map { point in
            CGPoint(
                x: point.x - centerPoint.x,
                y: point.y - centerPoint.y
            )
        }
        
        // 肩の距離で正規化
        let normalizedPoints = relativePoints.map { point in
            CGPoint(
                x: point.x / shoulderDistance,
                y: point.y / shoulderDistance
            )
        }
        
        // 12次元の特徴量に変換 (x, y) × 6点
        return normalizedPoints.flatMap { [Float($0.x), Float($0.y)] }
    }
}