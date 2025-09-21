import SwiftUI

/// キーポイントと骨格を描画するオーバーレイビュー
struct KeypointOverlayView: View {
    let keypoints: PoseKeypoints?
    let viewSize: CGSize
    
    // COCO-Pose 17キーポイントの接続定義
    private let connections: [(Int, Int)] = [
        // 顔・頭部
        (0, 1), (0, 2), (1, 3), (2, 4),  // 鼻→目→耳
        
        // 上半身
        (5, 6),   // 肩同士
        (5, 7), (6, 8),   // 肩→肘
        (7, 9), (8, 10),  // 肘→手首
        (5, 11), (6, 12), // 肩→腰
        (11, 12),         // 腰同士
        
        // 下半身
        (11, 13), (12, 14), // 腰→膝
        (13, 15), (14, 16)  // 膝→足首
    ]
    
    var body: some View {
        Canvas { context, size in
            guard let keypoints = keypoints else { return }
            
            // 座標変換係数を計算
            let scaleX = size.width / 640.0  // YOLO入力サイズ
            let scaleY = size.height / 640.0
            
            // 骨格の線を描画
            drawSkeleton(context: context, keypoints: keypoints, scaleX: scaleX, scaleY: scaleY)
            
            // キーポイントを描画
            drawKeypoints(context: context, keypoints: keypoints, scaleX: scaleX, scaleY: scaleY)
        }
        .allowsHitTesting(false) // タッチイベントを透過
    }
    
    private func drawSkeleton(context: GraphicsContext, keypoints: PoseKeypoints, scaleX: Double, scaleY: Double) {
        // 骨格の線を描画
        for (startIndex, endIndex) in connections {
            guard startIndex < keypoints.points.count,
                  endIndex < keypoints.points.count,
                  keypoints.confidence[startIndex] > 0.5,
                  keypoints.confidence[endIndex] > 0.5 else { continue }
            
            let startPoint = keypoints.points[startIndex]
            let endPoint = keypoints.points[endIndex]
            
            // 座標変換
            let start = CGPoint(
                x: startPoint.x * scaleX,
                y: startPoint.y * scaleY
            )
            let end = CGPoint(
                x: endPoint.x * scaleX,
                y: endPoint.y * scaleY
            )
            
            // 線の色を部位別に設定
            let color = connectionColor(startIndex: startIndex, endIndex: endIndex)
            
            // 線を描画
            var path = Path()
            path.move(to: start)
            path.addLine(to: end)
            
            context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3.0, lineCap: .round))
        }
    }
    
    private func drawKeypoints(context: GraphicsContext, keypoints: PoseKeypoints, scaleX: Double, scaleY: Double) {
        // キーポイントを描画
        for (index, point) in keypoints.points.enumerated() {
            guard index < keypoints.confidence.count,
                  keypoints.confidence[index] > 0.3 else { continue }
            
            // 座標変換
            let drawPoint = CGPoint(
                x: point.x * scaleX,
                y: point.y * scaleY
            )
            
            // 信頼度に基づいて色とサイズを決定
            let confidence = keypoints.confidence[index]
            let color = keypointColor(index: index, confidence: confidence)
            let radius = 4.0 + (Double(confidence) * 4.0) // 3-8pxの範囲
            
            // 円を描画
            let rect = CGRect(
                x: drawPoint.x - CGFloat(radius),
                y: drawPoint.y - CGFloat(radius),
                width: CGFloat(radius) * 2,
                height: CGFloat(radius) * 2
            )
            
            context.fill(Path(ellipseIn: rect), with: .color(color))
            
            // 境界線
            context.stroke(Path(ellipseIn: rect), with: .color(.white), style: StrokeStyle(lineWidth: 1.0))
        }
    }
    
    private func connectionColor(startIndex: Int, endIndex: Int) -> Color {
        // 接続線の色分け
        if [5, 6, 7, 8, 9, 10].contains(startIndex) && [5, 6, 7, 8, 9, 10].contains(endIndex) {
            return .green  // 上半身: 緑
        } else if [11, 12, 13, 14, 15, 16].contains(startIndex) && [11, 12, 13, 14, 15, 16].contains(endIndex) {
            return .blue   // 下半身: 青
        } else if [0, 1, 2, 3, 4].contains(startIndex) && [0, 1, 2, 3, 4].contains(endIndex) {
            return .yellow // 顔: 黄
        } else {
            return .orange // その他の接続: オレンジ
        }
    }
    
    private func keypointColor(index: Int, confidence: Float) -> Color {
        // 信頼度に基づく色の濃淡
        let alpha = Double(confidence)
        
        switch index {
        case 0...4:  // 顔
            return Color.yellow.opacity(alpha)
        case 5...10: // 上半身
            return Color.green.opacity(alpha)
        case 11...16: // 下半身
            return Color.blue.opacity(alpha)
        default:
            return Color.red.opacity(alpha)
        }
    }
}

// MARK: - Preview
#Preview {
    ZStack {
        Color.black.ignoresSafeArea()
        
        KeypointOverlayView(
            keypoints: {
                // モック用のキーポイントデータ
                let points = (0..<17).map { index -> CGPoint in
                    let x = Double.random(in: 100...540)
                    let y = Double.random(in: 100...540)
                    return CGPoint(x: x, y: y)
                }
                let confidence = (0..<17).map { _ in Float.random(in: 0.5...1.0) }
                return PoseKeypoints(points: points, confidence: confidence)
            }(),
            viewSize: CGSize(width: 390, height: 844)
        )
    }
}