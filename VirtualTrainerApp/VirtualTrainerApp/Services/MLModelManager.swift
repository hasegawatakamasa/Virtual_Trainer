import Foundation
import Vision
import CoreML
import UIKit
import Combine

/// AI/MLモデル管理サービス
@MainActor
class MLModelManager: ObservableObject {
    
    // MARK: - Published Properties
    @Published var isModelLoaded = false
    @Published var loadingError: AppError?
    @Published var lastInferenceTime: TimeInterval = 0.0
    @Published var lastDetectedPose: PoseKeypoints?
    
    // MARK: - Private Properties
    private var poseModel: VNCoreMLModel?
    private var formModel: MLModel?
    private let processingQueue = DispatchQueue(label: "ml.processing", qos: .userInitiated)
    
    // 模擬データ用の状態管理
    private var mockExercisePhase: Double = 0.0  // 0.0〜2π のサイクル
    private var mockPhaseSpeed: Double = 0.05    // フレームごとの進行速度
    
    // MARK: - Initialization
    init() {
        loadModels()
    }
    
    // MARK: - Model Loading
    
    private func loadModels() {
        Task {
            await loadPoseModel()
            await loadFormModel()
            
            await MainActor.run {
                self.isModelLoaded = (self.poseModel != nil) && (self.formModel != nil)
                if self.isModelLoaded {
                    print("✅ AIモデル読み込み完了 (YOLO11n-pose + GRU)")
                } else {
                    print("⚠️ AIモデルが見つかりません - 模擬モードで動作します")
                    self.loadingError = AppError.modelLoadingFailed("モデルファイルが見つかりません")
                }
            }
        }
    }
    
    private func loadPoseModel() async {
        // コンパイル済み(.mlmodelc)とパッケージ(.mlpackage)の両方をチェック
        var modelURL: URL?
        
        // まずコンパイル済みモデルを試す
        if let compiledURL = Bundle.main.url(forResource: "YOLO11nPose", withExtension: "mlmodelc") {
            modelURL = compiledURL
        }
        // 次にパッケージモデルを試す
        else if let packageURL = Bundle.main.url(forResource: "YOLO11nPose", withExtension: "mlpackage") {
            modelURL = packageURL
        }
        
        guard let modelURL = modelURL else {
            return
        }
        
        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            self.poseModel = try VNCoreMLModel(for: mlModel)
        } catch {
            print("❌ YOLO11nPoseモデルの読み込みに失敗: \(error)")
        }
    }
    
    private func loadFormModel() async {
        // コンパイル済み(.mlmodelc)とパッケージ(.mlpackage)の両方をチェック
        var modelURL: URL?
        
        // まずコンパイル済みモデルを試す
        if let compiledURL = Bundle.main.url(forResource: "GRUFormClassifier", withExtension: "mlmodelc") {
            modelURL = compiledURL
        }
        // 次にパッケージモデルを試す
        else if let packageURL = Bundle.main.url(forResource: "GRUFormClassifier", withExtension: "mlpackage") {
            modelURL = packageURL
        }
        
        guard let modelURL = modelURL else {
            return
        }
        
        do {
            self.formModel = try MLModel(contentsOf: modelURL)
        } catch {
            print("❌ GRUFormClassifierモデルの読み込みに失敗: \(error)")
        }
    }
    
    // MARK: - Pose Detection
    
    /// Vision フレームワークを使った姿勢検出
    func detectPose(in pixelBuffer: CVPixelBuffer) async -> PoseKeypoints? {
        guard let poseModel = poseModel else {
            // モデルが利用できない場合はnilを返す（模擬データは使わない）
            return nil
        }
        
        return await withCheckedContinuation { continuation in
            let request = VNCoreMLRequest(model: poseModel) { request, error in
                if let error = error {
                    print("❌ 姿勢検出エラー: \(error.localizedDescription)")
                    continuation.resume(returning: nil)
                    return
                }
                
                guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
                      let firstObservation = observations.first,
                      let multiArray = firstObservation.featureValue.multiArrayValue else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let keypoints = self.parseYOLOOutput(multiArray)
                
                // メインアクターでPublishedプロパティを更新
                Task { @MainActor in
                    self.lastDetectedPose = keypoints
                }
                
                continuation.resume(returning: keypoints)
            }
            
            request.imageCropAndScaleOption = .scaleFill
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("❌ Vision request実行エラー: \\(error)")
                continuation.resume(returning: nil)
            }
        }
    }
    
    private func parseYOLOOutput(_ multiArray: MLMultiArray) -> PoseKeypoints? {
        // YOLO11n-poseの出力形式: (1, 56, 8400)
        // 56 = 4 (bbox) + 1 (conf) + 51 (17 keypoints × 3: x, y, visibility)
        
        guard multiArray.shape.count >= 3 else {
            print("❌ 予期しない出力形状: \\(multiArray.shape)")
            return nil
        }
        
        let numDetections = multiArray.shape[2].intValue
        guard numDetections > 0 else { return nil }
        
        // 最も信頼度の高い検出を選択
        var bestConfidence: Float = 0.0
        var bestDetectionIndex = 0
        
        for i in 0..<numDetections {
            let confidence = multiArray[[0, 4, i] as [NSNumber]].floatValue
            if confidence > bestConfidence {
                bestConfidence = confidence
                bestDetectionIndex = i
            }
        }
        
        guard bestConfidence > 0.3 else { // 最小信頼度チェック
            return nil
        }
        
        // キーポイントを抽出
        var points: [CGPoint] = []
        var confidences: [Float] = []
        
        for keypointIndex in 0..<17 {
            let xIndex = 5 + keypointIndex * 3
            let yIndex = 5 + keypointIndex * 3 + 1
            let confIndex = 5 + keypointIndex * 3 + 2
            
            let x = multiArray[[0, xIndex, bestDetectionIndex] as [NSNumber]].floatValue
            let y = multiArray[[0, yIndex, bestDetectionIndex] as [NSNumber]].floatValue
            let conf = multiArray[[0, confIndex, bestDetectionIndex] as [NSNumber]].floatValue
            
            points.append(CGPoint(x: Double(x), y: Double(y)))
            confidences.append(conf)
        }
        
        return PoseKeypoints(points: points, confidence: confidences)
    }
    
    // MARK: - Form Classification
    
    /// GRUモデルを使ったフォーム分類
    func classifyForm(features: [[Float]]) async -> FormClassification {
        guard let formModel = formModel,
              features.count >= 10 else {
            // モデルが利用できない場合やデータが不足している場合は模擬結果を返す
            return generateMockFormClassification()
        }
        
        return await withCheckedContinuation { continuation in
            processingQueue.async {
                do {
                    // 入力データの準備（10フレーム × 12特徴量）
                    let inputArray = try MLMultiArray(shape: [1, 10, 12], dataType: .float32)
                    
                    for (frameIndex, frame) in features.prefix(10).enumerated() {
                        for (featureIndex, value) in frame.prefix(12).enumerated() {
                            inputArray[[0, frameIndex, featureIndex] as [NSNumber]] = NSNumber(value: value)
                        }
                    }
                    
                    let input = try MLDictionaryFeatureProvider(dictionary: ["features": MLFeatureValue(multiArray: inputArray)])
                    let prediction = try formModel.prediction(from: input)
                    
                    guard let outputArray = prediction.featureValue(for: "prediction")?.multiArrayValue else {
                        continuation.resume(returning: .ready)
                        return
                    }
                    
                    let confidence = outputArray[0].floatValue
                    let classification: FormClassification
                    
                    if confidence < 0.3 {
                        classification = .normal
                    } else if confidence < 0.7 {
                        classification = .elbowError
                    } else {
                        classification = .tooFast
                    }
                    
                    continuation.resume(returning: classification)
                    
                } catch {
                    print("❌ フォーム分類エラー: \\(error)")
                    continuation.resume(returning: .ready)
                }
            }
        }
    }
    
    // MARK: - Mock Data Generation
    
    private func generateMockPoseKeypoints() -> PoseKeypoints {
        let points = (0..<17).map { index -> CGPoint in
            switch index {
            case 5, 6: // 肩
                return CGPoint(x: 200 + Double(index - 5) * 100, y: 150)
            case 7, 8: // 肘
                let angle = Double.random(in: 90...140)
                return CGPoint(x: 200 + Double(index - 7) * 100, y: 200 + sin(angle * .pi / 180) * 50)
            case 9, 10: // 手首
                return CGPoint(x: 200 + Double(index - 9) * 100, y: 250 + Double.random(in: -30...30))
            default:
                return CGPoint(x: Double.random(in: 100...300), y: Double.random(in: 100...400))
            }
        }
        
        let confidences = (0..<17).map { _ in Float.random(in: 0.7...1.0) }
        let mockKeypoints = PoseKeypoints(points: points, confidence: confidences)
        
        // Published プロパティを更新
        Task { @MainActor in
            self.lastDetectedPose = mockKeypoints
        }
        
        return mockKeypoints
    }
    
    private func generateMockFormClassification() -> FormClassification {
        let classifications: [FormClassification] = [.normal, .elbowError, .tooFast, .ready]
        return classifications.randomElement() ?? .normal
    }
    
    // MARK: - Performance Monitoring
    
    func updatePerformanceMetrics(inferenceTime: TimeInterval) {
        Task { @MainActor in
            self.lastInferenceTime = inferenceTime
        }
    }
}

// MARK: - Extensions

extension MLModelManager {
    
    /// モデルの状態情報を取得
    var modelStatus: String {
        if isModelLoaded {
            return "✅ モデル読み込み完了"
        } else if loadingError != nil {
            return "❌ モデル読み込みエラー"
        } else {
            return "🔄 モデル読み込み中..."
        }
    }
    
    /// デバッグ情報
    var debugInfo: String {
        return """
        MLModelManager Debug Info:
        - Pose Model: \(poseModel != nil ? "Loaded" : "Mock Mode")
        - Form Model: \(formModel != nil ? "Loaded" : "Mock Mode")
        - Last Inference: \(String(format: "%.1fms", lastInferenceTime * 1000))
        - Status: \(modelStatus)
        """
    }
}