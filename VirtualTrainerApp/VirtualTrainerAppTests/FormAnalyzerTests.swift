import XCTest
import CoreGraphics
@testable import VirtualTrainerApp

final class FormAnalyzerTests: XCTestCase {
    var formAnalyzer: FormAnalyzer!
    
    override func setUp() {
        super.setUp()
        formAnalyzer = FormAnalyzer()
    }
    
    override func tearDown() {
        formAnalyzer = nil
        super.tearDown()
    }
    
    // MARK: - 角度計算テスト
    
    func testElbowAngleCalculation() {
        // 90度の角度を形成するキーポイント
        let keypoints = FilteredKeypoints(
            leftShoulder: CGPoint(x: 100, y: 100),
            rightShoulder: CGPoint(x: 200, y: 100),
            leftElbow: CGPoint(x: 100, y: 150),
            rightElbow: CGPoint(x: 200, y: 150),
            leftWrist: CGPoint(x: 50, y: 150),   // 90度
            rightWrist: CGPoint(x: 250, y: 150), // 90度
            timestamp: Date()
        )
        
        let angle = formAnalyzer.calculateElbowAngle(keypoints: keypoints)
        
        // 90度前後の範囲で検証（計算誤差を考慮）
        XCTAssertTrue(angle >= 80 && angle <= 100, "角度が期待値範囲外: \(angle)")
    }
    
    func testElbowAngleWithInvalidPoints() {
        // 無効な点（0,0）を含むキーポイント
        let keypoints = FilteredKeypoints(
            leftShoulder: CGPoint.zero, // 無効
            rightShoulder: CGPoint(x: 200, y: 100),
            leftElbow: CGPoint.zero, // 無効
            rightElbow: CGPoint(x: 200, y: 150),
            leftWrist: CGPoint.zero, // 無効
            rightWrist: CGPoint(x: 250, y: 150),
            timestamp: Date()
        )
        
        let angle = formAnalyzer.calculateElbowAngle(keypoints: keypoints)
        
        // 無効な点があっても、有効な右腕の角度が計算される
        XCTAssertTrue(angle >= 0, "角度は0以上である必要があります")
    }
    
    // MARK: - エクササイズゾーン判定テスト
    
    func testExerciseZoneDetection() {
        // 手首が肩より上にあるケース
        let keypointsInZone = FilteredKeypoints(
            leftShoulder: CGPoint(x: 100, y: 100),
            rightShoulder: CGPoint(x: 200, y: 100),
            leftElbow: CGPoint(x: 100, y: 80),
            rightElbow: CGPoint(x: 200, y: 80),
            leftWrist: CGPoint(x: 100, y: 60),   // 肩より上
            rightWrist: CGPoint(x: 200, y: 60),  // 肩より上
            timestamp: Date()
        )
        
        let inZone = formAnalyzer.isInExerciseZone(keypoints: keypointsInZone)
        XCTAssertTrue(inZone, "手首が肩より上にある場合、ゾーン内と判定されるべき")
    }
    
    func testExerciseZoneDetectionOutside() {
        // 手首が肩より下にあるケース
        let keypointsOutZone = FilteredKeypoints(
            leftShoulder: CGPoint(x: 100, y: 100),
            rightShoulder: CGPoint(x: 200, y: 100),
            leftElbow: CGPoint(x: 100, y: 120),
            rightElbow: CGPoint(x: 200, y: 120),
            leftWrist: CGPoint(x: 100, y: 140),  // 肩より下
            rightWrist: CGPoint(x: 200, y: 140), // 肩より下
            timestamp: Date()
        )
        
        let inZone = formAnalyzer.isInExerciseZone(keypoints: keypointsOutZone)
        XCTAssertFalse(inZone, "手首が肩より下にある場合、ゾーン外と判定されるべき")
    }
    
    // MARK: - 特徴量正規化テスト
    
    func testFeatureNormalization() {
        let keypoints = FilteredKeypoints(
            leftShoulder: CGPoint(x: 100, y: 100),
            rightShoulder: CGPoint(x: 200, y: 100),
            leftElbow: CGPoint(x: 100, y: 150),
            rightElbow: CGPoint(x: 200, y: 150),
            leftWrist: CGPoint(x: 100, y: 200),
            rightWrist: CGPoint(x: 200, y: 200),
            timestamp: Date()
        )
        
        let features = formAnalyzer.normalizeKeypoints(keypoints)
        
        XCTAssertNotNil(features, "特徴量が生成されるべき")
        XCTAssertEqual(features?.count, 12, "12次元の特徴量が生成されるべき") // 6点 × 2次元
    }
    
    // MARK: - 包括的分析テスト
    
    func testComprehensiveAnalysis() {
        let keypoints = FilteredKeypoints(
            leftShoulder: CGPoint(x: 100, y: 100),
            rightShoulder: CGPoint(x: 200, y: 100),
            leftElbow: CGPoint(x: 100, y: 150),
            rightElbow: CGPoint(x: 200, y: 150),
            leftWrist: CGPoint(x: 100, y: 80), // エクササイズゾーン内
            rightWrist: CGPoint(x: 200, y: 80),
            timestamp: Date()
        )
        
        let result = formAnalyzer.analyzeForm(keypoints: keypoints)
        
        XCTAssertTrue(result.isValid, "分析結果が有効であるべき")
        XCTAssertTrue(result.isInExerciseZone, "エクササイズゾーン内と判定されるべき")
        XCTAssertTrue(result.elbowAngle > 0, "角度が計算されるべき")
        XCTAssertNotNil(result.normalizedFeatures, "正規化特徴量が生成されるべき")
    }
    
    // MARK: - パフォーマンステスト
    
    func testAnalysisPerformance() {
        let keypoints = FilteredKeypoints(
            leftShoulder: CGPoint(x: 100, y: 100),
            rightShoulder: CGPoint(x: 200, y: 100),
            leftElbow: CGPoint(x: 100, y: 150),
            rightElbow: CGPoint(x: 200, y: 150),
            leftWrist: CGPoint(x: 100, y: 200),
            rightWrist: CGPoint(x: 200, y: 200),
            timestamp: Date()
        )
        
        measure {
            // 分析処理の性能を測定
            for _ in 0..<100 {
                _ = formAnalyzer.analyzeForm(keypoints: keypoints)
            }
        }
    }
}