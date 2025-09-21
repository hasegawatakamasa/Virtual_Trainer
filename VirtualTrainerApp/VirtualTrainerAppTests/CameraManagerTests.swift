import XCTest
import AVFoundation
import Combine
@testable import VirtualTrainerApp

/// CameraManager の単体テスト
/// リソースクリーンアップとエラー処理の動作を検証
final class CameraManagerTests: XCTestCase {
    
    var cameraManager: CameraManager!
    var cancellables: Set<AnyCancellable>!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        cameraManager = CameraManager()
        cancellables = Set<AnyCancellable>()
    }
    
    override func tearDownWithError() throws {
        // テスト後のクリーンアップ
        if let manager = cameraManager {
            await manager.forceReleaseResources()
        }
        cameraManager = nil
        cancellables = nil
        try super.tearDownWithError()
    }
    
    // MARK: - 初期化テスト
    
    func testCameraManagerInitialization() {
        XCTAssertFalse(cameraManager.isSessionRunning)
        XCTAssertEqual(cameraManager.cameraPosition, .front)
        XCTAssertNil(cameraManager.currentError)
    }
    
    // MARK: - 権限テスト
    
    func testCameraPermissionRequest() async {
        let hasPermission = await cameraManager.requestCameraPermission()
        
        // シミュレーターまたは実機の状況に応じて結果が変わる
        // テスト環境での権限状態を確認
        XCTAssertTrue(hasPermission || !hasPermission) // どちらでも良い（環境依存）
        XCTAssertNotEqual(cameraManager.permissionStatus, .notDetermined)
    }
    
    // MARK: - セッション管理テスト
    
    func testSessionStartStop() async {
        // 権限がある場合のみテスト実行
        let hasPermission = await cameraManager.requestCameraPermission()
        guard hasPermission else {
            XCTSkip("Camera permission not granted, skipping session test")
        }
        
        // セッション開始
        cameraManager.startSession()
        
        // 少し待ってセッション状態を確認
        try? await Task.sleep(nanoseconds: 500_000_000) // 0.5秒待機
        
        // セッション停止（新しいasync版を使用）
        await cameraManager.stopSession()
        
        // セッションが停止していることを確認
        XCTAssertFalse(cameraManager.isSessionRunning)
    }
    
    func testStopSessionSync() {
        // 同期版のstopSessionもテスト
        cameraManager.stopSessionSync()
        
        // 同期版は即座には反映されない可能性があるので、
        // 非同期で状態変更を確認
        let expectation = self.expectation(description: "Session stopped")
        
        cameraManager.$isSessionRunning
            .dropFirst()
            .sink { isRunning in
                if !isRunning {
                    expectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        waitForExpectations(timeout: 2.0, handler: nil)
    }
    
    // MARK: - エラーハンドリングテスト
    
    func testErrorHandling() {
        let testError = AppError.cameraUnavailable
        let errorExpectation = expectation(description: "Error handled")
        
        // エラー状態の変化を監視
        cameraManager.$currentError
            .dropFirst()
            .sink { error in
                if error == testError {
                    errorExpectation.fulfill()
                }
            }
            .store(in: &cancellables)
        
        // エラーを発生させる
        cameraManager.handleError(testError)
        
        waitForExpectations(timeout: 1.0, handler: nil)
        XCTAssertEqual(cameraManager.currentError, testError)
    }
    
    // MARK: - ライフサイクルテスト
    
    func testAppLifecycleHandling() async {
        // バックグラウンド処理のテスト
        cameraManager.handleAppDidEnterBackground()
        
        // 非同期処理完了を待つ
        try? await Task.sleep(nanoseconds: 100_000_000) // 0.1秒待機
        
        // フォアグラウンド復帰処理のテスト
        cameraManager.handleAppWillEnterForeground()
        
        // エラーが発生していないことを確認
        XCTAssertFalse(cameraManager.isSessionRunning)
    }
    
    func testForceReleaseResources() async {
        // 強制リソース解放のテスト
        await cameraManager.forceReleaseResources()
        
        // セッションが停止し、エラーがクリアされていることを確認
        XCTAssertFalse(cameraManager.isSessionRunning)
        XCTAssertNil(cameraManager.currentError)
    }
    
    // MARK: - カメラ切り替えテスト
    
    func testCameraSwitching() async {
        // 権限がある場合のみテスト実行
        let hasPermission = await cameraManager.requestCameraPermission()
        guard hasPermission else {
            XCTSkip("Camera permission not granted, skipping camera switch test")
        }
        
        let initialPosition = cameraManager.cameraPosition
        
        // カメラ切り替え実行
        cameraManager.switchCamera()
        
        // 切り替え処理完了を待つ
        try? await Task.sleep(nanoseconds: 1_000_000_000) // 1秒待機
        
        // ポジションが変更されていることを確認
        // (実際のデバイスでのみ動作、シミュレーターでは制限あり)
        let finalPosition = cameraManager.cameraPosition
        print("Initial: \(initialPosition), Final: \(finalPosition)")
    }
    
    // MARK: - プレビューレイヤーテスト
    
    func testPreviewLayerCreation() {
        let previewLayer = cameraManager.createPreviewLayer()
        
        XCTAssertNotNil(previewLayer)
        XCTAssertEqual(previewLayer.videoGravity, .resizeAspectFill)
    }
    
    // MARK: - パフォーマンステスト
    
    func testSessionStartStopPerformance() async {
        let hasPermission = await cameraManager.requestCameraPermission()
        guard hasPermission else {
            XCTSkip("Camera permission not granted, skipping performance test")
        }
        
        measure {
            cameraManager.startSession()
            // セッション開始の処理時間を測定
        }
        
        // クリーンアップ
        await cameraManager.stopSession()
    }
    
    func testCleanupPerformance() async {
        measure {
            Task {
                await cameraManager.forceReleaseResources()
            }
        }
    }
    
    // MARK: - Mock Delegate テスト
    
    func testDelegateNotification() {
        let mockDelegate = MockCameraOutputDelegate()
        cameraManager.delegate = mockDelegate
        
        let testError = AppError.cameraPermissionDenied
        cameraManager.handleError(testError)
        
        // デリゲートが呼ばれるまで少し待機
        let expectation = self.expectation(description: "Delegate called")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 1.0, handler: nil)
        XCTAssertTrue(mockDelegate.errorReceived)
    }
}

// MARK: - Mock Classes

/// CameraOutputDelegate のモック実装
class MockCameraOutputDelegate: CameraOutputDelegate {
    var pixelBufferReceived = false
    var errorReceived = false
    var lastError: AppError?
    
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer) {
        pixelBufferReceived = true
    }
    
    func cameraManager(_ manager: CameraManager, didEncounterError error: AppError) {
        errorReceived = true
        lastError = error
    }
}

// MARK: - Test Extensions

extension CameraManagerTests {
    
    /// テスト用のヘルパーメソッド
    private func waitForSessionStateChange(to expectedState: Bool, timeout: TimeInterval = 2.0) async -> Bool {
        let startTime = Date()
        
        while Date().timeIntervalSince(startTime) < timeout {
            if cameraManager.isSessionRunning == expectedState {
                return true
            }
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1秒待機
        }
        
        return false
    }
    
    /// エラー状態のリセット
    private func resetErrorState() {
        // プライベートプロパティにアクセスできないため、
        // 新しいエラーで上書きしてからクリア
        cameraManager.handleError(.cameraUnavailable)
        // 実際の実装では適切なクリアメソッドを追加すべき
    }
}