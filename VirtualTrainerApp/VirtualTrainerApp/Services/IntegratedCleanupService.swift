import Foundation
import Combine

/// カメラと音声の統合クリーンアップを管理するサービス
/// 複数のサービス間の依存関係を考慮した適切な順序でクリーンアップを実行
@MainActor
class IntegratedCleanupService: ObservableObject {
    
    // MARK: - Published Properties
    
    @Published private(set) var isCleanupInProgress = false
    @Published private(set) var cleanupResult: CleanupResult?
    
    // MARK: - Private Properties
    
    private let cameraManager: CameraManager
    private let audioFeedbackService: AudioFeedbackService
    private let coordinatedCleanup = ResourceCleanupCoordinator()
    
    // MARK: - Initialization
    
    init(cameraManager: CameraManager, audioFeedbackService: AudioFeedbackService) {
        self.cameraManager = cameraManager
        self.audioFeedbackService = audioFeedbackService
        setupCoordinatedCleanup()
    }
    
    // MARK: - Public Methods
    
    /// すべてのリソースの統合クリーンアップを実行
    /// - Returns: クリーンアップが完全に成功した場合はtrue
    func performIntegratedCleanup() async -> Bool {
        guard !isCleanupInProgress else {
            print("[IntegratedCleanupService] Cleanup already in progress")
            return false
        }
        
        isCleanupInProgress = true
        cleanupResult = nil
        
        print("[IntegratedCleanupService] Starting integrated cleanup process")
        
        let startTime = Date()
        var cameraSuccess = false
        var audioSuccess = false
        
        // Phase 1: 音声を先に停止（カメラより応答が早い）
        print("[IntegratedCleanupService] Phase 1: Audio cleanup")
        audioSuccess = await audioFeedbackService.stopWithCleanup()
        
        // Phase 2: カメラクリーンアップ
        print("[IntegratedCleanupService] Phase 2: Camera cleanup")
        await cameraManager.stopSession()
        cameraSuccess = true // CameraManager.stopSession()は内部でクリーンアップ実行
        
        // Phase 3: 協調クリーンアップの最終確認
        print("[IntegratedCleanupService] Phase 3: Final coordination check")
        let coordinationSuccess = await coordinatedCleanup.initiateCleanup()
        
        let overallSuccess = cameraSuccess && audioSuccess && coordinationSuccess
        let duration = Date().timeIntervalSince(startTime)
        
        // 結果を保存
        cleanupResult = CleanupResult(
            cameraSuccess: cameraSuccess,
            audioSuccess: audioSuccess,
            coordinationSuccess: coordinationSuccess,
            overallSuccess: overallSuccess,
            duration: duration,
            timestamp: Date()
        )
        
        print("[IntegratedCleanupService] Integrated cleanup completed in \(String(format: "%.2f", duration))s - Success: \(overallSuccess)")
        
        isCleanupInProgress = false
        return overallSuccess
    }
    
    /// 緊急時の強制クリーンアップ
    /// 通常のクリーンアップが失敗した場合に使用
    func performEmergencyCleanup() async {
        print("[IntegratedCleanupService] Performing emergency cleanup")
        
        isCleanupInProgress = true
        
        // 並行で強制停止を実行
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await self.cameraManager.forceReleaseResources()
            }
            
            group.addTask {
                _ = await self.audioFeedbackService.stopWithCleanup()
            }
        }
        
        // 協調クリーンアップのリセット
        coordinatedCleanup.reset()
        
        cleanupResult = CleanupResult(
            cameraSuccess: true,
            audioSuccess: true,
            coordinationSuccess: true,
            overallSuccess: true,
            duration: 0,
            timestamp: Date(),
            isEmergencyCleanup: true
        )
        
        isCleanupInProgress = false
        print("[IntegratedCleanupService] Emergency cleanup completed")
    }
    
    /// クリーンアップ状態のリセット
    func resetCleanupState() {
        cleanupResult = nil
        coordinatedCleanup.reset()
        setupCoordinatedCleanup()
        print("[IntegratedCleanupService] Cleanup state reset")
    }
    
    /// 最後のクリーンアップが成功したかをチェック
    var lastCleanupWasSuccessful: Bool {
        return cleanupResult?.overallSuccess ?? false
    }
    
    // MARK: - Private Methods
    
    /// 協調クリーンアップの設定
    private func setupCoordinatedCleanup() {
        // 統合サービス用のクリーンアップハンドラーを登録
        coordinatedCleanup.registerCleanupHandler("integration_check") {
            await self.performIntegrationCheck()
        }
    }
    
    /// 統合チェック処理
    private func performIntegrationCheck() async {
        print("[IntegratedCleanupService] Performing integration check")
        
        // カメラとオーディオの状態を確認
        let cameraRunning = cameraManager.isSessionRunning
        let audioPlaying = audioFeedbackService.currentlyPlaying
        
        if cameraRunning || audioPlaying {
            print("[IntegratedCleanupService] Warning: Some resources are still active (Camera: \(cameraRunning), Audio: \(audioPlaying))")
        } else {
            print("[IntegratedCleanupService] Integration check passed - all resources properly cleaned")
        }
    }
}

// MARK: - CleanupResult Model

/// クリーンアップ結果の詳細情報
struct CleanupResult {
    let cameraSuccess: Bool
    let audioSuccess: Bool
    let coordinationSuccess: Bool
    let overallSuccess: Bool
    let duration: TimeInterval
    let timestamp: Date
    let error: String?
    let isEmergencyCleanup: Bool
    
    init(
        cameraSuccess: Bool,
        audioSuccess: Bool,
        coordinationSuccess: Bool,
        overallSuccess: Bool,
        duration: TimeInterval,
        timestamp: Date,
        error: String? = nil,
        isEmergencyCleanup: Bool = false
    ) {
        self.cameraSuccess = cameraSuccess
        self.audioSuccess = audioSuccess
        self.coordinationSuccess = coordinationSuccess
        self.overallSuccess = overallSuccess
        self.duration = duration
        self.timestamp = timestamp
        self.error = error
        self.isEmergencyCleanup = isEmergencyCleanup
    }
    
    /// クリーンアップの成功率（0.0 - 1.0）
    var successRate: Double {
        let total = 3.0 // camera, audio, coordination
        var successful = 0.0
        
        if cameraSuccess { successful += 1.0 }
        if audioSuccess { successful += 1.0 }
        if coordinationSuccess { successful += 1.0 }
        
        return successful / total
    }
}

// MARK: - CleanupResult + CustomStringConvertible

extension CleanupResult: CustomStringConvertible {
    var description: String {
        let type = isEmergencyCleanup ? "Emergency" : "Standard"
        let rate = String(format: "%.1f", successRate * 100)
        
        if let error = error {
            return "\(type) Cleanup: \(rate)% success (\(String(format: "%.2f", duration))s) - Error: \(error)"
        } else {
            return "\(type) Cleanup: \(rate)% success (\(String(format: "%.2f", duration))s)"
        }
    }
}