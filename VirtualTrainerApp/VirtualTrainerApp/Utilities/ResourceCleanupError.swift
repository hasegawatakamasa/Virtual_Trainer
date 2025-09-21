import Foundation

// MARK: - Camera Session Errors

/// カメラセッション関連のエラー
enum CameraSessionError: Error, LocalizedError {
    case sessionFailedToStop
    case resourcesNotReleased
    case forceCleanupRequired
    case sessionNotFound
    case permissionDenied
    
    var errorDescription: String? {
        switch self {
        case .sessionFailedToStop:
            return "カメラセッションの停止に失敗しました"
        case .resourcesNotReleased:
            return "カメラリソースの解放が完了していません"
        case .forceCleanupRequired:
            return "強制クリーンアップが必要です"
        case .sessionNotFound:
            return "カメラセッションが見つかりません"
        case .permissionDenied:
            return "カメラのアクセス許可が拒否されています"
        }
    }
    
    var recoverySuggestion: String? {
        switch self {
        case .sessionFailedToStop:
            return "強制停止処理を試行してください"
        case .resourcesNotReleased:
            return "アプリケーションを再起動してください"
        case .forceCleanupRequired:
            return "緊急クリーンアップ処理を実行してください"
        case .sessionNotFound:
            return "カメラセッションを再初期化してください"
        case .permissionDenied:
            return "設定でカメラのアクセス許可を確認してください"
        }
    }
}

// MARK: - Audio Feedback Errors

/// 音声フィードバッククリーンアップ関連のエラー
enum AudioCleanupError: Error, LocalizedError {
    case audioStopFailed
    case playerNotReleased
    case sessionDeactivationFailed
    case audioFileNotFound
    case playbackInterrupted
    
    var errorDescription: String? {
        switch self {
        case .audioStopFailed:
            return "音声の停止に失敗しました"
        case .playerNotReleased:
            return "オーディオプレーヤーの解放に失敗しました"
        case .sessionDeactivationFailed:
            return "オーディオセッションの無効化に失敗しました"
        case .audioFileNotFound:
            return "音声ファイルが見つかりません"
        case .playbackInterrupted:
            return "音声の再生が中断されました"
        }
    }
    
    var recoverySuggestion: String? {
        switch self {
        case .audioStopFailed:
            return "強制的に音声プレーヤーを停止してください"
        case .playerNotReleased:
            return "メモリクリーンアップを実行してください"
        case .sessionDeactivationFailed:
            return "オーディオセッションを再初期化してください"
        case .audioFileNotFound:
            return "音声ファイルの存在を確認してください"
        case .playbackInterrupted:
            return "音声の再生を再試行してください"
        }
    }
}

// MARK: - Resource Cleanup Errors

/// リソースクリーンアップ関連のエラー
enum ResourceCleanupError: Error, LocalizedError {
    case cleanupTimeoutExceeded
    case partialCleanupFailure(details: [String])
    case coordinatorNotInitialized
    case handlerRegistrationFailed
    
    var errorDescription: String? {
        switch self {
        case .cleanupTimeoutExceeded:
            return "クリーンアップ処理がタイムアウトしました"
        case .partialCleanupFailure(let details):
            return "一部のクリーンアップに失敗しました: \(details.joined(separator: ", "))"
        case .coordinatorNotInitialized:
            return "リソースクリーンアップコーディネーターが初期化されていません"
        case .handlerRegistrationFailed:
            return "クリーンアップハンドラーの登録に失敗しました"
        }
    }
    
    var recoverySuggestion: String? {
        switch self {
        case .cleanupTimeoutExceeded:
            return "強制クリーンアップ処理を実行してください"
        case .partialCleanupFailure:
            return "失敗したクリーンアップを個別に再実行してください"
        case .coordinatorNotInitialized:
            return "コーディネーターを再初期化してください"
        case .handlerRegistrationFailed:
            return "ハンドラーの登録を再試行してください"
        }
    }
}

// MARK: - Error Recovery Manager

/// エラー回復処理を管理するクラス
class ErrorRecoveryManager {
    
    // MARK: - Singleton
    
    static let shared = ErrorRecoveryManager()
    private init() {}
    
    // MARK: - Camera Error Recovery
    
    /// カメラセッションエラーの回復処理
    /// - Parameter error: 回復対象のエラー
    func handleCameraError(_ error: CameraSessionError) async {
        print("[ErrorRecoveryManager] Handling camera error: \(error)")
        
        switch error {
        case .sessionFailedToStop:
            await attemptForceStop()
        case .resourcesNotReleased:
            await forceResourceRelease()
        case .forceCleanupRequired:
            await executeEmergencyCleanup()
        case .sessionNotFound:
            await reinitializeCamera()
        case .permissionDenied:
            await handlePermissionDenied()
        }
    }
    
    /// 音声フィードバックエラーの回復処理
    /// - Parameter error: 回復対象のエラー
    func handleAudioError(_ error: AudioCleanupError) async {
        print("[ErrorRecoveryManager] Handling audio error: \(error)")
        
        switch error {
        case .audioStopFailed:
            await forceStopAllAudio()
        case .playerNotReleased:
            await forceReleaseAudioPlayers()
        case .sessionDeactivationFailed:
            await reinitializeAudioSession()
        case .audioFileNotFound:
            await handleMissingAudioFile()
        case .playbackInterrupted:
            await handlePlaybackInterruption()
        }
    }
    
    /// リソースクリーンアップエラーの回復処理
    /// - Parameter error: 回復対象のエラー
    func handleResourceCleanupError(_ error: ResourceCleanupError) async {
        print("[ErrorRecoveryManager] Handling resource cleanup error: \(error)")
        
        switch error {
        case .cleanupTimeoutExceeded:
            await executeTimeoutRecovery()
        case .partialCleanupFailure(let details):
            await retryFailedCleanup(details: details)
        case .coordinatorNotInitialized:
            await reinitializeCoordinator()
        case .handlerRegistrationFailed:
            await retryHandlerRegistration()
        }
    }
    
    // MARK: - Private Recovery Methods
    
    private func attemptForceStop() async {
        print("[ErrorRecoveryManager] Attempting force stop...")
        // Force stop implementation would go here
    }
    
    private func forceResourceRelease() async {
        print("[ErrorRecoveryManager] Force releasing resources...")
        // Force release implementation would go here
    }
    
    private func executeEmergencyCleanup() async {
        print("[ErrorRecoveryManager] Executing emergency cleanup...")
        // Emergency cleanup implementation would go here
    }
    
    private func reinitializeCamera() async {
        print("[ErrorRecoveryManager] Reinitializing camera...")
        // Camera reinitialization implementation would go here
    }
    
    private func handlePermissionDenied() async {
        print("[ErrorRecoveryManager] Handling permission denied...")
        // Permission handling implementation would go here
    }
    
    private func forceStopAllAudio() async {
        print("[ErrorRecoveryManager] Force stopping all audio...")
        // Force stop audio implementation would go here
    }
    
    private func forceReleaseAudioPlayers() async {
        print("[ErrorRecoveryManager] Force releasing audio players...")
        // Force release players implementation would go here
    }
    
    private func reinitializeAudioSession() async {
        print("[ErrorRecoveryManager] Reinitializing audio session...")
        // Audio session reinitialization implementation would go here
    }
    
    private func handleMissingAudioFile() async {
        print("[ErrorRecoveryManager] Handling missing audio file...")
        // Missing file handling implementation would go here
    }
    
    private func handlePlaybackInterruption() async {
        print("[ErrorRecoveryManager] Handling playback interruption...")
        // Interruption handling implementation would go here
    }
    
    private func executeTimeoutRecovery() async {
        print("[ErrorRecoveryManager] Executing timeout recovery...")
        // Timeout recovery implementation would go here
    }
    
    private func retryFailedCleanup(details: [String]) async {
        print("[ErrorRecoveryManager] Retrying failed cleanup: \(details)")
        // Failed cleanup retry implementation would go here
    }
    
    private func reinitializeCoordinator() async {
        print("[ErrorRecoveryManager] Reinitializing coordinator...")
        // Coordinator reinitialization implementation would go here
    }
    
    private func retryHandlerRegistration() async {
        print("[ErrorRecoveryManager] Retrying handler registration...")
        // Handler registration retry implementation would go here
    }
}

// MARK: - Error Extensions

extension Error {
    /// エラーを適切なカテゴリのエラー回復処理に振り分ける
    func handleRecovery() async {
        let manager = ErrorRecoveryManager.shared
        
        switch self {
        case let cameraError as CameraSessionError:
            await manager.handleCameraError(cameraError)
        case let audioError as AudioCleanupError:
            await manager.handleAudioError(audioError)
        case let cleanupError as ResourceCleanupError:
            await manager.handleResourceCleanupError(cleanupError)
        default:
            print("[ErrorRecoveryManager] Unhandled error type: \(self)")
        }
    }
}