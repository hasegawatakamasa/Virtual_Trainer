import Foundation
import Combine

/// 複数のリソースのクリーンアップを調整するコーディネーター
/// カメラセッション、音声フィードバック、その他のリソースの適切な解放を保証する
class ResourceCleanupCoordinator: ObservableObject {
    
    // MARK: - Types
    
    typealias CleanupHandler = () async -> Void
    
    // MARK: - Published Properties
    
    @Published private(set) var isCleanupInProgress = false
    @Published private(set) var cleanupStatus = CleanupStatus(
        cameraCleanupComplete: false,
        audioCleanupComplete: false,
        resourcesReleased: false,
        timestamp: Date()
    )
    
    // MARK: - Private Properties
    
    private var cleanupHandlers: [String: CleanupHandler] = [:]
    private var completionCallbacks: [String: Bool] = [:]
    private let cleanupQueue = DispatchQueue(label: "com.virtualtrainer.cleanup", qos: .userInitiated)
    
    // MARK: - Public Methods
    
    /// クリーンアップハンドラーを登録する
    /// - Parameters:
    ///   - identifier: ハンドラーの一意識別子
    ///   - handler: 実行するクリーンアップ処理
    func registerCleanupHandler(_ identifier: String, handler: @escaping CleanupHandler) {
        cleanupHandlers[identifier] = handler
        completionCallbacks[identifier] = false
    }
    
    /// すべてのリソースのクリーンアップを開始する
    /// - Returns: クリーンアップが完了した場合はtrue
    @MainActor
    func initiateCleanup() async -> Bool {
        guard !isCleanupInProgress else {
            print("[ResourceCleanupCoordinator] Cleanup already in progress")
            return false
        }
        
        isCleanupInProgress = true
        print("[ResourceCleanupCoordinator] Starting cleanup process with \(cleanupHandlers.count) handlers")
        
        // すべてのクリーンアップハンドラーを並行実行
        await withTaskGroup(of: (String, Bool).self) { group in
            for (identifier, handler) in cleanupHandlers {
                group.addTask {
                    await handler()
                    print("[ResourceCleanupCoordinator] Handler '\(identifier)' completed successfully")
                    return (identifier, true)
                }
            }
            
            // 結果を収集
            for await (identifier, success) in group {
                completionCallbacks[identifier] = success
            }
        }
        
        // ステータスを更新
        let cameraComplete = completionCallbacks["camera"] ?? false
        let audioComplete = completionCallbacks["audio"] ?? false
        let allComplete = completionCallbacks.values.allSatisfy { $0 }
        
        cleanupStatus = CleanupStatus(
            cameraCleanupComplete: cameraComplete,
            audioCleanupComplete: audioComplete,
            resourcesReleased: allComplete,
            timestamp: Date()
        )
        
        isCleanupInProgress = false
        
        print("[ResourceCleanupCoordinator] Cleanup completed. Status: \(cleanupStatus)")
        return allComplete
    }
    
    /// クリーンアップが完了しているかを確認する
    /// - Returns: 全てのリソースがクリーンアップされている場合はtrue
    func isCleanupComplete() -> Bool {
        return cleanupStatus.isFullyComplete && !isCleanupInProgress
    }
    
    /// 特定のハンドラーの完了状況を確認する
    /// - Parameter identifier: ハンドラー識別子
    /// - Returns: 指定されたハンドラーが完了している場合はtrue
    func isHandlerComplete(_ identifier: String) -> Bool {
        return completionCallbacks[identifier] ?? false
    }
    
    /// すべてのハンドラーと状態をリセットする
    /// 新しいセッション開始時に使用
    func reset() {
        cleanupHandlers.removeAll()
        completionCallbacks.removeAll()
        cleanupStatus = CleanupStatus(
            cameraCleanupComplete: false,
            audioCleanupComplete: false,
            resourcesReleased: false,
            timestamp: Date()
        )
        isCleanupInProgress = false
        print("[ResourceCleanupCoordinator] Reset completed")
    }
    
    // MARK: - Debug Methods
    
    /// デバッグ用: 現在の状態を出力する
    func printDebugStatus() {
        print("[ResourceCleanupCoordinator] Debug Status:")
        print("  - Cleanup in progress: \(isCleanupInProgress)")
        print("  - Registered handlers: \(cleanupHandlers.keys.sorted())")
        print("  - Completion status: \(completionCallbacks)")
        print("  - Overall status: \(cleanupStatus)")
    }
}

// MARK: - CleanupStatus Model

/// クリーンアップの進行状況を追跡する構造体
struct CleanupStatus {
    let cameraCleanupComplete: Bool
    let audioCleanupComplete: Bool
    let resourcesReleased: Bool
    let timestamp: Date
    
    /// 全てのクリーンアップが完了しているかを確認
    var isFullyComplete: Bool {
        return cameraCleanupComplete && audioCleanupComplete && resourcesReleased
    }
    
    /// 完了したタスクの割合（0.0 - 1.0）
    var completionRatio: Double {
        let completed = [cameraCleanupComplete, audioCleanupComplete, resourcesReleased].filter { $0 }.count
        return Double(completed) / 3.0
    }
}

// MARK: - CleanupStatus + CustomStringConvertible

extension CleanupStatus: CustomStringConvertible {
    var description: String {
        return "CleanupStatus(camera: \(cameraCleanupComplete), audio: \(audioCleanupComplete), resources: \(resourcesReleased), complete: \(isFullyComplete))"
    }
}