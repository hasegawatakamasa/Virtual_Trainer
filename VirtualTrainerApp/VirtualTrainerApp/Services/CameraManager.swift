import Foundation
import AVFoundation
import Combine
import UIKit

/// カメラ出力を受け取るためのデリゲート
protocol CameraOutputDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer)
    func cameraManager(_ manager: CameraManager, didEncounterError error: AppError)
}

/// カメラセッション管理クラス
class CameraManager: NSObject, ObservableObject {
    // MARK: - Published Properties
    @Published var isSessionRunning = false
    @Published var cameraPosition: AVCaptureDevice.Position = .front
    @Published var permissionStatus: AVAuthorizationStatus = .notDetermined
    @Published var currentError: AppError?
    
    // MARK: - Private Properties
    private let captureSession = AVCaptureSession()
    private var videoDeviceInput: AVCaptureDeviceInput?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue", qos: .userInitiated)
    private let dataOutputQueue = DispatchQueue(label: "camera.data.output.queue", qos: .userInteractive)
    
    weak var delegate: CameraOutputDelegate?
    private let cleanupCoordinator = ResourceCleanupCoordinator()
    
    // MARK: - Initialization
    override init() {
        super.init()
        setupSession()
        observePermissionStatus()
        registerCleanupHandlers()
    }
    
    // MARK: - Public Methods
    
    /// カメラ権限を要求
    func requestCameraPermission() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        
        switch status {
        case .authorized:
            await MainActor.run {
                permissionStatus = .authorized
            }
            return true
            
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            await MainActor.run {
                permissionStatus = granted ? .authorized : .denied
            }
            return granted
            
        case .denied, .restricted:
            await MainActor.run {
                permissionStatus = .denied
            }
            return false
            
        @unknown default:
            await MainActor.run {
                permissionStatus = .denied
            }
            return false
        }
    }
    
    /// セッション開始
    func startSession() {
        guard permissionStatus == .authorized else {
            handleError(.cameraPermissionDenied)
            return
        }
        
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.captureSession.startRunning()
            
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.isSessionRunning = self.captureSession.isRunning
            }
        }
    }
    
    /// セッション停止（改善版）
    /// ResourceCleanupCoordinatorを使用した確実なリソース解放
    func stopSession() async {
        await stopSessionWithCleanup()
    }
    
    /// 同期版セッション停止（後方互換性のため保持）
    func stopSessionSync() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
            
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.isSessionRunning = false
            }
        }
    }
    
    /// クリーンアップ付きセッション停止
    private func stopSessionWithCleanup() async {
        print("[CameraManager] Starting session cleanup process")
        
        // クリーンアップ開始
        let cleanupSuccess = await cleanupCoordinator.initiateCleanup()
        
        if !cleanupSuccess {
            handleError(.cameraUnavailable)
            // エラーが発生してもセッション停止は継続
        }
        
        await MainActor.run {
            isSessionRunning = false
        }
        
        print("[CameraManager] Session cleanup completed successfully: \(cleanupSuccess)")
    }
    
    /// カメラ切り替え
    func switchCamera() {
        guard permissionStatus == .authorized else { return }
        
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            let wasRunning = self.captureSession.isRunning
            self.captureSession.beginConfiguration()
            
            // 現在の入力を削除
            if let currentInput = self.videoDeviceInput {
                self.captureSession.removeInput(currentInput)
            }
            
            // 新しいポジションを設定
            let newPosition: AVCaptureDevice.Position = self.cameraPosition == .front ? .back : .front
            
            do {
                let newInput = try self.createCameraInput(for: newPosition)
                if self.captureSession.canAddInput(newInput) {
                    self.captureSession.addInput(newInput)
                    self.videoDeviceInput = newInput
                    
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        self.cameraPosition = newPosition
                        var settings = AppSettings.shared
                        settings.cameraPosition = newPosition
                    }
                } else {
                    throw AppError.cameraUnavailable
                }
            } catch {
                self.handleError(error as? AppError ?? .cameraUnavailable)
                
                // 元の入力を復元
                if let oldInput = self.videoDeviceInput,
                   self.captureSession.canAddInput(oldInput) {
                    self.captureSession.addInput(oldInput)
                }
            }
            
            self.captureSession.commitConfiguration()
            
            if wasRunning && !self.captureSession.isRunning {
                self.captureSession.startRunning()
            }
        }
    }
    
    /// プレビューレイヤーを作成
    func createPreviewLayer() -> AVCaptureVideoPreviewLayer {
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        return previewLayer
    }
}

// MARK: - Private Methods
private extension CameraManager {
    
    func setupSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.captureSession.beginConfiguration()
            
            // セッション品質を設定
            if self.captureSession.canSetSessionPreset(.high) {
                self.captureSession.sessionPreset = .high
            }
            
            do {
                // カメラ入力を設定
                let position = self.cameraPosition
                let cameraInput = try self.createCameraInput(for: position)
                if self.captureSession.canAddInput(cameraInput) {
                    self.captureSession.addInput(cameraInput)
                    self.videoDeviceInput = cameraInput
                } else {
                    throw AppError.cameraUnavailable
                }
                
                // ビデオ出力を設定
                self.setupVideoDataOutput()
                
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.handleError(error as? AppError ?? .cameraUnavailable)
                }
            }
            
            self.captureSession.commitConfiguration()
        }
    }
    
    func createCameraInput(for position: AVCaptureDevice.Position) throws -> AVCaptureDeviceInput {
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            throw AppError.cameraUnavailable
        }
        
        // フレームレート設定
        do {
            try camera.lockForConfiguration()
            
            // 15FPSに設定（パフォーマンス重視）
            let targetFPS = AppSettings.shared.targetFPS
            let desiredFrameRate = CMTime(value: 1, timescale: CMTimeScale(targetFPS))
            
            if camera.activeFormat.videoSupportedFrameRateRanges.contains(where: {
                $0.minFrameRate <= targetFPS && targetFPS <= $0.maxFrameRate
            }) {
                camera.activeVideoMinFrameDuration = desiredFrameRate
                camera.activeVideoMaxFrameDuration = desiredFrameRate
            }
            
            camera.unlockForConfiguration()
        } catch {
            // フレームレート設定は失敗してもカメラは使用可能
            print("⚠️ フレームレート設定に失敗: \(error)")
        }
        
        return try AVCaptureDeviceInput(device: camera)
    }
    
    func setupVideoDataOutput() {
        videoDataOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.setSampleBufferDelegate(self, queue: dataOutputQueue)
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
            
            // ビデオ出力の向きを設定
            if let connection = videoDataOutput.connection(with: .video) {
                // iOS 17以降の新しいAPIを使用
                if #available(iOS 17.0, *) {
                    if connection.isVideoRotationAngleSupported(90) {
                        connection.videoRotationAngle = 90
                    }
                } else {
                    // iOS 17未満の場合は従来のAPIを使用
                    if connection.isVideoOrientationSupported {
                        connection.videoOrientation = .portrait
                    }
                }
                
                // フロントカメラの場合はミラーリング
                if cameraPosition == .front && connection.isVideoMirroringSupported {
                    connection.isVideoMirrored = true
                }
            }
        }
    }
    
    func observePermissionStatus() {
        permissionStatus = AVCaptureDevice.authorizationStatus(for: .video)
    }
    
    func handleError(_ error: AppError) {
        DispatchQueue.main.async { [weak self] in
            self?.currentError = error
        }
        delegate?.cameraManager(self, didEncounterError: error)
        
        // 新しいエラー回復処理を追加
        Task {
            await handleCameraError(error)
        }
    }
    
    /// カメラエラーの回復処理
    private func handleCameraError(_ error: AppError) async {
        let cameraError: CameraSessionError
        
        switch error {
        case .cameraPermissionDenied:
            cameraError = .permissionDenied
        case .cameraUnavailable:
            cameraError = .sessionNotFound
        default:
            cameraError = .sessionFailedToStop
        }
        
        await ErrorRecoveryManager.shared.handleCameraError(cameraError)
    }
    
    /// クリーンアップハンドラーの登録
    private func registerCleanupHandlers() {
        cleanupCoordinator.registerCleanupHandler("camera") { [weak self] in
            await self?.performCameraCleanup()
        }
    }
    
    /// カメラリソースのクリーンアップ処理
    private func performCameraCleanup() async {
        return await withCheckedContinuation { continuation in
            sessionQueue.async { [weak self] in
                guard let self = self else {
                    continuation.resume()
                    return
                }
                
                print("[CameraManager] Performing camera cleanup")
                
                // セッション停止
                if self.captureSession.isRunning {
                    self.captureSession.stopRunning()
                }
                
                // 入力出力の削除
                self.captureSession.beginConfiguration()
                
                if let videoInput = self.videoDeviceInput {
                    self.captureSession.removeInput(videoInput)
                    self.videoDeviceInput = nil
                }
                
                if self.captureSession.outputs.contains(self.videoDataOutput) {
                    self.captureSession.removeOutput(self.videoDataOutput)
                }
                
                self.captureSession.commitConfiguration()
                
                print("[CameraManager] Camera cleanup completed")
                continuation.resume()
            }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // デリゲートに通知（バックグラウンドキュー）
        delegate?.cameraManager(self, didOutput: pixelBuffer)
    }
    
    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // フレームドロップは通常の動作（パフォーマンス維持のため）
        // ログ出力は行わない（頻繁すぎるため）
    }
}

// MARK: - Lifecycle Management
extension CameraManager {
    
    /// アプリがバックグラウンドに移行した時の処理（強化版）
    func handleAppDidEnterBackground() {
        Task {
            await stopSession()
            print("[CameraManager] Background cleanup completed successfully")
        }
    }
    
    /// アプリがフォアグラウンドに復帰した時の処理（強化版）
    func handleAppWillEnterForeground() {
        Task {
            // クリーンアップ状態をリセット
            await resetCleanupState()
            
            if permissionStatus == .authorized {
                startSession()
                print("[CameraManager] Foreground restoration completed successfully")
            } else {
                print("[CameraManager] Camera permission not authorized, skipping session start")
            }
        }
    }
    
    /// クリーンアップ状態のリセット
    private func resetCleanupState() async {
        cleanupCoordinator.reset()
        registerCleanupHandlers()
        print("[CameraManager] Cleanup state reset completed")
    }
    
    /// 緊急時のリソース強制解放
    func forceReleaseResources() async {
        print("[CameraManager] Executing force resource release")
        
        await performCameraCleanup()
        cleanupCoordinator.reset()
        
        await MainActor.run {
            isSessionRunning = false
            currentError = nil
        }
        
        print("[CameraManager] Force resource release completed")
    }
}