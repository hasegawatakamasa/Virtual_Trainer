import SwiftUI
import AVFoundation
import UIKit

/// AVFoundationのプレビューレイヤーをSwiftUIで使用するためのラッパー
struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager
    
    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.setupPreviewLayer(with: cameraManager)
        return view
    }
    
    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        // カメラマネージャーが変更された場合の処理
        uiView.updatePreviewLayer(with: cameraManager)
    }
}

/// カメラプレビューを表示するUIView
class CameraPreviewUIView: UIView {
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
    
    func setupPreviewLayer(with cameraManager: CameraManager) {
        guard previewLayer == nil else { return }
        
        let previewLayer = cameraManager.createPreviewLayer()
        previewLayer.frame = bounds
        previewLayer.videoGravity = .resizeAspectFill
        
        layer.addSublayer(previewLayer)
        self.previewLayer = previewLayer
    }
    
    func updatePreviewLayer(with cameraManager: CameraManager) {
        // 必要に応じてプレビューレイヤーの更新処理を実装
        previewLayer?.frame = bounds
    }
}

/// カメラプレビューとオーバーレイを含むメインビュー
struct CameraViewContainer: View {
    @StateObject private var cameraManager = CameraManager()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            switch cameraManager.permissionStatus {
            case .notDetermined, .denied, .restricted:
                // 権限要求画面
                PermissionView(cameraManager: cameraManager) {
                    // 権限が許可されたら自動的にExerciseTrainingViewに遷移
                }
                
            case .authorized:
                // メインエクササイズビュー
                ExerciseTrainingView()
                
            @unknown default:
                // 未知の状態の場合は権限要求画面を表示
                PermissionView(cameraManager: cameraManager) {
                    // 権限が許可されたら自動的にExerciseTrainingViewに遷移
                }
            }
        }
        .onAppear {
            setupInitialState()
        }
    }
    
    private func setupInitialState() {
        // デフォルト値を登録
        UserDefaultsKeys.registerDefaults()
    }
}

// MARK: - Preview
#Preview("権限許可済み") {
    CameraViewContainer()
}

#Preview("権限要求") {
    let manager = CameraManager()
    return PermissionView(cameraManager: manager) {
        print("Permission granted")
    }
}