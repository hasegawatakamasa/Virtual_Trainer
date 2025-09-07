import SwiftUI
import AVFoundation
import Combine

/// エクササイズトレーニングのメインビュー
struct ExerciseTrainingView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var formAnalyzer = FormAnalyzer()
    @StateObject private var repCounter = RepCounterManager()
    @StateObject private var mlModelManager = MLModelManager()
    @State private var isProcessing = false
    @State private var cancellables = Set<AnyCancellable>()
    @State private var showingSettings = false
    @State private var cameraOutputHandler = CameraOutputHandler()
    @State private var lastProcessingTime = Date()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // カメラプレビュー
            CameraPreviewView(cameraManager: cameraManager)
                .ignoresSafeArea()
            
            // フィードバックオーバーレイ
            FeedbackOverlayView(
                formAnalyzer: formAnalyzer,
                repCounter: repCounter,
                mlModelManager: mlModelManager
            )
            
            // コントロールUI
            controlOverlay
        }
        .onAppear {
            setupServices()
        }
        .onDisappear {
            cleanup()
        }
        .sheet(isPresented: $showingSettings) {
            ExerciseSettingsView()
        }
    }
    
    // MARK: - Control Overlay
    
    private var controlOverlay: some View {
        VStack {
            // 上部コントロール
            HStack {
                // 設定ボタン
                Button(action: { showingSettings = true }) {
                    Image(systemName: "gearshape.fill")
                        .font(.title2)
                        .foregroundColor(.white)
                        .background(Color.black.opacity(0.3))
                        .clipShape(Circle())
                }
                
                Spacer()
                
                // カメラ切り替え
                Button(action: { cameraManager.switchCamera() }) {
                    Image(systemName: "camera.rotate.fill")
                        .font(.title2)
                        .foregroundColor(.white)
                        .background(Color.black.opacity(0.3))
                        .clipShape(Circle())
                }
            }
            .padding()
            
            Spacer()
            
            // 下部コントロール
            HStack {
                // リセットボタン
                Button(action: { resetSession() }) {
                    Label("リセット", systemImage: "arrow.counterclockwise")
                        .font(.subheadline)
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.red.opacity(0.6))
                        .cornerRadius(20)
                }
                
                Spacer()
                
                // 手動カウントボタン（デバッグモード時のみ）
                if AppSettings.shared.debugMode {
                    Button(action: { manualCount() }) {
                        Label("+1", systemImage: "plus.circle.fill")
                            .font(.subheadline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.blue.opacity(0.6))
                            .cornerRadius(20)
                    }
                }
            }
            .padding()
        }
    }
    
    // MARK: - Setup and Cleanup
    
    private func setupServices() {
        // カメラマネージャーのデリゲート設定
        cameraOutputHandler.processFrameCallback = { pixelBuffer in
            await self.processFrame(pixelBuffer: pixelBuffer)
        }
        cameraManager.delegate = cameraOutputHandler
        
        // カメラ権限を要求してセッション開始（少し遅延を入れてUI初期化完了を待つ）
        Task {
            // UI初期化完了を待つ
            try? await Task.sleep(nanoseconds: 500_000_000) // 0.5秒
            
            let granted = await cameraManager.requestCameraPermission()
            if granted {
                // セッション開始も少し遅延
                try? await Task.sleep(nanoseconds: 100_000_000) // 0.1秒
                cameraManager.startSession()
            }
        }
        
        // 回数カウントイベントの監視
        repCounter.eventPublisher
            .receive(on: DispatchQueue.main)
            .sink { event in
                handleRepCountEvent(event)
            }
            .store(in: &cancellables)
            
        // アプリライフサイクルの監視
        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .sink { _ in
                cameraManager.handleAppDidEnterBackground()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .sink { _ in
                cameraManager.handleAppWillEnterForeground()
            }
            .store(in: &cancellables)
    }
    
    private func cleanup() {
        cameraManager.stopSession()
        cancellables.removeAll()
    }
    
    // MARK: - Actions
    
    private func resetSession() {
        withAnimation(.easeInOut(duration: 0.3)) {
            repCounter.reset()
        }
    }
    
    private func manualCount() {
        let currentAngle = formAnalyzer.currentAngle
        repCounter.incrementCount(angle: currentAngle, formClassification: .normal)
    }
    
    private func handleRepCountEvent(_ event: RepCountEvent) {
        switch event {
        case .repCompleted(let count):
            // 回数完了時の振動フィードバック
            let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
            impactFeedback.impactOccurred()
            
            if AppSettings.shared.debugMode {
                print("✅ Rep completed: \(count)")
            }
            
        case .stateChanged(_, _):
            break // RepCounterManager内で既にログ出力済み
            
        case .zoneEntered:
            // ゾーン入場時の軽い振動
            let selectionFeedback = UISelectionFeedbackGenerator()
            selectionFeedback.selectionChanged()
            
        case .zoneExited:
            break // 特別な処理なし
            
        case .sessionReset:
            // リセット時の通知フィードバック
            let notificationFeedback = UINotificationFeedbackGenerator()
            notificationFeedback.notificationOccurred(.success)
        }
    }
}

// MARK: - Camera Delegate Handler
final class CameraOutputHandler: NSObject, CameraOutputDelegate, @unchecked Sendable {
    var processFrameCallback: ((CVPixelBuffer) async -> Void)?
    
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer) {
        Task { @MainActor in
            // CVPixelBufferを直接MainActorコンテキストで処理
            await processFrameCallback?(pixelBuffer)
        }
    }
    
    func cameraManager(_ manager: CameraManager, didEncounterError error: AppError) {
        Task { @MainActor in
            // エラーハンドリング（現在は基本的な表示のみ）
            print("❌ Camera error: \(error.localizedDescription)")
        }
    }
}

// MARK: - Frame Processing
extension ExerciseTrainingView {
    
    @MainActor
    func processFrame(pixelBuffer: CVPixelBuffer) async {
        // フレームレート制限 (最大10FPS)
        let now = Date()
        let minInterval: TimeInterval = 1.0/10.0  // 100ms間隔
        
        // 処理中または制限時間内の場合はスキップ
        if isProcessing || now.timeIntervalSince(lastProcessingTime) < minInterval {
            return 
        }
        
        isProcessing = true
        lastProcessingTime = now
        
        // 実際のAI推論を実行（非同期で処理完了まで待たない）
        Task {
            await performAIAnalysis(pixelBuffer: pixelBuffer)
            await MainActor.run {
                self.isProcessing = false
            }
        }
    }
    
    private func performAIAnalysis(pixelBuffer: CVPixelBuffer) async {
        // AIモデルが読み込まれていない場合はスキップ
        guard mlModelManager.isModelLoaded else { return }
        
        // AI推論を背景キューで実行
        let result = await withTaskGroup(of: (PoseKeypoints?, FormClassification?, TimeInterval).self) { group in
            let startTime = CFAbsoluteTimeGetCurrent()
            
            group.addTask {
                // 姿勢検出を実行
                let poseKeypoints = await self.mlModelManager.detectPose(in: pixelBuffer)
                let formClassification = await self.mlModelManager.classifyForm(features: [])
                let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
                return (poseKeypoints, formClassification, inferenceTime)
            }
            
            return await group.next() ?? (nil, nil, 0.0)
        }
        
        // メインアクターでUI更新
        await MainActor.run {
            let (poseKeypoints, formClassification, inferenceTime) = result
            
            self.mlModelManager.updatePerformanceMetrics(inferenceTime: inferenceTime)
            
            if let poseKeypoints = poseKeypoints,
               let filteredKeypoints = FilteredKeypoints(from: poseKeypoints) {
                // 実際のAI結果を使用
                let analysisResult = self.formAnalyzer.analyzeForm(keypoints: filteredKeypoints)
                self.repCounter.updateState(analysisResult: analysisResult, formClassification: formClassification)
            }
        }
    }
    
}

// MARK: - Settings View
struct ExerciseSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage("debugMode") private var debugMode = false
    @AppStorage("showDebugInfo") private var showDebugInfo = false
    @AppStorage("topThreshold") private var topThreshold = 130.0
    @AppStorage("bottomThreshold") private var bottomThreshold = 100.0
    
    var body: some View {
        NavigationView {
            Form {
                Section("デバッグ") {
                    Toggle("デバッグモード", isOn: $debugMode)
                    Toggle("デバッグ情報表示", isOn: $showDebugInfo)
                        .disabled(!debugMode)
                }
                
                Section("エクササイズ設定") {
                    HStack {
                        Text("上位置閾値")
                        Spacer()
                        Text("\(Int(topThreshold))°")
                    }
                    Slider(value: $topThreshold, in: 120...150, step: 5)
                    
                    HStack {
                        Text("下位置閾値")
                        Spacer()
                        Text("\(Int(bottomThreshold))°")
                    }
                    Slider(value: $bottomThreshold, in: 80...110, step: 5)
                }
                
                Section("アプリ情報") {
                    HStack {
                        Text("バージョン")
                        Spacer()
                        Text(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("設定")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完了") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Sendable Extensions
// CVPixelBufferの並行性警告を抑制（Apple公式推奨）
extension CVPixelBuffer: @retroactive @unchecked Sendable {}

// MARK: - Preview
#Preview {
    ExerciseTrainingView()
}