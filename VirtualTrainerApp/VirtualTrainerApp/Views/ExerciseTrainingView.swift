import SwiftUI
import AVFoundation
import Combine
#if canImport(UIKit)
import UIKit
#endif

/// シートタイプの定義
enum SheetType: Identifiable {
    case settings
    case result

    var id: String {
        switch self {
        case .settings: return "settings"
        case .result: return "result"
        }
    }
}

/// エクササイズトレーニングのメインビュー
struct ExerciseTrainingView: View {
    let exerciseType: ExerciseType
    let onCompletion: ((SessionCompletionData) -> Void)?

    @StateObject private var cameraManager = CameraManager()
    @StateObject private var formAnalyzer: FormAnalyzer
    @StateObject private var repCounter: RepCounterManager
    @StateObject private var mlModelManager = MLModelManager()
    @StateObject private var audioFeedbackService = AudioFeedbackService()
    @StateObject private var trainingSessionService = TrainingSessionService.shared
    @StateObject private var achievementSystem = AchievementSystem.shared
    @StateObject private var oshiReactionManager = OshiReactionManager.shared
    @StateObject private var sessionTimer = SessionTimerManager()
    @StateObject private var completionCoordinator = SessionCompletionCoordinator()
    @StateObject private var interruptionHandler = InterruptionHandler.shared
    @State private var isProcessing = false
    @State private var cancellables = Set<AnyCancellable>()
    @State private var showingSettings = false
    @State private var showStartMessage = false
    @State private var showFinishMessage = false
    @State private var showingResultView = false
    @State private var sessionCompletionData: SessionCompletionData?
    @State private var cameraOutputHandler = CameraOutputHandler()
    @State private var lastProcessingTime = Date()
    @Environment(\.dismiss) private var dismiss
    
    // 統合クリーンアップサービス（遅延初期化）
    private var integratedCleanupService: IntegratedCleanupService {
        IntegratedCleanupService(cameraManager: cameraManager, audioFeedbackService: audioFeedbackService)
    }
    
    // デフォルトイニシャライザー（既存コードとの互換性）
    init(exerciseType: ExerciseType = .overheadPress, onCompletion: ((SessionCompletionData) -> Void)? = nil) {
        self.exerciseType = exerciseType
        self.onCompletion = onCompletion
        self._formAnalyzer = StateObject(wrappedValue: FormAnalyzer(exerciseType: exerciseType))
        self._repCounter = StateObject(wrappedValue: RepCounterManager(exerciseType: exerciseType))
    }
    
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
                mlModelManager: mlModelManager,
                audioFeedbackService: audioFeedbackService
            )

            // 種目名表示とタイマー表示
            VStack {
                Text(exerciseType.displayName)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(20)
                    .padding(.top, 80)

                // タイマー表示（waitingForRep状態以降で表示）
                if sessionTimer.timerState == .waitingForRep ||
                   sessionTimer.timerState == .showingStart ||
                   sessionTimer.timerState == .running ||
                   sessionTimer.timerState == .paused ||
                   sessionTimer.timerState == .completed {
                    TimerDisplayView(
                        remainingTime: sessionTimer.remainingTime,
                        isLastTenSeconds: sessionTimer.isLastTenSeconds,
                        timerState: sessionTimer.timerState,
                        canStartManually: sessionTimer.canStartManually,
                        onManualStart: {
                            sessionTimer.manualStart()
                        }
                    )
                    .padding(.horizontal)
                    .padding(.top, 20)
                }

                Spacer()
            }

            // コントロールUI
            controlOverlay

            // 開始メッセージオーバーレイ
            if showStartMessage || sessionTimer.showStartMessage {
                StartMessageOverlay(
                    isShowing: .constant(true),
                    message: "トレーニング開始！",
                    displayDuration: 2.0,
                    onComplete: nil
                )
            }

            // 終了メッセージオーバーレイ
            if showFinishMessage || completionCoordinator.showFinishMessage {
                FinishOverlayView(
                    isShowing: .constant(true),
                    completedReps: repCounter.repState.count,
                    trainingTime: TimeInterval(60 - sessionTimer.remainingTime),
                    customMessage: nil,
                    autoDismissDelay: nil,  // 自動非表示を無効化（手動で閉じる必要）
                    onComplete: {
                        // リザルトボタンが押された時の処理
                        // 完了コーディネーターのリザルト画面遷移メソッドを呼び出す
                        completionCoordinator.navigateToResultScreen()
                    }
                )
            }
        }
        .onAppear {
            print("🎥 ExerciseTrainingView appeared for exercise: \(exerciseType.displayName)")
            setupServices()
        }
        .onDisappear {
            cleanup()
        }
        .sheet(item: .constant(showingSettings ? SheetType.settings : showingResultView ? SheetType.result : nil)) { sheetType in
            switch sheetType {
            case .settings:
                ExerciseSettingsView()
            case .result:
                if let completionData = sessionCompletionData {
                    SessionResultView(completionData: completionData)
                }
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: Notification.Name("ReturnToHome"))) { _ in
            // ホームに戻る - ExerciseTrainingViewを閉じる
            dismiss()
        }
    }
    
    // MARK: - Control Overlay
    
    private var controlOverlay: some View {
        VStack {
            // 上部コントロール
            HStack {
                // 戻るボタン
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundColor(.white)
                        .background(Color.black.opacity(0.3))
                        .clipShape(Circle())
                }
                
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
            VStack(spacing: 12) {
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
                
            }
            .padding()
        }
    }
    
    // MARK: - Setup and Cleanup
    
    private func setupServices() {
        // AudioFeedbackServiceとTrainingSessionServiceを接続
        audioFeedbackService.trainingSessionService = trainingSessionService

        // タイマーと完了コーディネーターの設定
        completionCoordinator.configure(
            cleanupService: integratedCleanupService,
            sessionService: trainingSessionService
        )

        // InterruptionHandlerの設定
        interruptionHandler.configureWithTimerManager(sessionTimer)
        // SessionServiceの設定は自動的にsharedインスタンスを使用

        // カメラマネージャーのデリゲート設定
        cameraOutputHandler.processFrameCallback = { pixelBuffer in
            await self.processFrame(pixelBuffer: pixelBuffer)
        }
        cameraManager.delegate = cameraOutputHandler

        // トレーニングセッション開始
        let currentCharacter = VoiceCharacter(rawValue: UserDefaults.standard.string(forKey: "selectedCharacter") ?? "ずんだもん") ?? .zundamon
        repCounter.startTrainingSession(with: currentCharacter)

        // タイマーを待機状態にする
        sessionTimer.startWaitingForRep()
        
        // カメラ権限を要求してセッション開始（少し遅延を入れてUI初期化完了を待つ）
        Task {
            print("🎥 ExerciseTrainingView: カメラ初期化を開始")
            // UI初期化完了を待つ
            try? await Task.sleep(nanoseconds: 500_000_000) // 0.5秒
            
            print("🎥 カメラ権限を要求中...")
            let granted = await cameraManager.requestCameraPermission()
            print("🎥 カメラ権限結果: \(granted)")
            
            if granted {
                print("🎥 カメラセッション開始中...")
                // セッション開始も少し遅延
                try? await Task.sleep(nanoseconds: 100_000_000) // 0.1秒
                cameraManager.startSession()
                print("🎥 カメラセッション開始完了")
            } else {
                print("❌ カメラ権限が拒否されました")
            }
        }
        
        // 回数カウントイベントの監視
        repCounter.eventPublisher
            .receive(on: DispatchQueue.main)
            .sink { event in
                print("📌 RepCounter event received: \(event)")

                // 最初のレップでタイマーを開始（待機状態の場合のみ）
                if case .repCompleted(let count) = event {
                    print("📌 Rep completed: \(count), Timer state: \(sessionTimer.timerState.displayName)")
                    if sessionTimer.timerState == .waitingForRep {
                        print("🎯 Starting timer on first rep!")
                        sessionTimer.handleFirstRep()
                        // トレーニング開始音声を再生（1回目のカウントより優先）
                        audioFeedbackService.playTimerStart()
                        // 1回目のカウント音声は再生しないため、振動フィードバックのみ
                        #if os(iOS)
                        let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
                        impactFeedback.impactOccurred()
                        #endif
                        return  // 1回目はhandleRepCountEventを呼ばない
                    }
                }

                // その他のイベントまたは2回目以降のレップカウント
                handleRepCountEvent(event)
            }
            .store(in: &cancellables)

        // タイマー完了の監視
        sessionTimer.$timerState
            .receive(on: DispatchQueue.main)
            .sink { state in
                if state == .completed {
                    handleTimerCompletion()
                }
            }
            .store(in: &cancellables)

        // タイマーマイルストーンの監視
        sessionTimer.$remainingTime
            .receive(on: DispatchQueue.main)
            .sink { [weak audioFeedbackService] time in
                if let milestone = TimerMilestone.milestoneForRemainingTime(TimeInterval(time)) {
                    audioFeedbackService?.playTimerMilestone(milestone)
                }
            }
            .store(in: &cancellables)
            
        // フォーム分類結果の監視と音声フィードバック処理
        mlModelManager.$lastFormResult
            .compactMap { $0 }
            .receive(on: DispatchQueue.main)
            .sink { [weak audioFeedbackService, weak formAnalyzer] formResult in
                // エクササイズゾーン内でのみ音声フィードバックを有効にする
                let isInZone = formAnalyzer?.isInExerciseZone ?? false
                audioFeedbackService?.processFormResult(formResult, isInExerciseZone: isInZone)
            }
            .store(in: &cancellables)
            
        // アプリライフサイクルの監視
        #if os(iOS)
        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .sink { _ in
                cameraManager.handleAppDidEnterBackground()
            }
            .store(in: &cancellables)

        // セッション再開の監視（メモリ警告などで新しいセッションが開始された場合）
        NotificationCenter.default.publisher(for: Notification.Name("SessionRestarted"))
            .sink { _ in
                print("📱 Session restarted notification received")

                // RepCounterをリセットして新しいセッションを開始
                repCounter.reset()
                let currentCharacter = VoiceCharacter(rawValue: UserDefaults.standard.string(forKey: "selectedCharacter") ?? "ずんだもん") ?? .zundamon
                repCounter.startTrainingSession(with: currentCharacter)

                // タイマーをリセットして待機状態に
                sessionTimer.reset()
                sessionTimer.startWaitingForRep()

                print("⏰ Timer reset complete - state: \(sessionTimer.timerState.displayName)")
                print("⏰ Timer is now in waitingForRep: \(sessionTimer.timerState == .waitingForRep)")
                print("⏰ Remaining time: \(sessionTimer.remainingTime) seconds")

                // RepCounterのイベント購読を再設定（重要！）
                repCounter.eventPublisher
                    .receive(on: DispatchQueue.main)
                    .sink { event in
                        print("📌 [Resubscribed] RepCounter event: \(event)")

                        // 最初のレップでタイマーを開始
                        if case .repCompleted(let count) = event {
                            print("📌 [Resubscribed] Rep completed: \(count), Timer state: \(sessionTimer.timerState.displayName)")
                            if sessionTimer.timerState == .waitingForRep {
                                print("🎯 [Resubscribed] Starting timer on first rep!")
                                sessionTimer.handleFirstRep()
                                // トレーニング開始音声を再生（1回目のカウントより優先）
                                audioFeedbackService.playTimerStart()
                                // 1回目のカウント音声は再生しないため、振動フィードバックのみ
                                #if os(iOS)
                                let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
                                impactFeedback.impactOccurred()
                                #endif
                                return  // 1回目はhandleRepCountEventを呼ばない
                            }
                        }

                        // その他のイベントまたは2回目以降のレップカウント
                        handleRepCountEvent(event)
                    }
                    .store(in: &cancellables)

                print("✅ RepCounter event subscription re-established")
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .sink { _ in
                cameraManager.handleAppWillEnterForeground()
            }
            .store(in: &cancellables)
        #else
        print("[ExerciseTrainingView] Background/Foreground notifications not configured for macOS")
        #endif
    }
    
    private func cleanup() {
        // タイマーを停止
        sessionTimer.stopTimer()

        // トレーニングセッション終了とアチーブメント評価
        if let sessionSummary = trainingSessionService.getSessionSummary() {
            achievementSystem.evaluateAchievements(for: sessionSummary)
            oshiReactionManager.checkNewRecord(session: sessionSummary)
            oshiReactionManager.checkMilestone(session: sessionSummary)
        }

        // 通常のセッション終了処理
        repCounter.endCurrentSession()

        Task {
            let cleanupSuccess = await integratedCleanupService.performIntegratedCleanup()
            if !cleanupSuccess {
                print("[ExerciseTrainingView] Standard cleanup failed, performing emergency cleanup")
                await integratedCleanupService.performEmergencyCleanup()
            }
        }
        cancellables.removeAll()
    }

    private func handleTimerCompletion() {
        print("[ExerciseTrainingView] Timer completed, initiating completion sequence")

        // 注: タイマー終了音声は、SessionTimerManagerの残り時間監視で
        // 自動的に再生されるため、ここでは再生しない

        // 完了データを作成
        let completionData = SessionCompletionData(
            startTime: Date().addingTimeInterval(-60),  // 60秒前
            endTime: Date(),
            configuredDuration: 60,
            actualDuration: 60,
            completedReps: repCounter.repState.count,
            completionReason: .timerCompleted,
            formErrorCount: formAnalyzer.errorCount,
            speedWarningCount: trainingSessionService.sessionStats.speedWarnings,
            averageRepsPerMinute: Double(repCounter.repState.count),
            maxConsecutiveCorrectReps: repCounter.repState.count,  // TODO: 正確な連続カウントを追跡
            voiceCharacter: UserDefaults.standard.string(forKey: "selectedCharacter") ?? "ずんだもん",
            exerciseType: exerciseType.displayName
        )

        // 完了処理を開始
        completionCoordinator.initiateCompletion(with: completionData)

        // ナビゲーション監視
        completionCoordinator.$shouldNavigateToResult
            .filter { $0 }
            .sink { _ in
                // リザルト画面への遷移（ExerciseTrainingView内で表示）
                if let completionData = completionCoordinator.completionData {
                    sessionCompletionData = completionData
                    showingResultView = true
                }
            }
            .store(in: &cancellables)
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
            #if os(iOS)
            let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
            impactFeedback.impactOccurred()
            #endif
            
            // 回数カウント音声の再生
            audioFeedbackService.playRepCountAudio(count: count)
            
            if AppSettings.shared.debugMode {
                print("✅ Rep completed: \(count)")
            }
            
        case .stateChanged(_, _):
            break // RepCounterManager内で既にログ出力済み
            
        case .zoneEntered:
            // ゾーン入場時の軽い振動
            #if os(iOS)
            let selectionFeedback = UISelectionFeedbackGenerator()
            selectionFeedback.selectionChanged()
            #endif
            
        case .zoneExited:
            break // 特別な処理なし
            
        case .sessionReset:
            // リセット時の通知フィードバック
            #if os(iOS)
            let notificationFeedback = UINotificationFeedbackGenerator()
            notificationFeedback.notificationOccurred(.success)
            #endif
            
        case .speedFeedbackNeeded(let speed):
            // 速度フィードバック音声の再生
            audioFeedbackService.playSpeedFeedback(speed)
            // 実際に再生されたことをSpeedAnalyzerに記録
            repCounter.speedAnalyzer.recordFeedbackPlayed()
            
            if AppSettings.shared.debugMode {
                print("🏃 Speed feedback triggered: \(speed.displayName)")
            }
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
        let result = await withTaskGroup(of: (PoseKeypoints?, FormClassification.Result?, TimeInterval).self) { group in
            let startTime = CFAbsoluteTimeGetCurrent()
            
            group.addTask {
                // 姿勢検出を実行
                let poseKeypoints = await self.mlModelManager.detectPose(in: pixelBuffer)
                let formClassificationResult = await self.mlModelManager.classifyForm(features: [])
                let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
                return (poseKeypoints, formClassificationResult, inferenceTime)
            }
            
            return await group.next() ?? (nil, nil, 0.0)
        }
        
        // メインアクターでUI更新
        await MainActor.run {
            let (poseKeypoints, formClassificationResult, inferenceTime) = result
            
            self.mlModelManager.updatePerformanceMetrics(inferenceTime: inferenceTime)
            
            if let poseKeypoints = poseKeypoints,
               let filteredKeypoints = FilteredKeypoints(from: poseKeypoints) {
                // 実際のAI結果を使用
                let analysisResult = self.formAnalyzer.analyzeForm(keypoints: filteredKeypoints)
                // FormClassification.ResultからFormClassificationを取り出す
                var formClassification = formClassificationResult?.classification ?? .normal
                
                // フォーム分析結果からエラー検出を強化
                if formClassification == .normal {
                    // 角度に基づく簡易的なフォームエラー検出
                    let elbowAngle = analysisResult.elbowAngle
                    
                    // オーバーヘッドプレスの適正角度範囲を超えているかチェック
                    if elbowAngle < 45 || elbowAngle > 180 {
                        formClassification = .elbowError
                        print("🔍 Detected elbow error: angle = \(String(format: "%.1f", elbowAngle))°")
                    }
                }
                
                self.repCounter.updateState(analysisResult: analysisResult, formClassification: formClassification)
            }
        }
    }
    
}

// MARK: - Settings View
struct ExerciseSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var audioFeedbackService = AudioFeedbackService()
    @StateObject private var voiceSettings = VoiceSettings.shared
    @AppStorage("debugMode") private var debugMode = false
    @AppStorage("showDebugInfo") private var showDebugInfo = false
    @AppStorage("topThreshold") private var topThreshold = 130.0
    @AppStorage("bottomThreshold") private var bottomThreshold = 100.0
    
    var body: some View {
        NavigationView {
            Form {
                Section("音声フィードバック") {
                    Toggle("フォーム指導音声", isOn: $audioFeedbackService.isAudioEnabled)
                    
                    if !audioFeedbackService.isAudioEnabled {
                        Text("音声フィードバック機能が無効になっています")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Button("音声テスト") {
                        testAudioFeedback()
                    }
                    .disabled(!audioFeedbackService.isAudioEnabled || audioFeedbackService.currentlyPlaying)
                }
                
                Section("ボイスキャラクター") {
                    Picker("音声キャラクター", selection: $voiceSettings.selectedCharacter) {
                        ForEach(VoiceCharacter.allCases) { character in
                            HStack {
                                Image(systemName: character.iconName)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(character.displayName)
                                        .font(.subheadline)
                                    Text(character.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .tag(character)
                        }
                    }
                    #if os(iOS)
                    .pickerStyle(.navigationLink)
                    #endif
                    
                    Button("キャラクター音声テスト") {
                        testCharacterVoice()
                    }
                    .disabled(!audioFeedbackService.isAudioEnabled || audioFeedbackService.currentlyPlaying)
                }
                
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
                
                Section(footer: voicevoxCreditFooter) {
                    HStack {
                        Text("バージョン")
                        Spacer()
                        Text(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0")
                            .foregroundColor(.secondary)
                    }
                    
                    Link("ライセンス情報", destination: URL(string: "https://voicevox.hiroshiba.jp/")!)
                        .foregroundColor(.blue)
                }
            }
            .navigationTitle("設定")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: {
                    #if os(iOS)
                    return .navigationBarTrailing
                    #else
                    return .primaryAction
                    #endif
                }()) {
                    Button("完了") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func testAudioFeedback() {
        // フォームエラー音声のテスト再生
        let testResult = FormClassification.Result(
            classification: .elbowError,
            confidence: 0.9
        )
        audioFeedbackService.processFormResult(testResult, isInExerciseZone: true)
    }
    
    private func testCharacterVoice() {
        // 速度フィードバック音声のテスト再生（励まし音声）
        audioFeedbackService.playSpeedFeedback(.slow)
    }
    
    private var voicevoxCreditFooter: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("音声合成: VOICEVOX (ずんだもん)")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text("VOICEVOX:ずんだもん")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text("本アプリで使用している音声は、VOICEVOXを使用して生成されています。")
                .font(.caption)
                .foregroundColor(.secondary)
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