import SwiftUI

/// カメラプレビュー上に表示するフィードバックオーバーレイ
struct FeedbackOverlayView: View {
    @ObservedObject var formAnalyzer: FormAnalyzer
    @ObservedObject var repCounter: RepCounterManager
    @ObservedObject var mlModelManager: MLModelManager
    @ObservedObject var audioFeedbackService: AudioFeedbackService
    @State private var lastFormState: FormClassification = .ready
    @State private var lastDetectedKeypoints: PoseKeypoints?
    
    // DisplayState統合
    @State private var currentDisplayStates: Set<DisplayState> = []
    @StateObject private var audioTextQueue = AudioTextQueue()
    
    var body: some View {
        ZStack {
            // キーポイント可視化（設定で有効な場合）
            if AppSettings.shared.showKeypoints {
                GeometryReader { geometry in
                    KeypointOverlayView(
                        keypoints: mlModelManager.lastDetectedPose,
                        viewSize: geometry.size
                    )
                }
                .ignoresSafeArea()
            }
            
            // メインフィードバック表示
            VStack {
                // 上部：フォーム状態表示
                if shouldShow(.formMonitoring) {
                    topFeedbackArea
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }
                
                Spacer()
                
                // 中央：ライブ音声テキスト表示
                if shouldShow(.liveAudioText), let currentText = audioTextQueue.currentText {
                    liveAudioTextDisplay(for: currentText)
                        .transition(.scale.combined(with: .opacity))
                }
                
                Spacer()
                
                // 下部：回数とセッション情報
                if shouldShow(.exerciseZone) {
                    bottomInfoArea
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }
            .padding()
            .animation(.easeInOut(duration: 0.3), value: currentDisplayStates)
            
            // デバッグ情報（デバッグモード時のみ）
            if AppSettings.shared.showDebugInfo {
                debugOverlay
            }
        }
        .onAppear {
            updateDisplayStates()
        }
        .onReceive(formAnalyzer.$isInExerciseZone) { _ in
            updateDisplayStates()
        }
        .onReceive(audioFeedbackService.$currentlyPlaying) { _ in
            updateDisplayStates()
        }
    }
    
    // MARK: - Top Feedback Area
    
    private var topFeedbackArea: some View {
        VStack(spacing: 12) {
            // 速度フィードバック表示（デバッグモード時のみ）
            if AppSettings.shared.showDebugInfo {
                speedFeedbackIndicator
            }
            
            // フォーム状態の大きな表示
            formStatusDisplay
        }
    }
    
    
    private var speedFeedbackIndicator: some View {
        HStack(spacing: 8) {
            Image(systemName: "speedometer")
                .font(.caption)
                .foregroundColor(.blue)
            
            Text("速度フィードバック有効")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.blue.opacity(0.3))
        .cornerRadius(16)
    }
    
    private var formStatusDisplay: some View {
        HStack {
            // フォーム状態のアイコン
            Image(systemName: formStateIcon)
                .font(.title)
                .foregroundColor(Color(red: formStateColor.red, 
                                     green: formStateColor.green, 
                                     blue: formStateColor.blue))
            
            // フォーム状態のテキスト
            Text(formStateText)
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color.black.opacity(0.6))
        .cornerRadius(25)
        .scaleEffect(formAnalyzer.isInExerciseZone ? 1.1 : 1.0)
        .animation(.easeInOut(duration: 0.3), value: formAnalyzer.isInExerciseZone)
    }
    
    
    // MARK: - Bottom Info Area
    
    private var bottomInfoArea: some View {
        HStack {
            // 回数表示のみ（セッション情報は削除）
            repCountDisplay

            Spacer()
        }
    }
    
    private var repCountDisplay: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("回数")
                .font(.caption)
                .foregroundColor(.white.opacity(0.8))
            
            HStack(alignment: .bottom, spacing: 4) {
                Text("\(repCounter.repState.count)")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .contentTransition(.numericText())
                    .animation(.easeOut(duration: 0.3), value: repCounter.repState.count)
                
                Text("回")
                    .font(.title3)
                    .foregroundColor(.white.opacity(0.8))
            }
            
            // 状態表示
            Text(repCounter.repState.state.description)
                .font(.caption)
                .foregroundColor(repCounter.repState.isInZone ? .green : .gray)
        }
        .padding(16)
        .background(Color.black.opacity(0.6))
        .cornerRadius(16)
    }
    
    
    // MARK: - Debug Overlay
    
    private var debugOverlay: some View {
        VStack {
            HStack {
                Spacer()
                
                // デバッグ情報を右上に配置
                VStack(alignment: .trailing, spacing: 4) {
                    Text("デバッグ情報")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.yellow)
                    
                    Group {
                        Text("角度: \(String(format: "%.1f", formAnalyzer.currentAngle))°")
                        Text("ゾーン: \(formAnalyzer.isInExerciseZone ? "内" : "外")")
                        Text("状態: \(repCounter.repState.state.description)")
                        Text("推論: \(String(format: "%.0fms", mlModelManager.lastInferenceTime * 1000))")
                        Text("AI: \(mlModelManager.isModelLoaded ? "有効" : "模擬")")
                    }
                    .font(.caption)
                    .foregroundColor(.white)
                }
                .padding(12)
                .background(Color.black.opacity(0.8))
                .cornerRadius(8)
            }
            .padding(.horizontal)
            .padding(.top, 100) // 上部のフォーム状態表示と重ならないように
            
            Spacer()
        }
    }
    
    // MARK: - Computed Properties
    
    private var formStateText: String {
        // エクササイズゾーン外の場合は準備中
        if !formAnalyzer.isInExerciseZone {
            return "準備中"
        }
        
        // エクササイズゾーン内の場合は適切に状態を更新
        // TODO: 実際のフォーム分類結果に基づく詳細な状態管理
        return "フォーム監視中"
    }
    
    private var formStateIcon: String {
        if !formAnalyzer.isInExerciseZone {
            return "person.crop.circle"
        }
        
        // エクササイズゾーン内の場合は監視中アイコン
        return "eye.circle.fill"
    }
    
    private var formStateColor: (red: Double, green: Double, blue: Double) {
        if !formAnalyzer.isInExerciseZone {
            return (red: 1.0, green: 1.0, blue: 1.0) // 白（準備中）
        }
        
        // エクササイズゾーン内の場合は監視中の色（青）
        return (red: 0.0, green: 0.7, blue: 1.0) // 青（監視中）
    }
    
    
    // MARK: - Live Audio Text Display
    
    /// ライブ音声テキスト表示
    @ViewBuilder
    private func liveAudioTextDisplay(for audioText: AudioTextData) -> some View {
        VStack(spacing: 12) {
            // キャラクター名表示
            HStack(spacing: 8) {
                Text(audioText.character.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white.opacity(0.9))
                
                Spacer()
                
                // 音声再生インジケーター
                HStack(spacing: 3) {
                    ForEach(0..<3, id: \.self) { index in
                        RoundedRectangle(cornerRadius: 1)
                            .fill(Color.white.opacity(0.8))
                            .frame(width: 3, height: 8)
                            .scaleEffect(y: 0.5 + 0.5 * CGFloat(sin(Date().timeIntervalSince1970 * 3 + Double(index) * 0.3)))
                            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: audioText.id)
                    }
                }
            }
            
            // メインテキスト表示（5m離れた位置から見やすいよう大きめに）
            Text(audioText.displayText)
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .minimumScaleFactor(0.7)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color(hex: audioText.displayColor).opacity(0.9),
                            Color(hex: audioText.displayColor).opacity(0.7)
                        ]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.white.opacity(0.3), lineWidth: 1)
                )
        )
        .shadow(color: .black.opacity(0.3), radius: 8, x: 0, y: 4)
    }
    
    // MARK: - DisplayState Management
    
    /// 指定されたDisplayStateを表示すべきかを判定
    private func shouldShow(_ state: DisplayState) -> Bool {
        switch state {
        case .liveAudioText:
            // ライブ音声テキストは音声再生中のみ表示
            return audioFeedbackService.currentlyPlaying && shouldShowAudioText()
            
        case .formMonitoring:
            // フォーム監視状態は常に表示（音声再生の有無に関係なく）
            return true
            
        case .exerciseZone:
            // エクササイズゾーン内の場合は回数表示エリアを常に表示
            return formAnalyzer.isInExerciseZone
            
        default:
            return false
        }
    }
    
    /// 現在の表示状態を更新（onAppear/onReceiveで呼び出し）
    private func updateDisplayStates() {
        DispatchQueue.main.async {
            var newStates: Set<DisplayState> = []
            
            // フォーム監視状態（常に有効）
            newStates.insert(.formMonitoring)
            
            // エクササイズゾーン状態
            if formAnalyzer.isInExerciseZone {
                newStates.insert(.exerciseZone)
            }
            
            // ライブ音声テキスト状態（表示すべき音声タイプの場合のみ）
            if audioFeedbackService.currentlyPlaying && shouldShowAudioText() {
                newStates.insert(.liveAudioText)
                
                // 音声テキストデータを作成してキューに渡す
                if let audioText = createAudioTextData() {
                    audioTextQueue.enqueue(audioText)
                }
            }
            
            if newStates != currentDisplayStates {
                currentDisplayStates = newStates
            }
        }
    }
    
    /// 音声テキストを表示すべきかを判定
    private func shouldShowAudioText() -> Bool {
        guard let currentTaskType = audioFeedbackService.currentAudioType else { return false }
        
        // カウント音声以外のみテキスト表示
        switch currentTaskType {
        case .repCount:
            return false // カウント音声は表示しない
        case .speedFeedback, .formError:
            return true // 速度フィードバックとフォームエラーは表示
        case .timerMilestone, .timerStart:
            return false // タイマー関連は表示しない
        }
    }
    
    /// 現在の音声状態からAudioTextDataを作成
    private func createAudioTextData() -> AudioTextData? {
        guard audioFeedbackService.currentlyPlaying,
              let currentTaskType = audioFeedbackService.currentAudioType else { return nil }
        
        let character = VoiceSettings.shared.selectedCharacter
        
        // 音声タイプに応じてテキストと表示可否を判定
        switch currentTaskType {
        case .speedFeedback:
            // 速度フィードバックタイプに応じてテキスト表示
            guard let speedFeedbackType = audioFeedbackService.currentSpeedFeedbackType else { return nil }
            
            let audioType: AudioType = speedFeedbackType == .tooSlow ? .slowEncouragement : .fastWarning
            let text = speedFeedbackType.displayText
            
            return AudioTextData(
                text: text,
                character: character,
                audioType: audioType,
                estimatedDuration: 2.0,
                isActive: true
            )
            
        case .formError:
            // フォームエラーの場合もテキスト表示
            return AudioTextData(
                text: "フォームを確認してください",
                character: character,
                audioType: .formError,
                estimatedDuration: 3.0,
                isActive: true
            )
            
        case .repCount:
            // カウント音声の場合は表示しない
            return nil
        case .timerMilestone, .timerStart:
            // タイマー関連は表示しない
            return nil
        }
    }
}

// MARK: - Preview
#Preview("Active Session") {
    ZStack {
        Color.black.ignoresSafeArea()
        
        FeedbackOverlayView(
            formAnalyzer: {
                let analyzer = FormAnalyzer()
                analyzer.currentAngle = 95.5
                analyzer.isInExerciseZone = true
                return analyzer
            }(),
            repCounter: {
                let counter = RepCounterManager()
                counter.repState.count = 12
                counter.repState.state = .bottom
                return counter
            }(),
            mlModelManager: MLModelManager(),
            audioFeedbackService: AudioFeedbackService()
        )
    }
}

#Preview("Debug Mode") {
    ZStack {
        Color.black.ignoresSafeArea()
        
        FeedbackOverlayView(
            formAnalyzer: {
                let analyzer = FormAnalyzer()
                analyzer.currentAngle = 125.0
                analyzer.isInExerciseZone = true
                return analyzer
            }(),
            repCounter: {
                let counter = RepCounterManager()
                counter.repState.count = 5
                return counter
            }(),
            mlModelManager: MLModelManager(),
            audioFeedbackService: AudioFeedbackService()
        )
        .onAppear {
            var settings = AppSettings.shared
            settings.debugMode = true
            settings.showDebugInfo = true
        }
    }
}