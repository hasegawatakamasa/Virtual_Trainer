import SwiftUI

/// カメラプレビュー上に表示するフィードバックオーバーレイ
struct FeedbackOverlayView: View {
    @ObservedObject var formAnalyzer: FormAnalyzer
    @ObservedObject var repCounter: RepCounterManager
    @ObservedObject var mlModelManager: MLModelManager
    @ObservedObject var audioFeedbackService: AudioFeedbackService
    @State private var lastFormState: FormClassification = .ready
    @State private var showingZoneChange = false
    @State private var lastDetectedKeypoints: PoseKeypoints?
    
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
                topFeedbackArea
                
                Spacer()
                
                // 下部：回数とセッション情報
                bottomInfoArea
            }
            .padding()
            
            // デバッグ情報（デバッグモード時のみ）
            if AppSettings.shared.showDebugInfo {
                debugOverlay
            }
        }
    }
    
    // MARK: - Top Feedback Area
    
    private var topFeedbackArea: some View {
        VStack(spacing: 12) {
            // 音声フィードバック状態インジケーター
            if audioFeedbackService.currentlyPlaying {
                audioFeedbackIndicator
                    .transition(.opacity.combined(with: .scale))
            }
            
            // フォーム状態の大きな表示
            formStatusDisplay
            
            // エクササイズゾーン状態
            if formAnalyzer.isInExerciseZone || showingZoneChange {
                exerciseZoneIndicator
                    .transition(.opacity.combined(with: .scale))
            }
        }
    }
    
    private var audioFeedbackIndicator: some View {
        HStack(spacing: 8) {
            Image(systemName: "speaker.wave.2.fill")
                .font(.caption)
                .foregroundColor(.orange)
            
            Text("音声指導中")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.black.opacity(0.6))
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
    
    private var exerciseZoneIndicator: some View {
        HStack {
            Image(systemName: "target")
                .foregroundColor(.green)
            
            Text("エクササイズゾーン")
                .font(.subheadline)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.green.opacity(0.3))
        .cornerRadius(15)
        .onAppear {
            // ゾーン入場時のアニメーション
            withAnimation(.easeInOut(duration: 0.5)) {
                showingZoneChange = true
            }
            
            // 一定時間後に非表示
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                withAnimation(.easeOut(duration: 0.3)) {
                    showingZoneChange = false
                }
            }
        }
    }
    
    // MARK: - Bottom Info Area
    
    private var bottomInfoArea: some View {
        HStack {
            // 回数表示
            repCountDisplay
            
            Spacer()
            
            // セッション情報
            sessionInfoDisplay
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
    
    private var sessionInfoDisplay: some View {
        VStack(alignment: .trailing, spacing: 4) {
            Text("セッション")
                .font(.caption)
                .foregroundColor(.white.opacity(0.8))
            
            Text(sessionDurationText)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.white)
            
            if repCounter.repState.count > 0 {
                Text("\(String(format: "%.1f", repCounter.averageRepsPerMinute)) 回/分")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
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
        if !formAnalyzer.isInExerciseZone {
            return "準備中"
        }
        
        // TODO: 実際のフォーム分類結果を使用
        return "フォーム監視中"
    }
    
    private var formStateIcon: String {
        if !formAnalyzer.isInExerciseZone {
            return "person.crop.circle"
        }
        
        // TODO: フォーム分類結果に基づいてアイコンを変更
        return "checkmark.circle.fill"
    }
    
    private var formStateColor: (red: Double, green: Double, blue: Double) {
        if !formAnalyzer.isInExerciseZone {
            return (red: 1.0, green: 1.0, blue: 1.0) // 白
        }
        
        // TODO: フォーム分類結果に基づいて色を変更
        return (red: 0.0, green: 1.0, blue: 0.0) // 緑
    }
    
    private var sessionDurationText: String {
        let duration = repCounter.repState.sessionDuration
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
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