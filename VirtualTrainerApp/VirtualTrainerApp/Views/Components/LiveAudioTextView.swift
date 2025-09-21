import SwiftUI
import Combine

/// 音声再生中にリアルタイムでテキストを表示するコンポーネント
/// DisplayStateと連携して他のUI要素との表示優先度を管理
struct LiveAudioTextView: View {
    
    // MARK: - Properties
    
    @StateObject private var audioTextQueue = AudioTextQueue()
    @State private var currentDisplayState: DisplayState = .none
    @State private var animationOffset: CGSize = .zero
    @State private var animationScale: CGFloat = 1.0
    @State private var animationOpacity: Double = 0.0
    
    // MARK: - Body
    
    var body: some View {
        ZStack {
            if let currentText = audioTextQueue.currentText {
                audioTextView(for: currentText)
                    .transition(transition(for: currentText.animationConfig.entryAnimation))
            }
        }
        .animation(.easeInOut(duration: 0.3), value: audioTextQueue.currentText?.id)
        .onReceive(audioTextQueue.$currentText) { audioText in
            updateDisplayState(for: audioText)
        }
    }
    
    // MARK: - Private Methods
    
    /// 音声テキスト表示ビューを作成
    @ViewBuilder
    private func audioTextView(for audioText: AudioTextData) -> some View {
        VStack(spacing: 12) {
            // キャラクター名表示
            characterNameView(for: audioText.character)
            
            // メインテキスト表示
            mainTextView(for: audioText)
                .overlay(
                    // プログレスバー
                    progressOverlay(for: audioText),
                    alignment: .bottom
                )
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
        .background(backgroundView(for: audioText))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.3), radius: 8, x: 0, y: 4)
        .scaleEffect(animationScale)
        .offset(animationOffset)
        .opacity(animationOpacity)
        .onAppear {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                applyEntryAnimation(audioText.animationConfig.entryAnimation)
            }
        }
        .onDisappear {
            withAnimation(.easeOut(duration: 0.3)) {
                applyExitAnimation(audioText.animationConfig.exitAnimation)
            }
        }
    }
    
    /// キャラクター名表示
    @ViewBuilder
    private func characterNameView(for character: VoiceCharacter) -> some View {
        HStack(spacing: 8) {
            Text(character.displayName)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.white.opacity(0.9))
            
            Spacer()
            
            // 音声再生インジケーター
            audioIndicatorView()
        }
    }
    
    /// メインテキスト表示
    @ViewBuilder
    private func mainTextView(for audioText: AudioTextData) -> some View {
        Text(audioText.displayText)
            .font(.title3)
            .fontWeight(audioText.animationConfig.emphasis ? .bold : .semibold)
            .foregroundColor(.white)
            .multilineTextAlignment(.center)
            .lineLimit(3)
            .minimumScaleFactor(0.8)
    }
    
    /// 背景ビュー
    @ViewBuilder
    private func backgroundView(for audioText: AudioTextData) -> some View {
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
    }
    
    /// プログレスオーバーレイ
    @ViewBuilder
    private func progressOverlay(for audioText: AudioTextData) -> some View {
        GeometryReader { geometry in
            Rectangle()
                .fill(Color.white.opacity(0.2))
                .frame(width: geometry.size.width * (1.0 - audioText.progress))
                .animation(.linear(duration: 0.1), value: audioText.progress)
        }
        .frame(height: 2)
        .cornerRadius(1)
    }
    
    /// 音声再生インジケーター
    @ViewBuilder
    private func audioIndicatorView() -> some View {
        HStack(spacing: 3) {
            ForEach(0..<3, id: \.self) { index in
                RoundedRectangle(cornerRadius: 1)
                    .fill(Color.white.opacity(0.8))
                    .frame(width: 3, height: 8)
                    .scaleEffect(y: audioWaveScale(for: index))
                    .animation(
                        .easeInOut(duration: 0.5)
                            .repeatForever(autoreverses: true)
                            .delay(Double(index) * 0.1),
                        value: audioTextQueue.currentText?.id
                    )
            }
        }
    }
    
    /// 音声波形スケールを計算
    private func audioWaveScale(for index: Int) -> CGFloat {
        guard audioTextQueue.currentText?.isCurrentlyActive == true else { return 0.3 }
        
        let baseScale: CGFloat = 0.5
        let amplitude: CGFloat = 0.5
        let offset = Double(index) * 0.3
        
        return baseScale + amplitude * CGFloat(sin(Date().timeIntervalSince1970 * 3 + offset))
    }
    
    /// 表示状態を更新
    private func updateDisplayState(for audioText: AudioTextData?) {
        let newState: DisplayState = audioText?.isCurrentlyActive == true ? .liveAudioText : .none
        
        if newState != currentDisplayState {
            currentDisplayState = newState
        }
    }
    
    /// 遷移アニメーションを作成
    private func transition(for animation: AudioTextAnimation) -> AnyTransition {
        switch animation {
        case .fadeIn:
            return .opacity
        case .slideIn:
            return .move(edge: .bottom).combined(with: .opacity)
        case .bounceIn:
            return .scale.combined(with: .opacity)
        case .shake:
            return .offset(x: 10).combined(with: .opacity)
        default:
            return .opacity
        }
    }
    
    /// エントリーアニメーションを適用
    private func applyEntryAnimation(_ animation: AudioTextAnimation) {
        switch animation {
        case .fadeIn:
            animationOpacity = 1.0
        case .slideIn:
            animationOffset = .zero
            animationOpacity = 1.0
        case .bounceIn:
            animationScale = 1.0
            animationOpacity = 1.0
        case .shake:
            animationOffset = .zero
            animationOpacity = 1.0
            // シェイクアニメーション
            withAnimation(.easeInOut(duration: 0.1).repeatCount(3, autoreverses: true)) {
                animationOffset = CGSize(width: 5, height: 0)
            }
        case .fadeOut, .slideUp:
            animationOpacity = 1.0
        }
    }
    
    /// エグジットアニメーションを適用
    private func applyExitAnimation(_ animation: AudioTextAnimation) {
        switch animation {
        case .fadeOut:
            animationOpacity = 0.0
        case .slideUp:
            animationOffset = CGSize(width: 0, height: -50)
            animationOpacity = 0.0
        default:
            animationOpacity = 0.0
        }
    }
    
    // MARK: - Public Methods
    
    /// 音声テキストを表示キューに追加
    func displayAudioText(_ audioText: AudioTextData) {
        audioTextQueue.enqueue(audioText)
    }
    
    /// 現在の表示をクリア
    func clearCurrentDisplay() {
        audioTextQueue.clearCurrentText()
    }
    
    /// すべてのキューをクリア
    func clearAllDisplays() {
        audioTextQueue.clearAll()
    }
    
    /// 現在の表示状態を取得
    var displayState: DisplayState {
        return currentDisplayState
    }
}

// MARK: - Color Extension

extension Color {
    /// HEX文字列からColorを初期化
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
        }
        
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue:  Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

// MARK: - Preview

struct LiveAudioTextView_Previews: PreviewProvider {
    static var previews: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            LiveAudioTextView()
        }
        .onAppear {
            // プレビュー用のテストデータ（使用しない）
            _ = AudioTextData(
                text: "フォームを確認してください！",
                character: .zundamon,
                audioType: .formError,
                estimatedDuration: 3.0
            )
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                // プレビューでの表示テスト
                print("Displaying test audio text")
            }
        }
    }
}