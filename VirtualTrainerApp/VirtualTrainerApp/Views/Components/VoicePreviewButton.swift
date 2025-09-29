import SwiftUI

/// 音声プレビューボタンコンポーネント
struct VoicePreviewButton: View {
    let trainer: OshiTrainer

    @ObservedObject private var previewService = VoicePreviewService.shared
    @State private var isAnimating: Bool = false

    var body: some View {
        Button(action: {
            previewService.playRandomPreview(for: trainer)
        }) {
            HStack(spacing: 8) {
                Image(systemName: "waveform")
                    .font(.system(size: 16, weight: .medium))
                Text("声を聞く")
                    .font(.system(size: 14, weight: .medium))
            }
            .foregroundColor(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color.accentColor)
                    .opacity(previewService.isPlaying ? 0.7 : 1.0)
            )
            .scaleEffect(isAnimating ? 1.05 : 1.0)
            .opacity(isAnimating ? 0.8 : 1.0)
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(previewService.isPlaying)
        .accessibilityLabel("\(trainer.displayName)の声をプレビュー")
        .accessibilityHint("タップするとランダムな音声サンプルが再生されます")
        .onChange(of: previewService.isPlaying) { oldValue, newValue in
            if newValue {
                startPulseAnimation()
            } else {
                stopPulseAnimation()
            }
        }
    }

    /// パルスアニメーション開始
    private func startPulseAnimation() {
        withAnimation(
            Animation
                .easeInOut(duration: 0.6)
                .repeatForever(autoreverses: true)
        ) {
            isAnimating = true
        }
    }

    /// パルスアニメーション停止
    private func stopPulseAnimation() {
        withAnimation(.easeOut(duration: 0.2)) {
            isAnimating = false
        }
    }
}

#Preview {
    VoicePreviewButton(trainer: .oshinoAi)
        .padding()
}