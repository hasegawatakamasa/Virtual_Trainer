import SwiftUI
import AVFoundation

/// 音声キャラクター選択画面
struct VoiceCharacterSettingsView: View {
    @StateObject private var voiceSettings = VoiceSettings.shared
    @StateObject private var audioService = AudioFeedbackService()
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // ヘッダー
                headerSection
                
                // キャラクター選択セクション
                characterSelectionSection
                
                Spacer()
            }
            .navigationTitle("音声キャラクター")
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
    
    // MARK: - View Components
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform.and.mic")
                .font(.largeTitle)
                .foregroundColor(.accentColor)
            
            Text("トレーニング中の音声フィードバックを\n担当するキャラクターを選択してください")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 20)
    }
    
    private var characterSelectionSection: some View {
        VStack(spacing: 16) {
            Text("キャラクター選択")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
            
            LazyVGrid(columns: [
                GridItem(.flexible(), spacing: 12),
                GridItem(.flexible(), spacing: 12)
            ], spacing: 16) {
                ForEach(VoiceCharacter.allCases) { character in
                    CharacterPreviewCard(
                        character: character,
                        isSelected: voiceSettings.selectedCharacter == character,
                        onSelect: {
                            withAnimation(.easeInOut(duration: 0.3)) {
                                voiceSettings.updateCharacter(character)
                            }
                            playPreviewAudio(for: character)
                        }
                    )
                }
            }
            .padding(.horizontal)
        }
    }
    
    // MARK: - Helper Methods
    
    private func playPreviewAudio(for character: VoiceCharacter) {
        // サンプル音声を再生（フォームエラー音声を使用）
        if let audioURL = character.audioFileURL(for: .formError) {
            do {
                let audioPlayer = try AVAudioPlayer(contentsOf: audioURL)
                audioPlayer.volume = 0.8
                audioPlayer.play()
            } catch {
                print("[VoiceCharacterSettings] Failed to play preview audio: \(error)")
            }
        }
    }
}

/// キャラクター選択カード
struct CharacterPreviewCard: View {
    let character: VoiceCharacter
    let isSelected: Bool
    let onSelect: () -> Void
    
    @State private var isPressed = false
    
    var body: some View {
        Button(action: {
            onSelect()
        }) {
            VStack(spacing: 12) {
                // キャラクターアイコン
                Image(systemName: character.iconName)
                    .font(.system(size: 40, weight: .medium))
                    .foregroundColor(isSelected ? .white : .accentColor)
                    .scaleEffect(isPressed ? 0.95 : 1.0)
                
                // キャラクター情報
                VStack(spacing: 4) {
                    Text(character.displayName)
                        .font(.headline)
                        .foregroundColor(isSelected ? .white : .primary)
                    
                    Text(character.description)
                        .font(.caption)
                        .foregroundColor(isSelected ? .white.opacity(0.8) : .secondary)
                        .multilineTextAlignment(.center)
                        .lineLimit(2)
                }
                
                // 選択インジケーター
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 20))
                        .foregroundColor(.white)
                }
            }
            .frame(height: 160)
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(isSelected ? Color.accentColor : Color(.systemBackground))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(
                                isSelected ? Color.clear : Color(.systemGray4),
                                lineWidth: 1
                            )
                    )
                    .shadow(
                        color: isSelected ? Color.accentColor.opacity(0.3) : Color.black.opacity(0.1),
                        radius: isSelected ? 8 : 4,
                        x: 0,
                        y: isSelected ? 4 : 2
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
        .scaleEffect(isPressed ? 0.98 : 1.0)
        .onLongPressGesture(minimumDuration: 0, maximumDistance: .infinity, pressing: { pressing in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = pressing
            }
        }, perform: {})
        .accessibilityLabel("\(character.displayName): \(character.description)")
        .accessibilityHint(isSelected ? "現在選択中" : "タップして選択")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

#Preview {
    VoiceCharacterSettingsView()
}