import SwiftUI
import AVFoundation
#if canImport(UIKit)
import UIKit
#endif

/// スワイプ対応のキャラクター選択ビュー
struct SwipableCharacterSelectionView: View {
    @StateObject private var voiceSettings = VoiceSettings.shared
    @StateObject private var audioService = AudioFeedbackService()
    
    @State private var selectedIndex = 0
    @State private var dragOffset: CGFloat = 0
    @State private var isDragging = false
    @State private var audioPlayer: AVAudioPlayer?
    
    private let characters = VoiceCharacter.allCases
    
    var body: some View {
        VStack(spacing: 24) {
            // インジケーター
            pageIndicator
            
            // スワイプ可能なキャラクターカルーセル
            characterCarousel
            
            // 選択ボタン
            selectionButton
        }
        .onAppear {
            // 現在選択中のキャラクターのインデックスを設定
            if let currentIndex = characters.firstIndex(of: voiceSettings.selectedCharacter) {
                selectedIndex = currentIndex
            }
        }
    }
    
    // MARK: - UI Components
    
    private var pageIndicator: some View {
        HStack(spacing: 8) {
            ForEach(characters.indices, id: \.self) { index in
                Circle()
                    .fill(index == selectedIndex ? Color.accentColor : Color.gray.opacity(0.3))
                    .frame(width: 8, height: 8)
                    .scaleEffect(index == selectedIndex ? 1.2 : 1.0)
                    .animation(.easeInOut(duration: 0.2), value: selectedIndex)
            }
        }
    }
    
    private var characterCarousel: some View {
        GeometryReader { geometry in
            HStack(spacing: 0) {
                ForEach(characters.indices, id: \.self) { index in
                    characterCard(for: characters[index], at: index, geometry: geometry)
                }
            }
            .offset(x: -CGFloat(selectedIndex) * geometry.size.width + dragOffset)
            .animation(.easeOut(duration: 0.3), value: selectedIndex)
            .gesture(
                DragGesture()
                    .onChanged { value in
                        isDragging = true
                        dragOffset = value.translation.width
                    }
                    .onEnded { value in
                        isDragging = false
                        let threshold: CGFloat = 50
                        let velocity = value.predictedEndLocation.x - value.location.x
                        
                        withAnimation(.easeOut(duration: 0.3)) {
                            if value.translation.width > threshold || velocity > 100 {
                                // 右にスワイプ（前のキャラクター）
                                if selectedIndex > 0 {
                                    selectedIndex -= 1
                                    playPreviewAudio(for: characters[selectedIndex])
                                }
                            } else if value.translation.width < -threshold || velocity < -100 {
                                // 左にスワイプ（次のキャラクター）
                                if selectedIndex < characters.count - 1 {
                                    selectedIndex += 1
                                    playPreviewAudio(for: characters[selectedIndex])
                                }
                            }
                            dragOffset = 0
                        }
                    }
            )
        }
        .frame(height: 300)
        .clipped()
    }
    
    private func characterCard(for character: VoiceCharacter, at index: Int, geometry: GeometryProxy) -> some View {
        VStack(spacing: 16) {
            // 大きなキャラクター画像
            CharacterImageView(
                character: character,
                size: CGSize(width: 180, height: 180)
            )
            .scaleEffect(index == selectedIndex ? 1.0 : 0.8)
            .opacity(index == selectedIndex ? 1.0 : 0.6)
            .animation(.easeInOut(duration: 0.3), value: selectedIndex)
            
            // キャラクター情報
            VStack(spacing: 8) {
                Text(character.displayName)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                
                Text(character.description)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .opacity(index == selectedIndex ? 1.0 : 0.7)
            .animation(.easeInOut(duration: 0.3), value: selectedIndex)
        }
        .frame(width: geometry.size.width)
        .frame(maxWidth: .infinity)
        .multilineTextAlignment(.center)
    }
    
    private var selectionButton: some View {
        Button(action: {
            let selectedCharacter = characters[selectedIndex]
            withAnimation(.easeInOut(duration: 0.3)) {
                voiceSettings.updateCharacter(selectedCharacter)
            }
            playPreviewAudio(for: selectedCharacter)
        }) {
            HStack(spacing: 12) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 20))
                
                Text("\(characters[selectedIndex].displayName)を選択")
                    .font(.headline)
            }
            .foregroundColor(.white)
            .padding(.horizontal, 32)
            .padding(.vertical, 16)
            .background(
                RoundedRectangle(cornerRadius: 25)
                    .fill(Color.accentColor)
                    .shadow(color: Color.accentColor.opacity(0.3), radius: 8, x: 0, y: 4)
            )
        }
        .scaleEffect(isDragging ? 0.95 : 1.0)
        .animation(.easeInOut(duration: 0.2), value: isDragging)
    }
    
    // MARK: - Helper Methods
    
    private func playPreviewAudio(for character: VoiceCharacter) {
        // 現在再生中の音声を停止
        audioPlayer?.stop()
        
        // 音声プレビューを再生
        if let audioURL = character.audioFileURL(for: .formError) {
            do {
                audioPlayer = try AVAudioPlayer(contentsOf: audioURL)
                audioPlayer?.volume = 0.8
                audioPlayer?.play()
                
                // ハプティックフィードバック
                #if os(iOS)
                let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
                impactFeedback.impactOccurred()
                #endif
                
                print("[SwipableCharacterSelection] Playing preview audio for: \(character.displayName)")
            } catch {
                print("[SwipableCharacterSelection] Failed to play preview audio: \(error)")
            }
        }
    }
}

// MARK: - Preview
#Preview {
    SwipableCharacterSelectionView()
        .padding()
}