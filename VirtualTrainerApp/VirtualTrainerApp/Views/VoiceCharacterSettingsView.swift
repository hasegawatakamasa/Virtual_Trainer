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
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: {
                    #if os(iOS)
                    return .navigationBarTrailing
                    #else
                    return .automatic
                    #endif
                }()) {
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
            
            Text("左右にスワイプして選択")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
            
            // 新しいスワイプ対応キャラクター選択UI
            SwipableCharacterSelectionView()
                .padding(.horizontal)
        }
    }
    
}

#Preview {
    VoiceCharacterSettingsView()
}