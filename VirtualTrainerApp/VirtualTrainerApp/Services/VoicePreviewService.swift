import Foundation
import AVFoundation
import Combine

#if os(iOS)
import UIKit
#endif

/// ランダム音声プレビュー再生サービス
@MainActor
class VoicePreviewService: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var isPlaying: Bool = false

    private var audioPlayer: AVAudioPlayer?

    /// シングルトンインスタンス
    static let shared = VoicePreviewService()

    override private init() {
        super.init()
    }

    /// ランダム音声プレビューを再生
    /// - Parameter trainer: 対象の推しトレーナー
    func playRandomPreview(for trainer: OshiTrainer) {
        // 既に再生中の場合は停止
        if isPlaying {
            audioPlayer?.stop()
            isPlaying = false
        }

        // ランダム音声タイプ配列定義
        let audioTypes: [AudioType] = [
            .repCount(1),
            .repCount(2),
            .repCount(3),
            .slowEncouragement,
            .fastWarning,
            .formError
        ]

        // ランダムに音声タイプを選択
        guard let randomAudioType = audioTypes.randomElement() else {
            print("[VoicePreviewService] ランダム音声タイプの選択に失敗")
            return
        }

        // 音声URLを取得
        guard let audioURL = trainer.voiceCharacter.audioFileURL(for: randomAudioType) else {
            print("[VoicePreviewService] 音声ファイルが見つかりません: トレーナー=\(trainer.displayName), 音声タイプ=\(randomAudioType)")
            return
        }

        do {
            // AVAudioPlayerを初期化
            audioPlayer = try AVAudioPlayer(contentsOf: audioURL)
            audioPlayer?.delegate = self
            audioPlayer?.prepareToPlay()

            // 再生開始
            if audioPlayer?.play() == true {
                isPlaying = true

                // ハプティックフィードバック
                provideHapticFeedback()

                print("[VoicePreviewService] プレビュー音声を再生: トレーナー=\(trainer.displayName), 音声タイプ=\(randomAudioType)")
            } else {
                print("[VoicePreviewService] 音声再生の開始に失敗")
            }
        } catch {
            print("[VoicePreviewService] AVAudioPlayerの初期化に失敗: \(error.localizedDescription)")
        }
    }

    /// ハプティックフィードバックを提供
    private func provideHapticFeedback() {
        #if os(iOS)
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.prepare()
        generator.impactOccurred()
        print("[VoicePreviewService] ハプティックフィードバックを提供")
        #endif
    }

    /// 再生を停止
    func stopPreview() {
        audioPlayer?.stop()
        isPlaying = false
        print("[VoicePreviewService] プレビュー音声を停止")
    }

    // MARK: - AVAudioPlayerDelegate

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.isPlaying = false
            print("[VoicePreviewService] プレビュー音声の再生が完了: 成功=\(flag)")
        }
    }
}