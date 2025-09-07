import Foundation
import SwiftUI
import Combine
import AVFoundation

/// フォームエラー音声フィードバック機能のエラー定義
enum AudioFeedbackError: LocalizedError {
    case audioFileNotFound
    case audioSessionSetupFailed(Error)
    case playbackFailed(Error)
    case unsupportedAudioFormat
    
    var errorDescription: String? {
        switch self {
        case .audioFileNotFound:
            return "音声ファイルが見つかりません"
        case .audioSessionSetupFailed(let error):
            return "音声システムの初期化に失敗しました: \(error.localizedDescription)"
        case .playbackFailed(let error):
            return "音声再生エラー: \(error.localizedDescription)"
        case .unsupportedAudioFormat:
            return "サポートされていない音声形式です"
        }
    }
}

/// フォームエラー音声フィードバックサービス
@MainActor
class AudioFeedbackService: NSObject, ObservableObject, AVAudioPlayerDelegate {
    
    // MARK: - Published Properties
    @Published var isAudioEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isAudioEnabled, forKey: UserDefaultsKeys.audioFeedbackEnabled)
            if isAudioEnabled {
                setupAudioSession()
            } else {
                stopCurrentFeedback()
            }
        }
    }
    
    @Published var currentlyPlaying: Bool = false
    @Published var lastFeedbackTime: Date?
    
    // MARK: - Private Properties
    private var audioPlayer: AVAudioPlayer?
    private let feedbackCooldownInterval: TimeInterval = 3.0
    private let audioFileName = "zundamon_elbow_error.wav"
    private var cancellables = Set<AnyCancellable>()
    private var exerciseZoneEntryTime: Date?
    private let warmupPeriod: TimeInterval = 2.0  // エクササイズゾーン入場後2秒間は音声を再生しない
    private let repCountCooldownInterval: TimeInterval = 1.0  // 回数カウント音声のクールダウン
    
    // MARK: - Initialization
    override init() {
        // UserDefaultsから設定を読み込み、デフォルトはtrue
        self.isAudioEnabled = UserDefaults.standard.object(forKey: UserDefaultsKeys.audioFeedbackEnabled) as? Bool ?? true
        
        super.init()
        
        // 初期化時にAVAudioSessionをセットアップ
        if isAudioEnabled {
            setupAudioSession()
        }
    }
    
    // MARK: - Public Methods
    
    /// FormClassification結果を処理して適切な音声フィードバックを提供
    func processFormResult(_ result: FormClassification.Result, isInExerciseZone: Bool = false) {
        guard isAudioEnabled else { return }
        guard result.isReliable else { return }
        
        // エクササイズゾーンの状態を追跡
        if isInExerciseZone && exerciseZoneEntryTime == nil {
            // エクササイズゾーンに初めて入った
            exerciseZoneEntryTime = Date()
            print("[AudioFeedbackService] Entered exercise zone, starting warmup period")
        } else if !isInExerciseZone {
            // エクササイズゾーンから出た
            exerciseZoneEntryTime = nil
        }
        
        // エクササイズゾーン内でElbow Errorの場合のみ音声フィードバックを実行
        // 準備中（エクササイズゾーン外）では音声を再生しない
        if result.classification == .elbowError && isInExerciseZone {
            // ウォームアップ期間中は音声を再生しない
            if let entryTime = exerciseZoneEntryTime {
                let timeSinceEntry = Date().timeIntervalSince(entryTime)
                if timeSinceEntry < warmupPeriod {
                    print("[AudioFeedbackService] Skipping feedback - in warmup period (\(String(format: "%.1f", timeSinceEntry))s)")
                    return
                }
            }
            playElbowErrorFeedback()
        }
    }
    
    /// 現在の音声再生を停止
    func stopCurrentFeedback() {
        if currentlyPlaying {
            audioPlayer?.stop()
            audioPlayer = nil
            currentlyPlaying = false
            print("[AudioFeedbackService] Audio feedback stopped")
        }
    }
    
    /// 回数カウント音声の再生
    func playRepCountAudio(count: Int) {
        guard isAudioEnabled else { return }
        guard count >= 1 && count <= 10 else {
            print("[AudioFeedbackService] Rep count out of range: \(count)")
            return
        }
        
        // クールダウンチェック（短めの間隔）
        if let lastTime = lastFeedbackTime {
            let timeSinceLastFeedback = Date().timeIntervalSince(lastTime)
            if timeSinceLastFeedback < repCountCooldownInterval {
                print("[AudioFeedbackService] Skipping rep count - cooldown active")
                return
            }
        }
        
        // 既に再生中の場合はスキップ
        guard !currentlyPlaying else { 
            print("[AudioFeedbackService] Skipping rep count - already playing")
            return
        }
        
        // 回数音声ファイルの読み込み
        guard let audioURL = loadRepCountAudioFile(count: count) else {
            print("[AudioFeedbackService] Failed to load rep count audio file: \(count)")
            return
        }
        
        do {
            // AVAudioPlayerを作成
            audioPlayer = try AVAudioPlayer(contentsOf: audioURL)
            audioPlayer?.delegate = self
            audioPlayer?.prepareToPlay()
            
            // 音声再生開始
            if audioPlayer?.play() == true {
                currentlyPlaying = true
                lastFeedbackTime = Date()
                print("[AudioFeedbackService] Playing rep count audio: \(count)")
            } else {
                print("[AudioFeedbackService] Failed to start rep count playback")
            }
            
        } catch {
            print("[AudioFeedbackService] Failed to create rep count audio player: \(error)")
        }
    }
    
    // MARK: - Private Methods
    
    /// AVAudioSessionのセットアップ
    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default, options: [.mixWithOthers])
            try audioSession.setActive(true)
            
            print("[AudioFeedbackService] Audio session setup completed successfully")
        } catch {
            print("[AudioFeedbackService] Failed to setup audio session: \(error)")
            // エラーハンドリング - 音声機能は無効化するが、アプリ全体は継続
        }
    }
    
    /// 音声ファイルの読み込み
    private func loadAudioFile() -> URL? {
        // まず、subdirectoryなしで試す（最も一般的）
        if let audioURL = Bundle.main.url(forResource: "zundamon_elbow_error", withExtension: "wav") {
            print("[AudioFeedbackService] Audio file found at: \(audioURL.path)")
            return audioURL
        }
        
        // 次に、Audioディレクトリ内を試す
        if let audioURL = Bundle.main.url(forResource: "zundamon_elbow_error", withExtension: "wav", subdirectory: "Audio") {
            print("[AudioFeedbackService] Audio file found in Audio directory: \(audioURL.path)")
            return audioURL
        }
        
        // 最後に、Resources/Audioを試す
        if let audioURL = Bundle.main.url(forResource: "zundamon_elbow_error", withExtension: "wav", subdirectory: "Resources/Audio") {
            print("[AudioFeedbackService] Audio file found in Resources/Audio: \(audioURL.path)")
            return audioURL
        }
        
        print("[AudioFeedbackService] Audio file not found: \(audioFileName)")
        print("[AudioFeedbackService] Bundle resource path: \(Bundle.main.resourcePath ?? "nil")")
        return nil
    }
    
    /// 回数カウント音声ファイルの読み込み
    private func loadRepCountAudioFile(count: Int) -> URL? {
        let fileName = "\(count)"
        
        // まず、subdirectoryなしで試す
        if let audioURL = Bundle.main.url(forResource: fileName, withExtension: "wav") {
            print("[AudioFeedbackService] Rep count audio file found at: \(audioURL.path)")
            return audioURL
        }
        
        // 次に、Audioディレクトリ内を試す
        if let audioURL = Bundle.main.url(forResource: fileName, withExtension: "wav", subdirectory: "Audio") {
            print("[AudioFeedbackService] Rep count audio file found in Audio directory: \(audioURL.path)")
            return audioURL
        }
        
        // 最後に、Resources/Audioを試す
        if let audioURL = Bundle.main.url(forResource: fileName, withExtension: "wav", subdirectory: "Resources/Audio") {
            print("[AudioFeedbackService] Rep count audio file found in Resources/Audio: \(audioURL.path)")
            return audioURL
        }
        
        print("[AudioFeedbackService] Rep count audio file not found: \(fileName).wav")
        return nil
    }
    
    /// 音声ファイルの検証
    private func validateAudioFile() -> Bool {
        guard let audioURL = loadAudioFile() else {
            return false
        }
        
        // ファイルの存在確認
        let fileExists = FileManager.default.fileExists(atPath: audioURL.path)
        if !fileExists {
            print("[AudioFeedbackService] Audio file does not exist at path: \(audioURL.path)")
            return false
        }
        
        // ファイルサイズチェック（1MB以上は警告）
        do {
            let fileAttributes = try FileManager.default.attributesOfItem(atPath: audioURL.path)
            if let fileSize = fileAttributes[FileAttributeKey.size] as? Int64 {
                if fileSize > 1_024_000 { // 1MB
                    print("[AudioFeedbackService] Warning: Audio file size is large (\(fileSize) bytes)")
                }
                print("[AudioFeedbackService] Audio file validated - size: \(fileSize) bytes")
            }
        } catch {
            print("[AudioFeedbackService] Failed to get file attributes: \(error)")
            return false
        }
        
        return true
    }
    
    /// 音声ファイルサイズの取得
    private func getAudioFileSize() -> Int64 {
        guard let audioURL = loadAudioFile() else { return 0 }
        
        do {
            let fileAttributes = try FileManager.default.attributesOfItem(atPath: audioURL.path)
            return fileAttributes[FileAttributeKey.size] as? Int64 ?? 0
        } catch {
            return 0
        }
    }
    
    /// 肘エラー音声フィードバックの再生
    private func playElbowErrorFeedback() {
        // クールダウンチェック
        if let lastTime = lastFeedbackTime {
            let timeSinceLastFeedback = Date().timeIntervalSince(lastTime)
            if timeSinceLastFeedback < feedbackCooldownInterval {
                print("[AudioFeedbackService] Skipping feedback - cooldown active")
                return
            }
        }
        
        // 既に再生中の場合はスキップ
        guard !currentlyPlaying else { 
            print("[AudioFeedbackService] Skipping feedback - already playing")
            return
        }
        
        // 音声ファイルの読み込みと検証
        guard validateAudioFile(), let audioURL = loadAudioFile() else {
            print("[AudioFeedbackService] Failed to load audio file")
            return
        }
        
        do {
            // AVAudioPlayerを作成
            audioPlayer = try AVAudioPlayer(contentsOf: audioURL)
            audioPlayer?.delegate = self
            audioPlayer?.prepareToPlay()
            
            // 音声再生開始
            if audioPlayer?.play() == true {
                currentlyPlaying = true
                lastFeedbackTime = Date()
                print("[AudioFeedbackService] Playing elbow error feedback")
            } else {
                print("[AudioFeedbackService] Failed to start playback")
            }
            
        } catch {
            print("[AudioFeedbackService] Failed to create audio player: \(error)")
        }
    }
    
    // MARK: - AVAudioPlayerDelegate
    
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            currentlyPlaying = false
            print("[AudioFeedbackService] Audio playback finished successfully: \(flag)")
        }
    }
    
    nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        Task { @MainActor in
            currentlyPlaying = false
            if let error = error {
                print("[AudioFeedbackService] Audio decode error: \(error)")
            }
        }
    }
}