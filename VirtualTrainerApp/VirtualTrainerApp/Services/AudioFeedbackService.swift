import Foundation
import SwiftUI
import Combine
import AVFoundation

/// 音声タスクの種類と優先度
enum AudioTaskType: Int, CaseIterable {
    case repCount = 1        // 最高優先度：回数カウント
    case formError = 2       // 高優先度：フォームエラー
    case speedFeedback = 3   // 低優先度：速度フィードバック
    
    var priority: Int { rawValue }
    
    var displayName: String {
        switch self {
        case .repCount: return "Rep Count"
        case .formError: return "Form Error"
        case .speedFeedback: return "Speed Feedback"
        }
    }
}

/// 音声再生タスク
struct AudioTask {
    let type: AudioTaskType
    let audioURL: URL
    let timestamp: Date
    let metadata: [String: Any]
    
    init(type: AudioTaskType, audioURL: URL, metadata: [String: Any] = [:]) {
        self.type = type
        self.audioURL = audioURL
        self.timestamp = Date()
        self.metadata = metadata
    }
}

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
    @Published var lastSpeedFeedbackTime: Date?
    
    // MARK: - Private Properties
    private var audioPlayer: AVAudioPlayer?
    private let feedbackCooldownInterval: TimeInterval = 3.0
    private var cancellables = Set<AnyCancellable>()
    private var exerciseZoneEntryTime: Date?
    private let warmupPeriod: TimeInterval = 2.0  // エクササイズゾーン入場後2秒間は音声を再生しない
    private let repCountCooldownInterval: TimeInterval = 1.0  // 回数カウント音声のクールダウン
    private let voiceSettings = VoiceSettings.shared  // ボイス設定
    
    // 音声キューシステム
    private var audioQueue: [AudioTask] = []
    private var isProcessingQueue = false
    
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
        
        // キューをクリア
        audioQueue.removeAll()
        isProcessingQueue = false
    }
    
    /// 回数カウント音声の再生
    func playRepCountAudio(count: Int) {
        guard isAudioEnabled else { return }
        
        // 11以上の場合は10の音声を再生（音声ファイルは1-10のみ）
        let audioCount = min(count, 10)
        guard audioCount >= 1 else {
            print("[AudioFeedbackService] Rep count invalid: \(count)")
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
        guard let audioURL = voiceSettings.selectedCharacter.audioFileURL(for: .repCount(audioCount)) else {
            print("[AudioFeedbackService] Rep count audio file not found for character: \(voiceSettings.selectedCharacter.displayName), count: \(audioCount)")
            return
        }
        
        // 音声タスクをキューに追加
        let task = AudioTask(type: .repCount, audioURL: audioURL, metadata: ["count": count])
        enqueueAudioTask(task)
    }
    
    /// 速度フィードバック音声の再生
    func playSpeedFeedback(_ speed: ExerciseSpeed) {
        guard isAudioEnabled else { return }
        
        // 速度フィードバック専用のクールダウンチェック
        if let lastTime = lastSpeedFeedbackTime {
            let timeSinceLastFeedback = Date().timeIntervalSince(lastTime)
            if timeSinceLastFeedback < feedbackCooldownInterval {
                print("[AudioFeedbackService] Skipping speed feedback - cooldown active")
                return
            }
        }
        
        // 速度に応じた音声ファイルを取得
        let audioType: AudioType
        switch speed {
        case .fast:
            audioType = .fastWarning
        case .slow:
            audioType = .slowEncouragement
        case .normal:
            return // Normal speed doesn't need feedback
        }
        
        // 現在選択されているキャラクターから音声ファイルURLを取得
        guard let audioURL = voiceSettings.selectedCharacter.audioFileURL(for: audioType) else {
            print("[AudioFeedbackService] Speed feedback audio file not found for character: \(voiceSettings.selectedCharacter.displayName), type: \(audioType)")
            return
        }
        
        // 音声タスクをキューに追加
        let task = AudioTask(type: .speedFeedback, audioURL: audioURL, metadata: ["speed": speed.rawValue])
        enqueueAudioTask(task)
    }
    
    /// 速度フィードバック音声ファイルの事前読み込み確認
    func validateSpeedFeedbackAudio() -> Bool {
        let character = voiceSettings.selectedCharacter
        let audioTypes: [AudioType] = [.slowEncouragement, .fastWarning]
        
        var allFilesExist = true
        
        for audioType in audioTypes {
            if character.audioFileURL(for: audioType) == nil {
                print("[AudioFeedbackService] Missing speed feedback audio file for character: \(character.displayName), type: \(audioType)")
                allFilesExist = false
            }
        }
        
        if allFilesExist {
            print("[AudioFeedbackService] All speed feedback audio files validated successfully for character: \(character.displayName)")
        }
        
        return allFilesExist
    }
    
    /// 現在選択されているキャラクターの音声を再生
    func playAudioForCurrentCharacter(audioType: AudioType) {
        guard isAudioEnabled else { return }
        
        let character = voiceSettings.selectedCharacter
        print("[AudioFeedbackService] Attempting to play \(audioType) for character: \(character.displayName)")
        
        // 音声ファイルURL取得（フォールバック機能付き）
        guard let audioURL = getAudioURLWithFallback(for: audioType, character: character) else {
            print("[AudioFeedbackService] No audio file found for \(audioType) with any character")
            return
        }
        
        let task = AudioTask(type: audioTypeToTaskType(audioType), audioURL: audioURL, metadata: ["character": character.rawValue])
        enqueueAudioTask(task)
    }
    
    /// フォールバック機能付きの音声URL取得
    private func getAudioURLWithFallback(for audioType: AudioType, character: VoiceCharacter) -> URL? {
        // 1. 指定されたキャラクターの音声ファイルを試行
        if let url = character.audioFileURL(for: audioType) {
            print("[AudioFeedbackService] Found audio file for \(character.displayName): \(url.lastPathComponent)")
            return url
        }
        
        // 2. デフォルトキャラクター（ずんだもん）にフォールバック
        if character != .zundamon {
            print("[AudioFeedbackService] Falling back to ずんだもん for \(audioType)")
            if let fallbackURL = VoiceCharacter.zundamon.audioFileURL(for: audioType) {
                return fallbackURL
            }
        }
        
        // 3. 音声ファイルが見つからない
        print("[AudioFeedbackService] Audio file not found for \(audioType) with character: \(character.displayName)")
        return nil
    }
    
    /// AudioTypeをAudioTaskTypeに変換
    private func audioTypeToTaskType(_ audioType: AudioType) -> AudioTaskType {
        switch audioType {
        case .repCount:
            return .repCount
        case .formError:
            return .formError
        case .slowEncouragement, .fastWarning:
            return .speedFeedback
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
    
    // MARK: - Audio Queue System
    
    /// 音声タスクをキューに追加
    private func enqueueAudioTask(_ task: AudioTask) {
        print("[AudioFeedbackService] Enqueuing \(task.type.displayName) audio task")
        
        // 同じ種類のタスクが既にキューにある場合は置き換え（最新を優先）
        audioQueue.removeAll { $0.type == task.type }
        
        // 優先度に基づいてキューに挿入
        audioQueue.append(task)
        audioQueue.sort { $0.type.priority < $1.type.priority }
        
        // キューの処理を開始
        processAudioQueue()
    }
    
    /// 音声キューの処理
    private func processAudioQueue() {
        guard !isProcessingQueue else { return }
        guard !audioQueue.isEmpty else { return }
        guard !currentlyPlaying else {
            print("[AudioFeedbackService] Audio currently playing, will process queue after completion")
            return
        }
        
        isProcessingQueue = true
        
        // 最高優先度のタスクを取得
        guard let task = audioQueue.first else {
            isProcessingQueue = false
            return
        }
        
        // キューから削除
        audioQueue.removeFirst()
        
        // 実際に音声を再生
        playAudioTask(task)
    }
    
    /// 音声タスクの実際の再生
    private func playAudioTask(_ task: AudioTask) {
        print("[AudioFeedbackService] Playing \(task.type.displayName) audio")
        
        // クールダウンチェック
        if shouldSkipDueToCooldown(for: task.type) {
            print("[AudioFeedbackService] Skipping \(task.type.displayName) due to cooldown")
            isProcessingQueue = false
            // キューの次のアイテムを処理
            DispatchQueue.main.async {
                self.processAudioQueue()
            }
            return
        }
        
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: task.audioURL)
            audioPlayer?.delegate = self
            audioPlayer?.prepareToPlay()
            
            if audioPlayer?.play() == true {
                currentlyPlaying = true
                updateLastFeedbackTime(for: task.type)
                print("[AudioFeedbackService] Successfully started playing \(task.type.displayName)")
            } else {
                print("[AudioFeedbackService] Failed to start \(task.type.displayName) playback")
                isProcessingQueue = false
                // 次のタスクを処理
                DispatchQueue.main.async {
                    self.processAudioQueue()
                }
            }
        } catch {
            print("[AudioFeedbackService] Error creating audio player for \(task.type.displayName): \(error)")
            isProcessingQueue = false
            // 次のタスクを処理
            DispatchQueue.main.async {
                self.processAudioQueue()
            }
        }
    }
    
    /// クールダウンチェック
    private func shouldSkipDueToCooldown(for taskType: AudioTaskType) -> Bool {
        let now = Date()
        
        switch taskType {
        case .repCount:
            if let lastTime = lastFeedbackTime {
                return now.timeIntervalSince(lastTime) < repCountCooldownInterval
            }
        case .formError:
            if let lastTime = lastFeedbackTime {
                return now.timeIntervalSince(lastTime) < feedbackCooldownInterval
            }
        case .speedFeedback:
            if let lastTime = lastSpeedFeedbackTime {
                return now.timeIntervalSince(lastTime) < feedbackCooldownInterval
            }
        }
        
        return false
    }
    
    /// 最後のフィードバック時間を更新
    private func updateLastFeedbackTime(for taskType: AudioTaskType) {
        let now = Date()
        
        switch taskType {
        case .repCount, .formError:
            lastFeedbackTime = now
        case .speedFeedback:
            lastSpeedFeedbackTime = now
        }
    }
    
    /// 肘エラー音声フィードバックの再生
    private func playElbowErrorFeedback() {
        // フォームエラー音声ファイルの読み込み
        guard let audioURL = voiceSettings.selectedCharacter.audioFileURL(for: .formError) else {
            print("[AudioFeedbackService] Form error audio file not found for character: \(voiceSettings.selectedCharacter.displayName)")
            return
        }
        
        // 音声タスクをキューに追加
        let task = AudioTask(type: .formError, audioURL: audioURL, metadata: [:])
        enqueueAudioTask(task)
    }
    
    // MARK: - AVAudioPlayerDelegate
    
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            if player === self.audioPlayer {
                self.currentlyPlaying = false
                self.isProcessingQueue = false
                print("[AudioFeedbackService] Audio playback finished successfully: \(flag)")
                
                // キューの次のアイテムを処理
                self.processAudioQueue()
            }
        }
    }
    
    nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        Task { @MainActor in
            if player === self.audioPlayer {
                self.currentlyPlaying = false
                self.isProcessingQueue = false
                if let error = error {
                    print("[AudioFeedbackService] Audio decode error: \(error)")
                }
                
                // キューの次のアイテムを処理
                self.processAudioQueue()
            }
        }
    }
}