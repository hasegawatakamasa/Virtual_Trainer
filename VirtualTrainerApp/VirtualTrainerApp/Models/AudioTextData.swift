import Foundation

// MARK: - Audio Text Data

/// 音声フィードバックとテキスト表示の連携情報を管理する構造体
struct AudioTextData: Identifiable {
    let id = UUID()
    let text: String
    let character: VoiceCharacter
    let startTime: Date
    let estimatedDuration: TimeInterval
    let isActive: Bool
    let audioType: AudioType
    
    /// 音声の終了予定時刻
    var estimatedEndTime: Date {
        return startTime.addingTimeInterval(estimatedDuration)
    }
    
    /// 現在時刻での音声の残り時間
    var remainingDuration: TimeInterval {
        let now = Date()
        let endTime = estimatedEndTime
        return max(0, endTime.timeIntervalSince(now))
    }
    
    /// 音声が現在アクティブかどうか
    var isCurrentlyActive: Bool {
        return isActive && remainingDuration > 0
    }
    
    /// 音声の進行率（0.0 - 1.0）
    var progress: Double {
        guard estimatedDuration > 0 else { return 1.0 }
        
        let elapsed = Date().timeIntervalSince(startTime)
        let progress = min(max(elapsed / estimatedDuration, 0.0), 1.0)
        return progress
    }
    
    /// 表示用テキストの作成
    var displayText: String {
        // キャラクター固有のテキスト装飾
        switch character {
        case .zundamon:
            return "🌟 \(text)"
        case .shikokuMetan:
            return "🍃 \(text)"
        }
    }
    
    /// 初期化
    init(
        text: String,
        character: VoiceCharacter,
        audioType: AudioType,
        estimatedDuration: TimeInterval = 3.0,
        isActive: Bool = true
    ) {
        self.text = text
        self.character = character
        self.audioType = audioType
        self.startTime = Date()
        self.estimatedDuration = estimatedDuration
        self.isActive = isActive
    }
    
    /// 非アクティブ状態のデータを作成
    static func inactive(text: String, character: VoiceCharacter, audioType: AudioType) -> AudioTextData {
        return AudioTextData(
            text: text,
            character: character,
            audioType: audioType,
            estimatedDuration: 0,
            isActive: false
        )
    }
}

// MARK: - Audio Text Data Extensions

extension AudioTextData {
    /// 音声種別に応じた表示色の取得
    var displayColor: String {
        switch audioType {
        case .formError:
            return "#FF6B6B"        // 赤系：エラー
        case .repCount:
            return "#4ECDC4"        // 青緑系：回数カウント
        case .slowEncouragement:
            return "#45B7D1"        // 青系：励まし
        case .fastWarning:
            return "#FFA726"        // オレンジ系：警告
        }
    }
    
    /// 音声種別に応じたアニメーション設定
    var animationConfig: AudioTextAnimationConfig {
        switch audioType {
        case .formError:
            return AudioTextAnimationConfig(
                entryAnimation: .shake,
                exitAnimation: .fadeOut,
                duration: estimatedDuration,
                emphasis: true
            )
        case .repCount:
            return AudioTextAnimationConfig(
                entryAnimation: .bounceIn,
                exitAnimation: .slideUp,
                duration: estimatedDuration,
                emphasis: false
            )
        case .slowEncouragement, .fastWarning:
            return AudioTextAnimationConfig(
                entryAnimation: .slideIn,
                exitAnimation: .fadeOut,
                duration: estimatedDuration,
                emphasis: false
            )
        }
    }
}

// MARK: - Audio Text Animation Config

/// 音声テキストアニメーションの設定
struct AudioTextAnimationConfig {
    let entryAnimation: AudioTextAnimation
    let exitAnimation: AudioTextAnimation
    let duration: TimeInterval
    let emphasis: Bool
    
    static let `default` = AudioTextAnimationConfig(
        entryAnimation: .fadeIn,
        exitAnimation: .fadeOut,
        duration: 2.0,
        emphasis: false
    )
}

/// アニメーションタイプ
enum AudioTextAnimation {
    case fadeIn
    case fadeOut
    case slideIn
    case slideUp
    case bounceIn
    case shake
    
    /// アニメーション時間
    var duration: Double {
        switch self {
        case .fadeIn, .fadeOut:
            return 0.3
        case .slideIn, .slideUp:
            return 0.4
        case .bounceIn:
            return 0.5
        case .shake:
            return 0.6
        }
    }
}

// MARK: - Audio Text Queue

/// 音声テキストデータのキューを管理するクラス
class AudioTextQueue: ObservableObject {
    @Published private(set) var currentText: AudioTextData?
    @Published private(set) var queuedTexts: [AudioTextData] = []
    
    private var updateTimer: Timer?
    
    /// 音声テキストをキューに追加
    func enqueue(_ audioText: AudioTextData) {
        if currentText == nil {
            setCurrentText(audioText)
        } else {
            queuedTexts.append(audioText)
        }
    }
    
    /// 現在の音声テキストを設定
    private func setCurrentText(_ audioText: AudioTextData) {
        currentText = audioText
        startUpdateTimer()
    }
    
    /// 現在の音声テキストをクリア
    func clearCurrentText() {
        currentText = nil
        stopUpdateTimer()
        processQueue()
    }
    
    /// キューをクリア
    func clearAll() {
        currentText = nil
        queuedTexts.removeAll()
        stopUpdateTimer()
    }
    
    /// キューの処理
    private func processQueue() {
        guard !queuedTexts.isEmpty else { return }
        
        let nextText = queuedTexts.removeFirst()
        setCurrentText(nextText)
    }
    
    /// 更新タイマーの開始
    private func startUpdateTimer() {
        stopUpdateTimer()
        
        updateTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            if let current = self.currentText, !current.isCurrentlyActive {
                self.clearCurrentText()
            }
        }
    }
    
    /// 更新タイマーの停止
    private func stopUpdateTimer() {
        updateTimer?.invalidate()
        updateTimer = nil
    }
    
    deinit {
        stopUpdateTimer()
    }
}