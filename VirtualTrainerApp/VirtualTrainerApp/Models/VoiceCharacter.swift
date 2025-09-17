import Foundation

/// ボイスキャラクターの種類
enum VoiceCharacter: String, CaseIterable, Identifiable, Codable {
    case zundamon = "ずんだもん"
    case shikokuMetan = "四国めたん"
    
    /// Identifiable準拠のためのID
    var id: String { rawValue }
    
    /// 表示用の名前
    var displayName: String {
        switch self {
        case .zundamon:
            return "ずんだもん"
        case .shikokuMetan:
            return "四国めたん"
        }
    }
    
    /// キャラクターの説明
    var description: String {
        switch self {
        case .zundamon:
            return "可愛い声で応援してくれる東北の妖精"
        case .shikokuMetan:
            return "優しい関西弁で励ましてくれる四国の案内人"
        }
    }
    
    /// アイコン（システムアイコン名）
    var iconName: String {
        switch self {
        case .zundamon:
            return "heart.fill"
        case .shikokuMetan:
            return "leaf.fill"
        }
    }
    
    /// 音声ファイルのディレクトリ名
    var directoryName: String {
        return rawValue
    }

    /// 音声ファイルのフォルダ名（AudioFeedbackServiceで使用）
    var audioFolderName: String {
        switch self {
        case .zundamon:
            return "ずんだもん"
        case .shikokuMetan:
            return "四国めたん"
        }
    }
    
    /// VOICEVOX クレジット表記
    var creditText: String {
        switch self {
        case .zundamon:
            return "VOICEVOX:ずんだもん"
        case .shikokuMetan:
            return "VOICEVOX:四国めたん"
        }
    }
    
    /// キャラクター画像のファイル名
    var imageName: String {
        switch self {
        case .zundamon:
            return "zundamon_1"
        case .shikokuMetan:
            return "shikoku_metan_1"
        }
    }
    
    /// 画像ファイルのパス
    var imageFilePath: String? {
        let subdirectory = "Resources/Image/\(directoryName)"
        return Bundle.main.path(forResource: imageName, 
                              ofType: "png", 
                              inDirectory: subdirectory)
    }
    
    /// 画像の存在確認
    var hasImage: Bool {
        return imageFilePath != nil
    }
    
    /// アクセシビリティ用画像説明
    var accessibilityImageDescription: String {
        switch self {
        case .zundamon:
            return "ずんだもんのキャラクター画像。緑色の髪に白い服を着た可愛らしい東北の妖精"
        case .shikokuMetan:
            return "四国めたんのキャラクター画像。オレンジ色の髪の優しい関西弁を話す四国の案内人"
        }
    }
    
    /// 音声ファイルのパス構成
    struct AudioFiles {
        let slowEncouragement: String  // 動作が遅い時の励まし
        let fastWarning: String        // 動作が速い時の警告
        let formError: String          // フォームエラー
        let repCountPrefix: String     // 回数カウントのプレフィックス
        
        /// 回数カウント音声ファイル名を生成
        func repCountFileName(for count: Int) -> String {
            return "\(repCountPrefix)\(String(format: "%02d", count)).wav"
        }
    }
    
    /// キャラクター別の音声ファイル名
    var audioFiles: AudioFiles {
        switch self {
        case .zundamon:
            return AudioFiles(
                slowEncouragement: "zundamon_slow_encouragement.wav",
                fastWarning: "zundamon_fast_warning.wav",
                formError: "zundamon_form_error.wav",
                repCountPrefix: "zundamon_count_"
            )
        case .shikokuMetan:
            return AudioFiles(
                slowEncouragement: "shikoku_slow_encouragement.wav",
                fastWarning: "shikoku_fast_warning.wav",
                formError: "shikoku_form_error.wav",
                repCountPrefix: "shikoku_count_"
            )
        }
    }
    
    /// 音声ファイルのフルパス（Bundle内）を生成
    func audioFilePath(for audioType: AudioType) -> String? {
        let subdirectory = "Resources/Audio/\(directoryName)"
        
        switch audioType {
        case .slowEncouragement:
            return Bundle.main.path(forResource: audioFiles.slowEncouragement.replacingOccurrences(of: ".wav", with: ""),
                                  ofType: "wav",
                                  inDirectory: subdirectory)
        case .fastWarning:
            return Bundle.main.path(forResource: audioFiles.fastWarning.replacingOccurrences(of: ".wav", with: ""),
                                  ofType: "wav", 
                                  inDirectory: subdirectory)
        case .formError:
            return Bundle.main.path(forResource: audioFiles.formError.replacingOccurrences(of: ".wav", with: ""),
                                  ofType: "wav",
                                  inDirectory: subdirectory)
        case .repCount(let count):
            let fileName = audioFiles.repCountFileName(for: count)
            return Bundle.main.path(forResource: fileName.replacingOccurrences(of: ".wav", with: ""),
                                  ofType: "wav",
                                  inDirectory: subdirectory)
        }
    }
    
    /// 音声ファイルのURLを取得
    func audioFileURL(for audioType: AudioType) -> URL? {
        // 複数のサブディレクトリパターンを試す
        let subdirectoryPatterns = [
            "Resources/Audio/\(directoryName)",
            "Audio/\(directoryName)",
            directoryName
        ]
        
        let (fileName, resource): (String, String)
        
        switch audioType {
        case .slowEncouragement:
            fileName = audioFiles.slowEncouragement
            resource = fileName.replacingOccurrences(of: ".wav", with: "")
        case .fastWarning:
            fileName = audioFiles.fastWarning
            resource = fileName.replacingOccurrences(of: ".wav", with: "")
        case .formError:
            fileName = audioFiles.formError
            resource = fileName.replacingOccurrences(of: ".wav", with: "")
        case .repCount(let count):
            fileName = audioFiles.repCountFileName(for: count)
            resource = fileName.replacingOccurrences(of: ".wav", with: "")
        }
        
        // 各サブディレクトリパターンを試す
        for subdirectory in subdirectoryPatterns {
            if let url = Bundle.main.url(forResource: resource, withExtension: "wav", subdirectory: subdirectory) {
                print("[VoiceCharacter] Audio file found: \(url.path)")
                return url
            }
        }
        
        // メインバンドルからも試す（サブディレクトリなし）
        if let url = Bundle.main.url(forResource: resource, withExtension: "wav") {
            print("[VoiceCharacter] Audio file found in main bundle: \(url.path)")
            return url
        }
        
        print("[VoiceCharacter] Audio file not found: \(fileName) in character: \(displayName)")
        print("[VoiceCharacter] Searched subdirectories: \(subdirectoryPatterns)")
        return nil
    }
    
    /// 画像ファイルのURLを取得
    func imageFileURL() -> URL? {
        // 複数のサブディレクトリパターンを試す
        let subdirectoryPatterns = [
            "Resources/Image/\(directoryName)",
            "Image/\(directoryName)",
            directoryName
        ]
        
        let resource = imageName
        
        // 各サブディレクトリパターンを試す
        for subdirectory in subdirectoryPatterns {
            if let url = Bundle.main.url(forResource: resource, withExtension: "png", subdirectory: subdirectory) {
                print("[VoiceCharacter] Image file found: \(url.path)")
                return url
            }
        }
        
        // メインバンドルからも試す（サブディレクトリなし）
        if let url = Bundle.main.url(forResource: resource, withExtension: "png") {
            print("[VoiceCharacter] Image file found in main bundle: \(url.path)")
            return url
        }
        
        print("[VoiceCharacter] Image file not found: \(imageName).png in character: \(displayName)")
        print("[VoiceCharacter] Searched subdirectories: \(subdirectoryPatterns)")
        return nil
    }
}

/// 音声の種類
enum AudioType {
    case slowEncouragement      // 動作が遅い時の励まし
    case fastWarning           // 動作が速い時の警告
    case formError            // フォームエラー
    case repCount(Int)        // 回数カウント（数値指定）
}

/// ボイス設定の管理
class VoiceSettings: ObservableObject {
    @Published var selectedCharacter: VoiceCharacter {
        didSet {
            UserDefaults.standard.set(selectedCharacter.rawValue, forKey: UserDefaultsKeys.selectedVoiceCharacter)
        }
    }
    
    /// シングルトンインスタンス
    static let shared = VoiceSettings()
    
    private init() {
        // UserDefaultsから設定を読み込み、デフォルトはずんだもん
        if let savedCharacter = UserDefaults.standard.string(forKey: UserDefaultsKeys.selectedVoiceCharacter),
           let character = VoiceCharacter(rawValue: savedCharacter) {
            self.selectedCharacter = character
        } else {
            self.selectedCharacter = .zundamon
        }
    }
    
    /// キャラクターを更新
    func updateCharacter(_ character: VoiceCharacter) {
        selectedCharacter = character
    }
    
    /// 現在のキャラクターのクレジット文字列を取得
    func getCurrentCreditText() -> String {
        return selectedCharacter.creditText
    }
}

/// UserDefaultsのキー拡張
extension UserDefaultsKeys {
    static let selectedVoiceCharacter = "selectedVoiceCharacter"
}