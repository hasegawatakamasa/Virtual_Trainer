import Foundation

/// ボイスキャラクターの種類
enum VoiceCharacter: String, CaseIterable, Identifiable, Codable {
    case zundamon = "ずんだもん"
    
    /// Identifiable準拠のためのID
    var id: String { rawValue }
    
    /// 表示用の名前
    var displayName: String {
        switch self {
        case .zundamon:
            return "ずんだもん"
        }
    }
    
    /// キャラクターの説明
    var description: String {
        switch self {
        case .zundamon:
            return "可愛い声で応援してくれる東北の妖精"
        }
    }
    
    /// アイコン（システムアイコン名）
    var iconName: String {
        switch self {
        case .zundamon:
            return "heart.fill"
        }
    }
    
    /// 音声ファイルのディレクトリ名
    var directoryName: String {
        return rawValue
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
                slowEncouragement: "slow_encouragement.wav",
                fastWarning: "fast_warning.wav",
                formError: "form_error.wav",
                repCountPrefix: "count_"
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
}

/// UserDefaultsのキー拡張
extension UserDefaultsKeys {
    static let selectedVoiceCharacter = "selectedVoiceCharacter"
}