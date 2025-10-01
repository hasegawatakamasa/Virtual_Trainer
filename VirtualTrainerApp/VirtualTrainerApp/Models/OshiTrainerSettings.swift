import Foundation
import Combine

/// 推しトレーナー設定管理（ObservableObject）
class OshiTrainerSettings: ObservableObject {
    @Published var selectedTrainer: OshiTrainer {
        didSet {
            // 共有UserDefaultsへの永続化
            sharedDefaults.set(selectedTrainer.id, forKey: UserDefaultsKeys.selectedOshiTrainerId)

            // VoiceSettings連携
            VoiceSettings.shared.updateCharacter(selectedTrainer.voiceCharacter)

            print("[OshiTrainerSettings] トレーナーが変更されました: \(selectedTrainer.displayName)")
        }
    }

    /// シングルトンインスタンス
    static let shared = OshiTrainerSettings()

    /// App Group共有UserDefaults
    private let sharedDefaults: UserDefaults

    private init() {
        // App Group共有UserDefaultsの初期化
        self.sharedDefaults = UserDefaults(suiteName: "group.com.yourcompany.VirtualTrainer") ?? .standard

        // UserDefaultsから設定を読み込み、デフォルトは推乃 藍
        if let savedId = sharedDefaults.string(forKey: UserDefaultsKeys.selectedOshiTrainerId),
           let trainer = OshiTrainer.allTrainers.first(where: { $0.id == savedId }) {
            self.selectedTrainer = trainer
            print("[OshiTrainerSettings] 保存されたトレーナーを復元: \(trainer.displayName)")
        } else {
            self.selectedTrainer = .oshinoAi
            print("[OshiTrainerSettings] デフォルトトレーナーを使用: 推乃 藍")

            // 初回起動時にマイグレーション実行
            migrateVoiceCharacterToOshiTrainer()
        }
    }

    /// トレーナー更新
    func updateTrainer(_ trainer: OshiTrainer) {
        selectedTrainer = trainer
    }

    /// 現在の音声キャラクター取得（AudioFeedbackService連携用）
    var currentVoiceCharacter: VoiceCharacter {
        return selectedTrainer.voiceCharacter
    }

    /// データマイグレーション: VoiceCharacterからOshiTrainerへ
    private func migrateVoiceCharacterToOshiTrainer() {
        // 既存のselectedVoiceCharacterを読み込み（標準UserDefaultsから）
        if let savedVoiceCharacter = UserDefaults.standard.string(forKey: UserDefaultsKeys.selectedVoiceCharacter),
           let voiceChar = VoiceCharacter(rawValue: savedVoiceCharacter) {

            // 既存のボイスキャラクターに対応するデフォルトトレーナーにマッピング
            let matchingTrainer: OshiTrainer
            switch voiceChar {
            case .zundamon:
                matchingTrainer = .oshinoAi  // ずんだもんボイスを使用している推乃 藍にマッピング
            case .shikokuMetan:
                // 将来: 四国めたんボイスを使用するデフォルトトレーナーを追加予定
                matchingTrainer = .oshinoAi  // 現在は推乃 藍のみ
            }

            // 共有UserDefaultsに保存
            sharedDefaults.set(matchingTrainer.id, forKey: UserDefaultsKeys.selectedOshiTrainerId)
            self.selectedTrainer = matchingTrainer

            print("[OshiTrainerSettings] マイグレーション完了: \(voiceChar.displayName) → \(matchingTrainer.displayName)")
        }
    }
}

/// UserDefaultsのキー拡張
extension UserDefaultsKeys {
    static let selectedOshiTrainerId = "selectedOshiTrainerId"
}