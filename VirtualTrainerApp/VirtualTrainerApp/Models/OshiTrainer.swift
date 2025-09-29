import Foundation

/// 推しトレーナーデータモデル
struct OshiTrainer: Identifiable, Codable, Equatable {
    let id: String
    let displayName: String        // トレーナー名
    let firstPerson: String         // 一人称
    let secondPerson: String        // 二人称（呼び方）
    let personality: String         // 性格設定
    let voiceCharacter: VoiceCharacter  // 紐付けられた音声キャラクター
    let imageName: String          // 画像リソース識別子
    let imageDirectory: String     // 画像ディレクトリ

    /// 画像ファイルのURL取得
    func imageFileURL() -> URL? {
        let subdirectoryPatterns = [
            "Resources/Image/\(imageDirectory)",
            "Image/\(imageDirectory)",
            imageDirectory
        ]

        for subdirectory in subdirectoryPatterns {
            if let url = Bundle.main.url(
                forResource: imageName,
                withExtension: "png",
                subdirectory: subdirectory
            ) {
                print("[OshiTrainer] Image file found: \(url.path)")
                return url
            }
        }

        // メインバンドルからも試す（サブディレクトリなし）
        if let url = Bundle.main.url(forResource: imageName, withExtension: "png") {
            print("[OshiTrainer] Image file found in main bundle: \(url.path)")
            return url
        }

        print("[OshiTrainer] Image file not found: \(imageName).png for trainer: \(displayName)")
        print("[OshiTrainer] Searched subdirectories: \(subdirectoryPatterns)")
        return nil
    }

    /// アクセシビリティ説明
    var accessibilityDescription: String {
        return "\(displayName)のキャラクター画像。性格：\(personality)、一人称：\(firstPerson)、呼び方：\(secondPerson)"
    }
}

extension OshiTrainer {
    /// デフォルト推しトレーナー「推乃 藍」
    static let oshinoAi = OshiTrainer(
        id: "oshino-ai",
        displayName: "推乃 藍",
        firstPerson: "うち",
        secondPerson: "あんた",
        personality: "ツンデレ",
        voiceCharacter: .zundamon,
        imageName: "normal",
        imageDirectory: "OshinoAi"
    )

    /// 利用可能な全トレーナー
    static let allTrainers: [OshiTrainer] = [
        .oshinoAi
        // 将来: カスタムトレーナーがここに追加される
    ]
}