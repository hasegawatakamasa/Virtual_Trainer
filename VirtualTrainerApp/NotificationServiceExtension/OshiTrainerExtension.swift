//
//  OshiTrainerExtension.swift
//  NotificationServiceExtension
//
//  NotificationServiceExtension専用のOshiTrainer定義
//

import Foundation

/// 推しトレーナーデータモデル（Extension専用）
struct OshiTrainer: Identifiable, Codable, Equatable {
    let id: String
    let displayName: String
    let imageName: String
    let imageDirectory: String

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
}

extension OshiTrainer {
    /// デフォルト推しトレーナー「推乃 藍」
    static let oshinoAi = OshiTrainer(
        id: "oshino-ai",
        displayName: "推乃 藍",
        imageName: "normal",
        imageDirectory: "OshinoAi"
    )

    /// 利用可能な全トレーナー
    static let allTrainers: [OshiTrainer] = [
        .oshinoAi
    ]
}
