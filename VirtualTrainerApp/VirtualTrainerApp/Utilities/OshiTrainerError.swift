import Foundation

/// 推しトレーナー関連エラー
enum OshiTrainerError: LocalizedError {
    case trainerNotFound(id: String)
    case imageLoadFailed(trainer: OshiTrainer, underlyingError: Error)
    case audioPreviewFailed(trainer: OshiTrainer, audioType: AudioType)
    case migrationFailed(reason: String)

    var errorDescription: String? {
        switch self {
        case .trainerNotFound(let id):
            return "指定されたトレーナーが見つかりません: \(id)"
        case .imageLoadFailed(let trainer, let error):
            return "\(trainer.displayName)の画像読み込みに失敗: \(error.localizedDescription)"
        case .audioPreviewFailed(let trainer, let audioType):
            return "\(trainer.displayName)のプレビュー音声（\(audioType)）の再生に失敗"
        case .migrationFailed(let reason):
            return "データ移行に失敗: \(reason)"
        }
    }
}