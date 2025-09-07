import Foundation
import UIKit

/// アプリケーションで発生するエラーの種類
enum AppError: LocalizedError, Equatable {
    case cameraPermissionDenied
    case cameraUnavailable
    case modelLoadingFailed(String)
    case inferenceTimeout
    case inferenceError(String)
    case invalidKeypoints
    case networkUnavailable
    case storageError(String)
    case configurationError(String)
    
    /// ユーザー向けエラーメッセージ
    var errorDescription: String? {
        switch self {
        case .cameraPermissionDenied:
            return "カメラへのアクセスが許可されていません"
        case .cameraUnavailable:
            return "カメラが使用できません"
        case .modelLoadingFailed(let model):
            return "AIモデル(\(model))の読み込みに失敗しました"
        case .inferenceTimeout:
            return "AIの処理がタイムアウトしました"
        case .inferenceError(let message):
            return "AI処理エラー: \(message)"
        case .invalidKeypoints:
            return "有効なキーポイントが検出されませんでした"
        case .networkUnavailable:
            return "ネットワーク接続が利用できません"
        case .storageError(let message):
            return "データ保存エラー: \(message)"
        case .configurationError(let message):
            return "設定エラー: \(message)"
        }
    }
    
    /// 回復方法の提案
    var recoverySuggestion: String? {
        switch self {
        case .cameraPermissionDenied:
            return "設定アプリでカメラへのアクセスを許可してください"
        case .cameraUnavailable:
            return "他のアプリでカメラが使用されていないか確認してください"
        case .modelLoadingFailed:
            return "アプリを再起動してください。問題が続く場合は再インストールをお試しください"
        case .inferenceTimeout:
            return "デバイスのパフォーマンスを確認し、他のアプリを終了してください"
        case .inferenceError:
            return "アプリを再起動してください"
        case .invalidKeypoints:
            return "カメラに全身が映るように調整してください。明るい場所で使用してください"
        case .networkUnavailable:
            return "Wi-Fiまたはモバイルデータ接続を確認してください"
        case .storageError:
            return "デバイスの空き容量を確認してください"
        case .configurationError:
            return "アプリの設定を確認してください"
        }
    }
    
    /// ヘルプURL
    var helpAnchor: String? {
        switch self {
        case .cameraPermissionDenied:
            return "camera-permission"
        case .cameraUnavailable:
            return "camera-troubleshooting"
        case .modelLoadingFailed:
            return "model-issues"
        case .inferenceTimeout, .inferenceError:
            return "ai-performance"
        case .invalidKeypoints:
            return "pose-detection-tips"
        case .networkUnavailable:
            return "network-setup"
        case .storageError:
            return "storage-management"
        case .configurationError:
            return "app-settings"
        }
    }
    
    /// エラーの重要度
    enum Severity {
        case low    // ユーザーに影響なし
        case medium // 一部機能が制限
        case high   // アプリが使用不可
        case critical // データ損失の可能性
    }
    
    var severity: Severity {
        switch self {
        case .invalidKeypoints, .inferenceTimeout:
            return .low
        case .inferenceError, .networkUnavailable:
            return .medium
        case .cameraPermissionDenied, .cameraUnavailable, .modelLoadingFailed:
            return .high
        case .storageError, .configurationError:
            return .critical
        }
    }
    
    /// ログ記録用の識別子
    var identifier: String {
        switch self {
        case .cameraPermissionDenied:
            return "CAMERA_PERMISSION_DENIED"
        case .cameraUnavailable:
            return "CAMERA_UNAVAILABLE"
        case .modelLoadingFailed:
            return "MODEL_LOADING_FAILED"
        case .inferenceTimeout:
            return "INFERENCE_TIMEOUT"
        case .inferenceError:
            return "INFERENCE_ERROR"
        case .invalidKeypoints:
            return "INVALID_KEYPOINTS"
        case .networkUnavailable:
            return "NETWORK_UNAVAILABLE"
        case .storageError:
            return "STORAGE_ERROR"
        case .configurationError:
            return "CONFIGURATION_ERROR"
        }
    }
}

/// エラーレポート機能
struct ErrorReport: Codable {
    let error: String
    let identifier: String
    let timestamp: Date
    let deviceInfo: DeviceInfo
    let appVersion: String
    let context: [String: String]
    
    struct DeviceInfo: Codable {
        let model: String
        let systemVersion: String
        let appVersion: String
        
        static var current: DeviceInfo {
            return DeviceInfo(
                model: UIDevice.current.model,
                systemVersion: UIDevice.current.systemVersion,
                appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
            )
        }
    }
    
    init(error: AppError, context: [String: String] = [:]) {
        self.error = error.localizedDescription
        self.identifier = error.identifier
        self.timestamp = Date()
        self.deviceInfo = DeviceInfo.current
        self.appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
        self.context = context
    }
}