import Foundation
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

/// キャラクター画像関連のエラー
enum CharacterImageError: Error {
    case imageNotFound(character: VoiceCharacter)
    case invalidImageData(character: VoiceCharacter)
    case bundleResourceMissing(path: String)
    case memoryAllocationFailed
    
    /// エラーメッセージ
    var localizedDescription: String {
        switch self {
        case .imageNotFound(let character):
            return "Character image not found for: \(character.displayName)"
        case .invalidImageData(let character):
            return "Invalid image data for character: \(character.displayName)"
        case .bundleResourceMissing(let path):
            return "Bundle resource missing at path: \(path)"
        case .memoryAllocationFailed:
            return "Failed to allocate memory for image"
        }
    }
    
    /// デバッグ用の詳細な説明
    var debugDescription: String {
        switch self {
        case .imageNotFound(let character):
            return "Image file '\(character.imageName).png' not found in bundle for character '\(character.displayName)'"
        case .invalidImageData(let character):
            return "Image data could not be converted to UIImage for character '\(character.displayName)'"
        case .bundleResourceMissing(let path):
            return "Expected bundle resource at path '\(path)' does not exist"
        case .memoryAllocationFailed:
            return "System failed to allocate sufficient memory for image processing"
        }
    }
}

/// 画像読み込み結果
enum ImageLoadResult {
    #if os(iOS)
    case success(UIImage)
    #else
    case success(NSImage)
    #endif
    case fallback(Error)

    /// 成功時の画像を取得
    #if os(iOS)
    var image: UIImage? {
        switch self {
        case .success(let image):
            return image
        case .fallback:
            return nil
        }
    }
    #else
    var image: NSImage? {
        switch self {
        case .success(let image):
            return image
        case .fallback:
            return nil
        }
    }
    #endif
    
    /// エラー情報を取得
    var error: Error? {
        switch self {
        case .success:
            return nil
        case .fallback(let error):
            return error
        }
    }
}