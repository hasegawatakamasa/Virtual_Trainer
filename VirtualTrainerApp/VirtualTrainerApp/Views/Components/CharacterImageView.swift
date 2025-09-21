import SwiftUI
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

/// キャラクター画像表示コンポーネント
struct CharacterImageView: View {
    let character: VoiceCharacter
    let size: CGSize
    
    @State private var loadResult: ImageLoadResult? = nil
    @State private var isLoading: Bool = true
    
    var body: some View {
        Group {
            if isLoading {
                loadingView()
            } else if let result = loadResult, let image = result.image {
                characterImageView(image: image)
            } else {
                fallbackIcon()
            }
        }
        .accessibilityLabel(accessibilityLabel)
        .onAppear {
            loadImageAsync()
        }
    }
    
    // MARK: - Private Methods
    
    /// 非同期画像読み込み
    private func loadImageAsync() {
        Task {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            // 非同期でバックグラウンドスレッドで実行
            let result = await Task.detached { [character] in
                return CharacterImageView.loadCharacterImageSync(character: character)
            }.value
            
            let loadTime = CFAbsoluteTimeGetCurrent() - startTime
            
            await MainActor.run {
                self.loadResult = result
                self.isLoading = false
                
                // パフォーマンスログ
                if loadTime > 0.1 {
                    print("[CharacterImageView] Warning: Image load took \(String(format: "%.2f", loadTime * 1000))ms for \(character.displayName)")
                } else {
                    print("[CharacterImageView] Image loaded in \(String(format: "%.2f", loadTime * 1000))ms for \(character.displayName)")
                }
            }
        }
    }
    
    /// 静的な画像読み込みメソッド
    nonisolated private static func loadCharacterImageSync(character: VoiceCharacter) -> ImageLoadResult {
        guard let imageURL = character.imageFileURL() else {
            let error = CharacterImageError.imageNotFound(character: character)
            print("[CharacterImageView] \(error.debugDescription)")
            return .fallback(error)
        }
        
        do {
            let imageData = try Data(contentsOf: imageURL)
            #if os(iOS)
            guard let uiImage = UIImage(data: imageData) else {
                let error = CharacterImageError.invalidImageData(character: character)
                print("[CharacterImageView] \(error.debugDescription)")
                return .fallback(error)
            }
            #else
            guard let nsImage = NSImage(data: imageData) else {
                let error = CharacterImageError.invalidImageData(character: character)
                print("[CharacterImageView] \(error.debugDescription)")
                return .fallback(error)
            }
            #endif
            
            // メモリ使用量チェック
            let imageSize = imageData.count / (1024 * 1024) // MB
            if imageSize > 2 {
                print("[CharacterImageView] Warning: Large image size \(imageSize)MB for \(character.displayName)")
            }
            
            print("[CharacterImageView] Successfully loaded image for \(character.displayName)")
            #if os(iOS)
            return .success(uiImage)
            #else
            return .success(nsImage)
            #endif
            
        } catch {
            let characterError = CharacterImageError.bundleResourceMissing(path: imageURL.path)
            print("[CharacterImageView] \(characterError.debugDescription), underlying error: \(error)")
            return .fallback(characterError)
        }
    }
    
    /// ローディング表示
    private func loadingView() -> some View {
        ProgressView()
            .frame(width: size.width, height: size.height)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.systemGray6)
                    .opacity(0.3)
            )
    }
    
    /// キャラクター画像表示
    #if os(iOS)
    private func characterImageView(image: UIImage) -> some View {
        Image(uiImage: image)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(maxWidth: size.width, maxHeight: size.height)
            .clipped()
            .cornerRadius(8)
    }
    #else
    private func characterImageView(image: NSImage) -> some View {
        Image(nsImage: image)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(maxWidth: size.width, maxHeight: size.height)
            .clipped()
            .cornerRadius(8)
    }
    #endif
    
    /// フォールバック時のシステムアイコン表示
    private func fallbackIcon() -> some View {
        Image(systemName: character.iconName)
            .font(.system(size: min(size.width, size.height) * 0.6, weight: .medium))
            .foregroundColor(.accentColor)
            .frame(width: size.width, height: size.height)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.systemGray6)
                    .opacity(0.3)
            )
    }
    
    /// アクセシビリティラベル
    private var accessibilityLabel: String {
        if let result = loadResult, result.error != nil {
            return "\(character.displayName)のアイコン（画像読み込みエラー）: \(character.description)"
        } else if isLoading {
            return "\(character.displayName)の画像を読み込み中"
        } else {
            return character.accessibilityImageDescription
        }
    }
}

// MARK: - Preview
#Preview {
    VStack(spacing: 20) {
        CharacterImageView(
            character: .zundamon,
            size: CGSize(width: 80, height: 80)
        )
        
        CharacterImageView(
            character: .shikokuMetan,
            size: CGSize(width: 80, height: 80)
        )
    }
    .padding()
}