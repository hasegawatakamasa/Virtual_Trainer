import SwiftUI

#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

/// 推しトレーナーの画像表示コンポーネント
struct TrainerImageView: View {
    let trainer: OshiTrainer
    let size: CGSize

    @State private var loadResult: ImageLoadResult?
    @State private var isLoading: Bool = true

    var body: some View {
        ZStack {
            if isLoading {
                loadingView()
            } else if let result = loadResult, let image = result.image {
                characterImageView(image: image)
            } else if let result = loadResult, let error = result.error {
                fallbackIcon()
                    .onAppear {
                        if let oshiError = error as? OshiTrainerError {
                            print("[TrainerImageView] 画像読み込み失敗: \(oshiError.localizedDescription)")
                        } else {
                            print("[TrainerImageView] 画像読み込み失敗: \(error.localizedDescription)")
                        }
                    }
            } else {
                fallbackIcon()
            }
        }
        .frame(width: size.width, height: size.height)
        .clipShape(Circle())
        .accessibilityLabel(accessibilityLabel)
        .task {
            await loadImageAsync()
        }
    }

    /// ローディング表示
    @ViewBuilder
    private func loadingView() -> some View {
        ZStack {
            Color.gray.opacity(0.2)
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle())
        }
    }

    /// キャラクター画像表示
    @ViewBuilder
    private func characterImageView(image: ImageType) -> some View {
        #if os(iOS)
        Image(uiImage: image)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: size.width, height: size.height)
        #elseif os(macOS)
        Image(nsImage: image)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: size.width, height: size.height)
        #endif
    }

    #if os(iOS)
    typealias ImageType = UIImage
    #else
    typealias ImageType = NSImage
    #endif

    /// フォールバックアイコン
    @ViewBuilder
    private func fallbackIcon() -> some View {
        ZStack {
            Color.gray.opacity(0.2)
            Image(systemName: "figure.stand")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: size.width * 0.5, height: size.height * 0.5)
                .foregroundColor(.gray)
        }
    }

    /// アクセシビリティラベル
    private var accessibilityLabel: String {
        if let result = loadResult, case .success = result {
            return trainer.accessibilityDescription
        } else {
            return "\(trainer.displayName)のキャラクター画像（読み込み中または利用不可）"
        }
    }

    /// 画像を非同期で読み込み
    private func loadImageAsync() async {
        let startTime = CFAbsoluteTimeGetCurrent()

        let result = await Task.detached(priority: .userInitiated) {
            return loadCharacterImageSync(trainer: trainer)
        }.value

        let loadTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        await MainActor.run {
            self.loadResult = result
            self.isLoading = false

            if loadTime > 100 {
                print("[TrainerImageView] 画像ロード時間: \(String(format: "%.2f", loadTime))ms （警告: 100ms超過）")
            } else {
                print("[TrainerImageView] 画像ロード完了: \(String(format: "%.2f", loadTime))ms")
            }
        }
    }

    /// 画像を同期的に読み込む（バックグラウンドスレッド用）
    nonisolated private func loadCharacterImageSync(trainer: OshiTrainer) -> ImageLoadResult {
        guard let imageURL = trainer.imageFileURL() else {
            let error = OshiTrainerError.imageLoadFailed(trainer: trainer, underlyingError: NSError(domain: "TrainerImageView", code: 404, userInfo: [NSLocalizedDescriptionKey: "Image file not found"]))
            return .fallback(error)
        }

        do {
            let imageData = try Data(contentsOf: imageURL)

            #if os(iOS)
            if let image = UIImage(data: imageData) {
                return .success(image)
            } else {
                let error = OshiTrainerError.imageLoadFailed(trainer: trainer, underlyingError: NSError(domain: "TrainerImageView", code: 500, userInfo: [NSLocalizedDescriptionKey: "Invalid image data"]))
                return .fallback(error)
            }
            #elseif os(macOS)
            if let image = NSImage(data: imageData) {
                return .success(image)
            } else {
                let error = OshiTrainerError.imageLoadFailed(trainer: trainer, underlyingError: NSError(domain: "TrainerImageView", code: 500, userInfo: [NSLocalizedDescriptionKey: "Invalid image data"]))
                return .fallback(error)
            }
            #endif
        } catch {
            return .fallback(OshiTrainerError.imageLoadFailed(trainer: trainer, underlyingError: error))
        }
    }
}

#Preview {
    TrainerImageView(trainer: .oshinoAi, size: CGSize(width: 120, height: 120))
        .padding()
}