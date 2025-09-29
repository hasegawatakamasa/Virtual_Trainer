import SwiftUI

#if os(iOS)
import UIKit
#endif

/// スワイプ可能な推しトレーナー選択ビュー
struct SwipableTrainerSelectionView: View {
    let onSelection: (OshiTrainer) -> Void

    @ObservedObject private var oshiTrainerSettings = OshiTrainerSettings.shared
    @ObservedObject private var previewService = VoicePreviewService.shared

    @State private var selectedIndex = 0
    @State private var dragOffset: CGFloat = 0

    private let trainers = OshiTrainer.allTrainers

    var body: some View {
        VStack(spacing: 24) {
            // キャラクターカルーセル
            GeometryReader { geometry in
                characterCarousel(geometry: geometry)
            }
            .frame(height: 300)

            // ページインジケーター
            if trainers.count > 1 {
                pageIndicator
            }

            // 音声プレビューボタン
            VoicePreviewButton(trainer: currentTrainer)

            // 選択ボタン
            selectionButton
        }
        .padding()
        .onAppear {
            initializeSelection()
        }
    }

    // MARK: - Components

    /// キャラクターカルーセル
    private func characterCarousel(geometry: GeometryProxy) -> some View {
        HStack(spacing: 40) {
            ForEach(Array(trainers.enumerated()), id: \.element.id) { index, trainer in
                characterCard(for: trainer, at: index, geometry: geometry)
            }
        }
        .offset(x: calculateCarouselOffset(geometry: geometry))
        .gesture(
            DragGesture()
                .onChanged { value in
                    dragOffset = value.translation.width
                }
                .onEnded { value in
                    handleDragEnded(translation: value.translation.width)
                }
        )
    }

    /// トレーナーカード
    private func characterCard(for trainer: OshiTrainer, at index: Int, geometry: GeometryProxy) -> some View {
        let isSelected = index == selectedIndex
        let scale: CGFloat = isSelected ? 1.0 : 0.75
        let opacity: Double = isSelected ? 1.0 : 0.5

        return VStack(spacing: 16) {
            TrainerImageView(trainer: trainer, size: CGSize(width: 180, height: 180))

            VStack(spacing: 4) {
                Text(trainer.displayName)
                    .font(.system(size: 22, weight: .bold))

                Text(trainer.personality)
                    .font(.system(size: 16, weight: .regular))
                    .foregroundColor(.secondary)
            }
        }
        .scaleEffect(scale)
        .opacity(opacity)
        .frame(width: geometry.size.width)
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: selectedIndex)
    }

    /// ページインジケーター
    private var pageIndicator: some View {
        HStack(spacing: 8) {
            ForEach(0..<trainers.count, id: \.self) { index in
                Circle()
                    .fill(index == selectedIndex ? Color.accentColor : Color.gray.opacity(0.3))
                    .frame(width: 8, height: 8)
                    .animation(.easeInOut(duration: 0.2), value: selectedIndex)
            }
        }
    }

    /// 選択ボタン
    private var selectionButton: some View {
        Button(action: {
            handleTrainerSelection()
        }) {
            Text("\(currentTrainer.displayName)を選択")
                .font(.system(size: 18, weight: .semibold))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.accentColor)
                )
        }
        .buttonStyle(PlainButtonStyle())
        .accessibilityLabel("\(currentTrainer.displayName)をトレーナーとして選択")
    }

    // MARK: - Helper Properties

    /// 現在選択中のトレーナー
    private var currentTrainer: OshiTrainer {
        trainers[selectedIndex]
    }

    // MARK: - Helper Methods

    /// カルーセルのオフセット計算
    private func calculateCarouselOffset(geometry: GeometryProxy) -> CGFloat {
        let screenWidth = geometry.size.width
        let baseOffset = -CGFloat(selectedIndex) * (screenWidth + 40)
        return baseOffset + dragOffset
    }

    /// ドラッグ終了処理
    private func handleDragEnded(translation: CGFloat) {
        let threshold: CGFloat = 50

        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
            if translation < -threshold && selectedIndex < trainers.count - 1 {
                selectedIndex += 1
                playPreviewForCurrentTrainer()
            } else if translation > threshold && selectedIndex > 0 {
                selectedIndex -= 1
                playPreviewForCurrentTrainer()
            }
            dragOffset = 0
        }
    }

    /// トレーナー選択処理
    private func handleTrainerSelection() {
        let trainer = currentTrainer

        // ハプティックフィードバック
        provideHapticFeedback()

        // トレーナー更新
        oshiTrainerSettings.updateTrainer(trainer)

        // クロージャ呼び出し
        onSelection(trainer)

        print("[SwipableTrainerSelectionView] トレーナーが選択されました: \(trainer.displayName)")
    }

    /// 初期化処理
    private func initializeSelection() {
        // 現在選択中のトレーナーのインデックスを設定
        if let index = trainers.firstIndex(of: oshiTrainerSettings.selectedTrainer) {
            selectedIndex = index
        }
    }

    /// プレビュー音声再生
    private func playPreviewForCurrentTrainer() {
        // スワイプ時の自動再生は任意実装
        // 必要に応じてpreviewService.playRandomPreview(for: currentTrainer)を呼び出し
    }

    /// ハプティックフィードバック
    private func provideHapticFeedback() {
        #if os(iOS)
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.prepare()
        generator.impactOccurred()
        #endif
    }
}

#Preview {
    SwipableTrainerSelectionView { trainer in
        print("Selected: \(trainer.displayName)")
    }
}