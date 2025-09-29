import SwiftUI

/// 推しトレーナー設定画面
struct OshiTrainerSettingsView: View {
    @ObservedObject private var oshiTrainerSettings = OshiTrainerSettings.shared
    @Environment(\.dismiss) private var dismiss

    @State private var showSuccessMessage: Bool = false
    @State private var showInitialHint: Bool = false
    @State private var selectedTrainerForMessage: OshiTrainer?

    var body: some View {
        NavigationStack {
            ZStack {
                ScrollView {
                    VStack(spacing: 32) {
                        // ヘッダーセクション
                        headerSection

                        // 初回ヒント
                        if showInitialHint {
                            HintView(message: "左右にスワイプしてトレーナーを選択できます")
                                .transition(.opacity.combined(with: .move(edge: .top)))
                        }

                        // トレーナー選択カルーセル
                        SwipableTrainerSelectionView { trainer in
                            handleTrainerSelection(trainer)
                        }
                    }
                    .padding()
                }

                // 成功メッセージオーバーレイ
                if showSuccessMessage, let trainer = selectedTrainerForMessage {
                    TrainerSelectionSuccessMessage(trainer: trainer, isPresented: $showSuccessMessage)
                }
            }
            .navigationTitle("推しトレーナー選択")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完了") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                checkInitialHint()
            }
        }
    }

    // MARK: - Components

    /// ヘッダーセクション
    private var headerSection: some View {
        VStack(spacing: 16) {
            Image(systemName: "figure.stand")
                .font(.system(size: 60))
                .foregroundColor(.accentColor)

            Text("あなたのトレーニングパートナーを選んでください")
                .font(.system(size: 18, weight: .medium))
                .multilineTextAlignment(.center)
                .foregroundColor(.primary)
        }
        .padding(.top, 16)
    }

    // MARK: - Helper Methods

    /// トレーナー選択処理
    private func handleTrainerSelection(_ trainer: OshiTrainer) {
        selectedTrainerForMessage = trainer

        withAnimation {
            showSuccessMessage = true
        }

        // 2秒後に成功メッセージを非表示
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            withAnimation {
                showSuccessMessage = false
            }
        }
    }

    /// 初回ヒント表示チェック
    private func checkInitialHint() {
        let hasSeenHintKey = "hasSeenTrainerSelectionHint"

        if !UserDefaults.standard.bool(forKey: hasSeenHintKey) {
            withAnimation {
                showInitialHint = true
            }

            // UserDefaultsに保存
            UserDefaults.standard.set(true, forKey: hasSeenHintKey)

            // 3秒後にヒントを非表示
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                withAnimation {
                    showInitialHint = false
                }
            }
        }
    }
}

#Preview {
    OshiTrainerSettingsView()
}