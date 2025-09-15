//
//  FinishOverlayView.swift
//  VirtualTrainerApp
//
//  Created on 2025/09/15.
//

import SwiftUI

/// トレーニング終了オーバーレイコンポーネント
/// 「トレーニング終了！」メッセージと統計情報を表示するオーバーレイ
struct FinishOverlayView: View {
    /// 表示状態を管理するバインディング
    @Binding var isShowing: Bool

    /// 完了したレップ数
    let completedReps: Int?

    /// トレーニング時間（秒）
    let trainingTime: TimeInterval?

    /// カスタムメッセージ（省略可能）
    let customMessage: String?

    /// 自動非表示までの時間（デフォルト3秒、nilで自動非表示なし）
    let autoDismissDelay: Double?

    /// 完了時のコールバック
    let onComplete: (() -> Void)?

    /// タイマーの参照を保持
    @State private var hideTimer: Timer?

    /// アニメーション用の状態
    @State private var contentScale: CGFloat = 0.3
    @State private var contentOpacity: Double = 0.0
    @State private var backgroundOpacity: Double = 0.0
    @State private var celebrationScale: CGFloat = 0.5
    @State private var statisticsOffset: CGFloat = 50

    /// デフォルトイニシャライザー
    init(
        isShowing: Binding<Bool>,
        completedReps: Int? = nil,
        trainingTime: TimeInterval? = nil,
        customMessage: String? = nil,
        autoDismissDelay: Double? = 3.0,
        onComplete: (() -> Void)? = nil
    ) {
        self._isShowing = isShowing
        self.completedReps = completedReps
        self.trainingTime = trainingTime
        self.customMessage = customMessage
        self.autoDismissDelay = autoDismissDelay
        self.onComplete = onComplete
    }

    var body: some View {
        ZStack {
            // 半透明背景オーバーレイ
            Color.black
                .opacity(backgroundOpacity * 0.5)
                .ignoresSafeArea()

            // メインコンテンツ
            mainContent
        }
        .opacity(isShowing ? 1.0 : 0.0)
        .animation(.easeInOut(duration: 0.4), value: isShowing)
        .onAppear {
            if isShowing {
                showFinishOverlay()
            }
        }
        .onChange(of: isShowing) { newValue in
            if newValue {
                showFinishOverlay()
            } else {
                hideFinishOverlay()
            }
        }
        .onDisappear {
            // タイマーをクリーンアップ
            hideTimer?.invalidate()
            hideTimer = nil
        }
    }

    /// メインコンテンツ
    private var mainContent: some View {
        VStack(spacing: 25) {
            // 祝福アイコンとメッセージ
            celebrationHeader

            // 統計情報（利用可能な場合）
            if completedReps != nil || trainingTime != nil {
                statisticsView
            }

            // 終了ボタン（自動非表示が無効の場合）
            if autoDismissDelay == nil {
                dismissButton
            }
        }
        .padding(30)
        .background(finishCardBackground)
        .scaleEffect(contentScale)
        .opacity(contentOpacity)
    }

    /// 祝福ヘッダー部分
    private var celebrationHeader: some View {
        VStack(spacing: 15) {
            // 成功アイコン
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 70))
                .foregroundColor(.green)
                .symbolEffect(.bounce, value: isShowing)
                .scaleEffect(celebrationScale)

            // メインメッセージ
            Text(displayMessage)
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .shadow(color: .black.opacity(0.5), radius: 2, x: 1, y: 1)

            // サブメッセージ
            Text("お疲れ様でした！")
                .font(.headline)
                .foregroundColor(.white.opacity(0.9))
                .shadow(color: .black.opacity(0.3), radius: 1, x: 1, y: 1)
        }
    }

    /// 統計情報ビュー
    private var statisticsView: some View {
        VStack(spacing: 12) {
            // セクションタイトル
            Text("トレーニング結果")
                .font(.headline)
                .foregroundColor(.white.opacity(0.9))
                .padding(.bottom, 5)

            // 統計カード
            VStack(spacing: 8) {
                // レップ数
                if let reps = completedReps {
                    statisticRow(
                        icon: "number.circle.fill",
                        label: "完了レップ数",
                        value: "\(reps) 回",
                        color: .blue
                    )
                }

                // トレーニング時間
                if let time = trainingTime {
                    statisticRow(
                        icon: "timer.circle.fill",
                        label: "トレーニング時間",
                        value: formatTrainingTime(time),
                        color: .orange
                    )
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.white.opacity(0.1))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                    )
            )
        }
        .offset(y: statisticsOffset)
    }

    /// 統計情報の行
    private func statisticRow(icon: String, label: String, value: String, color: Color) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
                .frame(width: 24)

            Text(label)
                .font(.subheadline)
                .foregroundColor(.white.opacity(0.8))

            Spacer()

            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.white)
                .monospacedDigit()
        }
    }

    /// 手動終了ボタン
    private var dismissButton: some View {
        Button(action: {
            hideFinishOverlay()
        }) {
            HStack(spacing: 8) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)

                Text("閉じる")
                    .font(.headline)
                    .fontWeight(.medium)
            }
            .foregroundColor(.white)
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 25)
                    .fill(Color.white.opacity(0.2))
                    .overlay(
                        RoundedRectangle(cornerRadius: 25)
                            .stroke(Color.white.opacity(0.3), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
    }

    /// 背景カードスタイル
    private var finishCardBackground: some View {
        RoundedRectangle(cornerRadius: 24)
            .fill(
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color.green.opacity(0.9),
                        Color.green.darker(by: 0.3)
                    ]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .shadow(color: .black.opacity(0.4), radius: 15, x: 0, y: 8)
            .overlay(
                RoundedRectangle(cornerRadius: 24)
                    .stroke(
                        LinearGradient(
                            gradient: Gradient(colors: [
                                Color.white.opacity(0.3),
                                Color.white.opacity(0.1)
                            ]),
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1
                    )
            )
    }

    /// 表示メッセージの決定
    private var displayMessage: String {
        return customMessage ?? "トレーニング終了！"
    }

    /// 終了オーバーレイ表示アニメーション
    private func showFinishOverlay() {
        // 段階的なアニメーション
        withAnimation(.easeOut(duration: 0.6)) {
            backgroundOpacity = 1.0
            contentScale = 1.0
            contentOpacity = 1.0
        }

        // 祝福アイコンのアニメーション（遅延）
        withAnimation(.spring(response: 0.8, dampingFraction: 0.6).delay(0.2)) {
            celebrationScale = 1.0
        }

        // 統計情報のアニメーション（さらに遅延）
        withAnimation(.easeOut(duration: 0.5).delay(0.4)) {
            statisticsOffset = 0
        }

        // 自動非表示の設定
        if let delay = autoDismissDelay {
            hideTimer?.invalidate()
            hideTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { _ in
                hideFinishOverlay()
            }
        }
    }

    /// 終了オーバーレイ非表示アニメーション
    private func hideFinishOverlay() {
        withAnimation(.easeIn(duration: 0.4)) {
            contentScale = 0.8
            contentOpacity = 0.0
            backgroundOpacity = 0.0
        }

        // アニメーション完了後に状態をリセット
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            isShowing = false
            resetAnimationStates()
            onComplete?()
        }

        // タイマーをクリーンアップ
        hideTimer?.invalidate()
        hideTimer = nil
    }

    /// アニメーション状態をリセット
    private func resetAnimationStates() {
        contentScale = 0.3
        celebrationScale = 0.5
        statisticsOffset = 50
    }

    /// トレーニング時間のフォーマット
    private func formatTrainingTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60

        if minutes > 0 {
            return String(format: "%d分%02d秒", minutes, seconds)
        } else {
            return String(format: "%d秒", seconds)
        }
    }
}

// MARK: - Color Extension
extension Color {
    /// より濃い色を生成するヘルパーメソッド
    func darker(by percentage: Double = 0.2) -> Color {
        if #available(iOS 14.0, *) {
            return self.opacity(1.0 - percentage)
        } else {
            return self
        }
    }
}

// MARK: - Preview
struct FinishOverlayView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            // 統計情報ありのプレビュー
            ZStack {
                LinearGradient(
                    gradient: Gradient(colors: [.blue.opacity(0.3), .purple.opacity(0.3)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                FinishOverlayView(
                    isShowing: .constant(true),
                    completedReps: 15,
                    trainingTime: 125.5,
                    autoDismissDelay: nil,
                    onComplete: {
                        print("Finish overlay completed")
                    }
                )
            }
            .previewDisplayName("完了オーバーレイ（統計あり）")

            // シンプルなプレビュー
            ZStack {
                Color.black.ignoresSafeArea()

                FinishOverlayView(
                    isShowing: .constant(true),
                    customMessage: "素晴らしい！",
                    autoDismissDelay: 3.0
                )
            }
            .previewDisplayName("完了オーバーレイ（シンプル）")
        }
    }
}