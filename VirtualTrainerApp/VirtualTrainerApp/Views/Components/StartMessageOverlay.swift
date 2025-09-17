//
//  StartMessageOverlay.swift
//  VirtualTrainerApp
//
//  Created on 2025/09/15.
//

import SwiftUI

/// トレーニング開始メッセージオーバーレイコンポーネント
/// 「トレーニング開始！」メッセージを2秒間表示するオーバーレイ
struct StartMessageOverlay: View {
    /// 表示状態を管理するバインディング
    @Binding var isShowing: Bool

    /// 表示メッセージ（カスタマイズ可能）
    let message: String

    /// 表示時間（デフォルト2秒）
    let displayDuration: Double

    /// 完了時のコールバック
    let onComplete: (() -> Void)?

    /// タイマーの参照を保持
    @State private var hideTimer: Timer?

    /// アニメーション用の状態
    @State private var messageScale: CGFloat = 0.5
    @State private var messageOpacity: Double = 0.0
    @State private var backgroundOpacity: Double = 0.0

    /// デフォルトイニシャライザー
    init(
        isShowing: Binding<Bool>,
        message: String = "トレーニング開始！",
        displayDuration: Double = 2.0,
        onComplete: (() -> Void)? = nil
    ) {
        self._isShowing = isShowing
        self.message = message
        self.displayDuration = displayDuration
        self.onComplete = onComplete
    }

    var body: some View {
        ZStack {
            // 半透明背景オーバーレイ
            Color.black
                .opacity(backgroundOpacity * 0.4)
                .ignoresSafeArea()

            // メッセージコンテンツ
            messageContent
        }
        .opacity(isShowing ? 1.0 : 0.0)
        .animation(.easeInOut(duration: 0.3), value: isShowing)
        .onAppear {
            if isShowing {
                showMessage()
            }
        }
        .onChange(of: isShowing) { _, newValue in
            if newValue {
                showMessage()
            } else {
                hideMessage()
            }
        }
        .onDisappear {
            // タイマーをクリーンアップ
            hideTimer?.invalidate()
            hideTimer = nil
        }
    }

    /// メッセージコンテンツ
    private var messageContent: some View {
        VStack(spacing: 20) {
            // 開始アイコン
            Image(systemName: "play.circle.fill")
                .font(.system(size: 60))
                .foregroundColor(.green)
                .symbolEffect(.bounce, value: isShowing)
                .scaleEffect(messageScale)
                .opacity(messageOpacity)

            // メッセージテキスト
            Text(message)
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .scaleEffect(messageScale)
                .opacity(messageOpacity)
                .shadow(color: .black.opacity(0.5), radius: 2, x: 1, y: 1)

            // サブメッセージ
            Text("頑張って！")
                .font(.headline)
                .foregroundColor(.white.opacity(0.9))
                .scaleEffect(messageScale * 0.9)
                .opacity(messageOpacity * 0.8)
                .shadow(color: .black.opacity(0.3), radius: 1, x: 1, y: 1)
        }
        .padding(40)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color.green.opacity(0.8),
                            Color.green.darker().opacity(0.9)
                        ]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .shadow(color: .black.opacity(0.3), radius: 10, x: 0, y: 5)
        )
        .scaleEffect(messageScale)
        .opacity(messageOpacity)
    }

    /// メッセージ表示アニメーション
    private func showMessage() {
        // アニメーション開始
        withAnimation(.easeOut(duration: 0.5)) {
            backgroundOpacity = 1.0
            messageScale = 1.0
            messageOpacity = 1.0
        }

        // 指定時間後に自動的に非表示
        hideTimer?.invalidate()
        hideTimer = Timer.scheduledTimer(withTimeInterval: displayDuration, repeats: false) { _ in
            hideMessage()
        }
    }

    /// メッセージ非表示アニメーション
    private func hideMessage() {
        withAnimation(.easeIn(duration: 0.3)) {
            messageScale = 0.8
            messageOpacity = 0.0
            backgroundOpacity = 0.0
        }

        // アニメーション完了後に状態をリセット
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            isShowing = false
            messageScale = 0.5
            onComplete?()
        }

        // タイマーをクリーンアップ
        hideTimer?.invalidate()
        hideTimer = nil
    }
}

// MARK: - Color Extension
extension Color {
    /// より濃い色を生成するヘルパーメソッド
    func darker(by percentage: Double = 0.2) -> Color {
        return self.opacity(1.0 - percentage)
    }
}

// MARK: - Preview
struct StartMessageOverlay_Previews: PreviewProvider {
    static var previews: some View {
        ZStack {
            // 背景（カメラプレビューを模擬）
            LinearGradient(
                gradient: Gradient(colors: [.blue.opacity(0.3), .purple.opacity(0.3)]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            // オーバーレイプレビュー
            StartMessageOverlay(
                isShowing: .constant(true),
                message: "トレーニング開始！",
                displayDuration: 2.0,
                onComplete: {
                    print("Start message overlay completed")
                }
            )
        }
        .previewDisplayName("開始メッセージオーバーレイ")
    }
}