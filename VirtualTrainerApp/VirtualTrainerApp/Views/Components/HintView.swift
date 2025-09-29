import SwiftUI

#if os(iOS)
import UIKit
#endif

/// ヒントメッセージコンポーネント
struct HintView: View {
    let message: String
    @State private var isVisible: Bool = false

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "lightbulb.fill")
                .font(.system(size: 16))
                .foregroundColor(.yellow)

            Text(message)
                .font(.system(size: 14, weight: .regular))
                .foregroundColor(.primary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                #if os(iOS)
                .fill(Color(UIColor.secondarySystemBackground))
                #else
                .fill(Color(NSColor.controlBackgroundColor))
                #endif
        )
        .opacity(isVisible ? 1.0 : 0.0)
        .offset(y: isVisible ? 0 : -10)
        .animation(.easeOut(duration: 0.3), value: isVisible)
        .onAppear {
            withAnimation {
                isVisible = true
            }
        }
    }
}

#Preview {
    VStack(spacing: 20) {
        HintView(message: "左右にスワイプしてトレーナーを選択できます")
        HintView(message: "タップして音声をプレビュー")
        HintView(message: "気に入ったトレーナーを選びましょう")
    }
    .padding()
}