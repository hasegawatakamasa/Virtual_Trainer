import SwiftUI

#if os(iOS)
import UIKit
#endif

/// トレーナー選択成功メッセージコンポーネント
struct TrainerSelectionSuccessMessage: View {
    let trainer: OshiTrainer
    @Binding var isPresented: Bool

    var body: some View {
        ZStack {
            // 半透明背景
            Color.black.opacity(0.4)
                .edgesIgnoringSafeArea(.all)

            // 成功メッセージカード
            VStack(spacing: 16) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 50))
                    .foregroundColor(.green)

                Text("\(trainer.displayName)が")
                    .font(.system(size: 18, weight: .medium))

                Text("あなたのトレーナーになりました！")
                    .font(.system(size: 20, weight: .bold))
                    .multilineTextAlignment(.center)
            }
            .padding(32)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    #if os(iOS)
                    .fill(Color(UIColor.systemBackground))
                    #else
                    .fill(Color(NSColor.windowBackgroundColor))
                    #endif
                    .shadow(color: Color.black.opacity(0.2), radius: 10, x: 0, y: 5)
            )
            .padding(40)
        }
        .opacity(isPresented ? 1.0 : 0.0)
        .scaleEffect(isPresented ? 1.0 : 0.8)
        .animation(.spring(response: 0.4, dampingFraction: 0.7), value: isPresented)
        .onAppear {
            // 2秒後に自動的に非表示
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                withAnimation {
                    isPresented = false
                }
            }
        }
    }
}

#Preview {
    TrainerSelectionSuccessMessage(
        trainer: .oshinoAi,
        isPresented: .constant(true)
    )
}