import SwiftUI

/// 将来拡張予定バナーコンポーネント
struct FutureExpansionBanner: View {
    let comingSoonCount: Int

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "hourglass")
                    .font(.headline)
                    .foregroundColor(.orange)

                Text("今後\(comingSoonCount)種目が追加予定です")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
            }

            Text("スクワット、腕立て伏せなど、多彩なトレーニングをお楽しみに！")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.orange.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.orange.opacity(0.3), lineWidth: 1)
                )
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("将来の拡張予定: 今後\(comingSoonCount)種目が追加予定です。スクワット、腕立て伏せなど、多彩なトレーニングをお楽しみに！")
    }
}

// MARK: - Previews

#Preview("デフォルト") {
    VStack(spacing: 20) {
        FutureExpansionBanner(comingSoonCount: 5)
        FutureExpansionBanner(comingSoonCount: 3)
    }
    .padding()
    .preferredColorScheme(.dark)
}

#Preview("ExerciseSelectionView統合イメージ") {
    VStack(spacing: 16) {
        Text("トレーニング種目を選択")
            .font(.title2)
            .fontWeight(.bold)

        // 利用可能な種目カード（例）
        ExerciseCardView(exercise: .overheadPress)

        // 将来拡張バナー
        FutureExpansionBanner(comingSoonCount: 5)

        Spacer()
    }
    .padding()
    .preferredColorScheme(.dark)
}