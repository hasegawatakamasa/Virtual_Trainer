import SwiftUI

/// 近日公開種目のクイックプレビューオーバーレイ
struct QuickPreviewOverlay: View {
    let exercise: ExerciseType
    @Binding var isPresented: Bool

    var body: some View {
        ZStack {
            // 背景ディム
            Color.black.opacity(0.6)
                .ignoresSafeArea()
                .onTapGesture {
                    isPresented = false
                }

            // プレビューカード
            VStack(spacing: 20) {
                // ヘッダー
                HStack {
                    Image(systemName: exercise.iconSystemName)
                        .font(.title2)
                        .foregroundColor(.orange)

                    Text(exercise.displayName)
                        .font(.title2)
                        .fontWeight(.bold)

                    Spacer()

                    Button(action: {
                        isPresented = false
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(.secondary)
                    }
                }

                Divider()

                // 種目情報
                VStack(alignment: .leading, spacing: 16) {
                    InfoRow(
                        icon: "text.alignleft",
                        label: "説明",
                        value: exercise.description
                    )

                    InfoRow(
                        icon: "target",
                        label: "目標",
                        value: exercise.targetDisplayText
                    )

                    InfoRow(
                        icon: "flame.fill",
                        label: "推定カロリー",
                        value: "\(exercise.estimatedCalories) kcal/10分"
                    )

                    InfoRow(
                        icon: "star.fill",
                        label: "難易度",
                        value: exercise.difficultyStars
                    )
                }

                Divider()

                // 近日公開メッセージ
                VStack(spacing: 8) {
                    Image(systemName: "hourglass")
                        .font(.largeTitle)
                        .foregroundColor(.orange)

                    Text("この種目は近日公開予定です")
                        .font(.headline)
                        .fontWeight(.semibold)

                    Text("もうしばらくお待ちください")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.orange.opacity(0.1))
                )
            }
            .padding(24)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color.systemBackground)
            )
            .padding(40)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(exercise.displayName)のプレビュー")
        .accessibilityHint("この種目は近日公開予定です")
    }
}

// MARK: - Subviews

private struct InfoRow: View {
    let icon: String
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.body)
                .foregroundColor(.blue)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 4) {
                Text(label)
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text(value)
                    .font(.body)
                    .foregroundColor(.primary)
            }

            Spacer()
        }
    }
}

// MARK: - Previews

#Preview("スクワット") {
    QuickPreviewOverlay(
        exercise: .squat,
        isPresented: .constant(true)
    )
    .preferredColorScheme(.dark)
}

#Preview("腕立て伏せ") {
    QuickPreviewOverlay(
        exercise: .pushUp,
        isPresented: .constant(true)
    )
    .preferredColorScheme(.dark)
}

#Preview("サイドレイズ") {
    QuickPreviewOverlay(
        exercise: .sideRaise,
        isPresented: .constant(true)
    )
    .preferredColorScheme(.dark)
}