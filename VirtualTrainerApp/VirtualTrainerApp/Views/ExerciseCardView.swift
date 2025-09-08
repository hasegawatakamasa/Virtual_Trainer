import SwiftUI

/// 種目選択カードビュー
struct ExerciseCardView: View {
    let exercise: ExerciseType
    let isHighlighted: Bool
    
    init(exercise: ExerciseType, isHighlighted: Bool = false) {
        self.exercise = exercise
        self.isHighlighted = isHighlighted
    }
    
    var body: some View {
        VStack(spacing: 12) {
            // アイコンとタイトル
            VStack(spacing: 8) {
                Image(systemName: exercise.iconSystemName)
                    .font(.system(size: 32, weight: .semibold))
                    .foregroundColor(exercise.isAvailable ? .primary : .secondary)
                
                Text(exercise.displayName)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(exercise.isAvailable ? .primary : .secondary)
                    .multilineTextAlignment(.center)
            }
            
            // 説明文
            Text(exercise.description)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .lineLimit(2)
            
            // メタデータ（難易度、カロリー）
            HStack(spacing: 16) {
                VStack(spacing: 4) {
                    HStack(spacing: 2) {
                        ForEach(0..<5) { index in
                            Image(systemName: index < exercise.difficulty ? "star.fill" : "star")
                                .font(.caption2)
                                .foregroundColor(index < exercise.difficulty ? .orange : .gray.opacity(0.3))
                        }
                    }
                    
                    Text("難易度")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                Divider()
                    .frame(height: 20)
                
                VStack(spacing: 4) {
                    Text("\(exercise.estimatedCalories)")
                        .font(.caption)
                        .fontWeight(.semibold)
                    
                    Text("kcal/10分")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            
            // 「近日公開」ラベル
            if let comingSoonLabel = exercise.comingSoonLabel {
                Text(comingSoonLabel)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(Color.orange)
                    .cornerRadius(12)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .frame(height: 200)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(cardBackgroundColor)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(borderColor, lineWidth: isHighlighted ? 2 : 1)
                )
        )
        .scaleEffect(exercise.isAvailable ? 1.0 : 0.95)
        .opacity(exercise.isAvailable ? 1.0 : 0.7)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint(accessibilityHint)
        .accessibilityAddTraits(exercise.isAvailable ? [.isButton] : [])
    }
    
    // MARK: - Computed Properties
    
    private var cardBackgroundColor: Color {
        if isHighlighted && exercise.isAvailable {
            return Color.blue.opacity(0.1)
        } else {
            return Color(.systemBackground)
        }
    }
    
    private var borderColor: Color {
        if isHighlighted && exercise.isAvailable {
            return .blue
        } else if exercise.isAvailable {
            return Color(.systemGray4)
        } else {
            return Color(.systemGray5)
        }
    }
    
    private var accessibilityLabel: String {
        var label = exercise.displayName
        label += ", \(exercise.description)"
        label += ", 難易度\(exercise.difficulty)段階"
        label += ", \(exercise.estimatedCalories)カロリー消費"
        
        if !exercise.isAvailable {
            label += ", 近日公開予定"
        }
        
        if isHighlighted {
            label += ", 前回選択した種目"
        }
        
        return label
    }
    
    private var accessibilityHint: String {
        if exercise.isAvailable {
            return "タップして詳細を表示し、トレーニングを開始できます"
        } else {
            return "この種目は現在準備中です"
        }
    }
}

// MARK: - Previews

#Preview("利用可能な種目") {
    VStack(spacing: 20) {
        ExerciseCardView(exercise: .overheadPress)
        ExerciseCardView(exercise: .overheadPress, isHighlighted: true)
    }
    .padding()
    .preferredColorScheme(.dark)
}

#Preview("利用不可能な種目") {
    VStack(spacing: 20) {
        ExerciseCardView(exercise: .squat)
        ExerciseCardView(exercise: .burpee)
    }
    .padding()
    .preferredColorScheme(.dark)
}

#Preview("グリッドレイアウト") {
    let columns = [
        GridItem(.flexible()),
        GridItem(.flexible())
    ]
    
    LazyVGrid(columns: columns, spacing: 16) {
        ForEach(ExerciseType.allCases, id: \.self) { exercise in
            ExerciseCardView(
                exercise: exercise,
                isHighlighted: exercise == .overheadPress
            )
        }
    }
    .padding()
    .preferredColorScheme(.dark)
}