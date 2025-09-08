import SwiftUI

/// 前回のトレーニング表示セクション
struct LastWorkoutSection: View {
    let lastExercise: ExerciseType?
    let onExerciseSelected: ((ExerciseType) -> Void)?
    
    init(lastExercise: ExerciseType?, onExerciseSelected: ((ExerciseType) -> Void)? = nil) {
        self.lastExercise = lastExercise
        self.onExerciseSelected = onExerciseSelected
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // セクションヘッダー
            HStack {
                Image(systemName: "clock.arrow.circlepath")
                    .foregroundColor(.blue)
                    .font(.title2)
                
                Text("前回のトレーニング")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Spacer()
            }
            
            // 前回の種目表示または初回メッセージ
            if let exercise = lastExercise {
                lastWorkoutCard(for: exercise)
            } else {
                firstTimeMessage
            }
        }
        .padding(.horizontal)
    }
    
    // MARK: - サブビュー
    
    private func lastWorkoutCard(for exercise: ExerciseType) -> some View {
        Button(action: {
            onExerciseSelected?(exercise)
        }) {
            HStack(spacing: 16) {
                // アイコン
                Image(systemName: exercise.iconSystemName)
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundColor(.blue)
                    .frame(width: 40, height: 40)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(10)
                
                // 種目情報
                VStack(alignment: .leading, spacing: 4) {
                    Text(exercise.displayName)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text("前回選択した種目")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                // 継続表示
                VStack(spacing: 4) {
                    Image(systemName: "arrow.right.circle.fill")
                        .font(.title2)
                        .foregroundColor(.blue)
                    
                    Text("続ける")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(.systemGray6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
        .accessibilityLabel("\(exercise.displayName)を続ける")
        .accessibilityHint("タップして前回の種目を再度選択します")
    }
    
    private var firstTimeMessage: some View {
        HStack(spacing: 16) {
            // ウェルカムアイコン
            Image(systemName: "hand.wave.fill")
                .font(.system(size: 24))
                .foregroundColor(.orange)
                .frame(width: 40, height: 40)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(10)
            
            // メッセージ
            VStack(alignment: .leading, spacing: 4) {
                Text("ようこそ！")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text("下から最初の種目を選んでトレーニングを始めましょう")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
        .accessibilityLabel("初回利用のメッセージ")
        .accessibilityHint("下のリストから種目を選択してください")
    }
}

// MARK: - 便利なイニシャライザー

extension LastWorkoutSection {
    /// AppSettingsから自動的に前回の選択を読み込む
    init(onExerciseSelected: ((ExerciseType) -> Void)? = nil) {
        let settings = AppSettings.shared
        let lastExercise = settings.isFirstLaunch ? nil : settings.lastSelectedExercise
        
        self.init(lastExercise: lastExercise, onExerciseSelected: onExerciseSelected)
    }
}

// MARK: - Previews

#Preview("前回のトレーニングあり") {
    VStack(spacing: 20) {
        LastWorkoutSection(lastExercise: .overheadPress) { exercise in
            print("Selected: \(exercise.displayName)")
        }
        
        Spacer()
    }
    .padding()
    .preferredColorScheme(.dark)
}

#Preview("初回利用") {
    VStack(spacing: 20) {
        LastWorkoutSection(lastExercise: nil)
        
        Spacer()
    }
    .padding()
    .preferredColorScheme(.dark)
}

#Preview("異なる種目") {
    VStack(spacing: 20) {
        LastWorkoutSection(lastExercise: .squat) { exercise in
            print("Selected: \(exercise.displayName)")
        }
        
        LastWorkoutSection(lastExercise: .burpee) { exercise in
            print("Selected: \(exercise.displayName)")
        }
        
        Spacer()
    }
    .padding()
    .preferredColorScheme(.dark)
}