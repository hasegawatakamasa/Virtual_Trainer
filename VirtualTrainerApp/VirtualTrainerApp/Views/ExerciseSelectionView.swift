import SwiftUI

/// 種目選択画面のメインビュー
struct ExerciseSelectionView: View {
    @State private var selectedExercise: ExerciseType?
    @AppStorage("lastSelectedExercise") private var lastSelectedExerciseRaw: String = ExerciseType.overheadPress.rawValue
    
    // 前回選択した種目
    private var lastSelectedExercise: ExerciseType {
        ExerciseType(rawValue: lastSelectedExerciseRaw) ?? .overheadPress
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // ヘッダー
                headerSection
                
                // 前回のトレーニングセクション
                LastWorkoutSection(lastExercise: AppSettings.shared.isFirstLaunch ? nil : lastSelectedExercise) { exercise in
                    handleExerciseSelection(exercise)
                }
                
                // 種目選択セクション
                exerciseSelectionSection
            }
            .padding(.top)
        }
        .navigationTitle("種目選択")
        .navigationBarTitleDisplayMode(.large)
        .sheet(item: $selectedExercise) { exercise in
            ExerciseDetailView(exercise: exercise)
        }
        .onAppear {
            // 初回起動時の設定
            if AppSettings.shared.isFirstLaunch {
                AppSettings.shared.isFirstLaunch = false
            }
        }
    }
    
    // MARK: - サブビュー
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 8) {
                    Text("今日のトレーニング")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text("種目を選んで始めましょう")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                // トレーニングアイコン
                Image(systemName: "figure.strengthtraining.traditional")
                    .font(.system(size: 40))
                    .foregroundColor(.blue)
            }
        }
        .padding(.horizontal)
    }
    
    private var exerciseSelectionSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            // セクションヘッダー
            HStack {
                Text("種目一覧")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Text("\(ExerciseType.availableExercises.count)/\(ExerciseType.allCases.count) 利用可能")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(.systemGray5))
                    .cornerRadius(8)
            }
            .padding(.horizontal)
            
            // 種目グリッド
            exerciseGrid
        }
    }
    
    private var exerciseGrid: some View {
        let columns = [
            GridItem(.flexible(), spacing: 12),
            GridItem(.flexible(), spacing: 12)
        ]
        
        return LazyVGrid(columns: columns, spacing: 16) {
            ForEach(ExerciseType.allCases, id: \.self) { exercise in
                ExerciseCardView(
                    exercise: exercise,
                    isHighlighted: !AppSettings.shared.isFirstLaunch && exercise == lastSelectedExercise
                )
                .onTapGesture {
                    handleExerciseSelection(exercise)
                }
                .disabled(!exercise.isAvailable)
            }
        }
        .padding(.horizontal)
    }
    
    // MARK: - アクション
    
    private func handleExerciseSelection(_ exercise: ExerciseType) {
        guard exercise.isAvailable else {
            // 利用不可能な種目の場合は何もしない
            // 将来的にはアラートを表示可能
            return
        }
        
        // 最後に選択した種目を保存
        lastSelectedExerciseRaw = exercise.rawValue
        AppSettings.shared.lastSelectedExercise = exercise
        
        // シートを表示
        selectedExercise = exercise
    }
}

// MARK: - Previews

#Preview("通常状態") {
    NavigationStack {
        ExerciseSelectionView()
    }
    .preferredColorScheme(.dark)
}

#Preview("初回起動") {
    NavigationStack {
        ExerciseSelectionView()
    }
    .preferredColorScheme(.dark)
    .onAppear {
        // 初回起動をシミュレート
        AppSettings.shared.isFirstLaunch = true
    }
}

#Preview("ライトモード") {
    NavigationStack {
        ExerciseSelectionView()
    }
    .preferredColorScheme(.light)
}