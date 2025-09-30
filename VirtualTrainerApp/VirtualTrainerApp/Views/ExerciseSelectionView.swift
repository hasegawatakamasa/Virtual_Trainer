import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

/// 種目選択画面のメインビュー
struct ExerciseSelectionView: View {
    @State private var selectedExercise: ExerciseType?
    @State private var showingVoiceSettings = false
    @State private var showingRecords = false
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

                // 将来拡張予定バナー
                if !ExerciseType.comingSoonExercises.isEmpty {
                    FutureExpansionBanner(comingSoonCount: ExerciseType.comingSoonExercises.count)
                        .padding(.horizontal)
                }
            }
            .padding(.top)
        }
        .navigationTitle("種目選択")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.large)
        #endif
        .sheet(item: $selectedExercise) { exercise in
            ExerciseDetailView(exercise: exercise)
        }
        .sheet(isPresented: $showingVoiceSettings) {
            OshiTrainerSettingsView()
        }
        .sheet(isPresented: $showingRecords) {
            RecordsTabView()
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
                
                HStack(spacing: 16) {
                    // トレーニング記録ボタン
                    Button(action: {
                        showingRecords = true
                    }) {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.title2)
                            .foregroundColor(.green)
                            .frame(width: 32, height: 32)
                    }
                    .accessibilityLabel("トレーニング記録")

                    // 推しトレーナー設定ボタン
                    Button(action: {
                        showingVoiceSettings = true
                    }) {
                        Image(systemName: "figure.stand")
                            .font(.title2)
                            .foregroundColor(.blue)
                            .frame(width: 32, height: 32)
                    }
                    .accessibilityLabel("推しトレーナー設定")
                }
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
                    .background(Color.systemGray5)
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
            }
        }
        .padding(.horizontal)
    }
    
    // MARK: - アクション

    private func handleExerciseSelection(_ exercise: ExerciseType) {
        guard exercise.isAvailable else {
            // 利用不可能な種目の場合はハプティックフィードバック
            #if canImport(UIKit)
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.warning)
            #endif
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