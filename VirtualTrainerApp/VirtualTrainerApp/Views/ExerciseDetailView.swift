import SwiftUI

/// 種目詳細画面とトレーニング開始
struct ExerciseDetailView: View {
    let exercise: ExerciseType
    @Environment(\.dismiss) private var dismiss
    @State private var isStartingTraining = false
    @State private var showingTrainingView = false
    
    // 明示的なイニシャライザー
    init(exercise: ExerciseType) {
        self.exercise = exercise
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // ヘッダー
                headerSection
                
                // 基本情報
                basicInfoSection
                
                // 詳細情報
                detailInfoSection
                
                // 注意事項とポイント
                tipsSection
                
                Spacer(minLength: 100) // ボタンのための余白
            }
            .padding()
        }
        .navigationTitle(exercise.displayName)
        .navigationBarTitleDisplayMode(.large)
        .navigationBarBackButtonHidden(false)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("閉じる") {
                    dismiss()
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            startButton
        }
        .fullScreenCover(isPresented: $showingTrainingView) {
            print("🎬 FullScreenCover presenting ExerciseTrainingView for \(exercise.displayName)")
            return ExerciseTrainingView(exerciseType: exercise)
        }
    }
    
    // MARK: - サブビュー
    
    private var headerSection: some View {
        VStack(spacing: 16) {
            // アイコン
            Image(systemName: exercise.iconSystemName)
                .font(.system(size: 80, weight: .light))
                .foregroundColor(.blue)
            
            // 説明
            Text(exercise.description)
                .font(.title3)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
        }
    }
    
    private var basicInfoSection: some View {
        VStack(spacing: 16) {
            Text("基本情報")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 16) {
                infoCard(
                    icon: "star.fill",
                    title: "難易度",
                    value: exercise.difficultyStars,
                    color: .orange
                )
                
                infoCard(
                    icon: "flame.fill",
                    title: "カロリー消費",
                    value: "\(exercise.estimatedCalories) kcal/10分",
                    color: .red
                )
            }
        }
    }
    
    private var detailInfoSection: some View {
        VStack(spacing: 16) {
            Text("トレーニング詳細")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                detailRow(icon: "target", title: "主な効果", description: exerciseEffects)
                detailRow(icon: "timer", title: "推奨時間", description: "10-15分")
                detailRow(icon: "repeat", title: "推奨回数", description: exerciseReps)
            }
        }
    }
    
    private var tipsSection: some View {
        VStack(spacing: 16) {
            Text("ポイント & 注意事項")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(alignment: .leading, spacing: 8) {
                ForEach(exerciseTips, id: \.self) { tip in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.caption)
                        
                        Text(tip)
                            .font(.caption)
                            .fixedSize(horizontal: false, vertical: true)
                        
                        Spacer()
                    }
                }
            }
            .padding(16)
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
    }
    
    private var startButton: some View {
        VStack(spacing: 12) {
            // 利用不可能な場合の情報
            if !exercise.isAvailable {
                Text("この種目は現在開発中です")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Button(action: startTraining) {
                HStack(spacing: 12) {
                    Image(systemName: "play.fill")
                        .font(.title2)
                    
                    Text(exercise.isAvailable ? "トレーニング開始" : "近日公開")
                        .font(.headline)
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(exercise.isAvailable ? Color.blue : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(16)
                .scaleEffect(isStartingTraining ? 0.95 : 1.0)
            }
            .disabled(!exercise.isAvailable || isStartingTraining)
            .animation(.easeInOut(duration: 0.1), value: isStartingTraining)
        }
        .padding()
        .background(Color(.systemBackground).opacity(0.95))
    }
    
    // MARK: - ヘルパービュー
    
    private func infoCard(icon: String, title: String, value: String, color: Color) -> some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(height: 80)
        .frame(maxWidth: .infinity)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func detailRow(icon: String, title: String, description: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 20)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
        .padding(.vertical, 4)
    }
    
    // MARK: - アクション
    
    private func startTraining() {
        print("🚀 ExerciseDetailView: startTraining called for \(exercise.displayName)")
        guard exercise.isAvailable else { 
            print("❌ Exercise not available: \(exercise.displayName)")
            return 
        }
        
        isStartingTraining = true
        print("📱 Starting training for \(exercise.displayName)")
        
        // 選択を保存
        AppSettings.shared.lastSelectedExercise = exercise
        
        // 履歴に追加
        let historyItem = ExerciseHistoryItem(
            exerciseType: exercise,
            date: Date(),
            repCount: 0,
            accuracy: 0.0
        )
        AppSettings.shared.saveExerciseHistory(historyItem)
        
        // トレーニング画面を表示
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            print("🎬 Showing training view...")
            showingTrainingView = true
            isStartingTraining = false
        }
    }
    
    // MARK: - 種目別の情報
    
    private var exerciseEffects: String {
        switch exercise {
        case .overheadPress:
            return "肩（三角筋）、上腕（上腕三頭筋）、体幹"
        case .squat:
            return "大腿四頭筋、ハムストリング、臀筋、体幹"
        case .plank:
            return "腹筋、背筋、肩、全身の体幹"
        case .pushUp:
            return "胸筋、上腕三頭筋、肩、体幹"
        case .lunge:
            return "大腿四頭筋、臀筋、ハムストリング、バランス感覚"
        case .burpee:
            return "全身（有酸素運動 + 筋力トレーニング）"
        }
    }
    
    private var exerciseReps: String {
        switch exercise {
        case .overheadPress:
            return "8-12回 × 3セット"
        case .squat:
            return "12-15回 × 3セット"
        case .plank:
            return "30-60秒 × 3セット"
        case .pushUp:
            return "8-12回 × 3セット"
        case .lunge:
            return "各脚 8-10回 × 3セット"
        case .burpee:
            return "5-10回 × 3セット"
        }
    }
    
    private var exerciseTips: [String] {
        switch exercise {
        case .overheadPress:
            return [
                "肘が体の前に出過ぎないよう注意",
                "腰を反らさず、体幹を安定させる",
                "重量は適切に調整し、正しいフォームを優先",
                "肩甲骨を下げて安定させる"
            ]
        case .squat:
            return [
                "膝がつま先より前に出ないよう注意",
                "背中をまっすぐ保つ",
                "太ももが床と平行になるまで下げる",
                "かかとに体重をかける"
            ]
        case .plank:
            return [
                "頭から足まで一直線を保つ",
                "お尻を上げすぎたり下げすぎたりしない",
                "呼吸を止めずに続ける",
                "肘は肩の真下に置く"
            ]
        case .pushUp:
            return [
                "体を一直線に保つ",
                "胸がしっかりと床に近づくまで下げる",
                "手は肩幅より少し広めに",
                "腰を落とさない"
            ]
        case .lunge:
            return [
                "前膝が90度になるまで下げる",
                "前膝がつま先より前に出ないよう注意",
                "体幹を安定させる",
                "バランスを保ちながらゆっくりと"
            ]
        case .burpee:
            return [
                "各動作を丁寧に行う",
                "着地時は膝を軽く曲げる",
                "無理せず自分のペースで",
                "水分補給を忘れずに"
            ]
        }
    }
}

// MARK: - Previews

#Preview("オーバーヘッドプレス") {
    NavigationView {
        ExerciseDetailView(exercise: .overheadPress)
    }
    .preferredColorScheme(.dark)
}

#Preview("利用不可能な種目") {
    NavigationView {
        ExerciseDetailView(exercise: .squat)
    }
    .preferredColorScheme(.dark)
}

#Preview("バーピー") {
    NavigationView {
        ExerciseDetailView(exercise: .burpee)
    }
    .preferredColorScheme(.light)
}