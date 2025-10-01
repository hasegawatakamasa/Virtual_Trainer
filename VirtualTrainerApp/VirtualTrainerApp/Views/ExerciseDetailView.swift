import SwiftUI
#if os(iOS)
import UIKit
#endif

/// 種目詳細画面とトレーニング開始
struct ExerciseDetailView: View {
    let exercise: ExerciseType
    @Environment(\.dismiss) private var dismiss
    @State private var isStartingTraining = false
    @State private var showingTrainingView = false
    @State private var showingResultView = false
    @State private var sessionCompletionData: SessionCompletionData?
    
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
        #if os(iOS)
        .navigationBarTitleDisplayMode(.large)
        #endif
        .navigationBarBackButtonHidden(false)
        .toolbar {
            #if os(iOS)
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("閉じる") {
                    dismiss()
                }
            }
            #else
            ToolbarItem(placement: .primaryAction) {
                Button("閉じる") {
                    dismiss()
                }
            }
            #endif
        }
        .safeAreaInset(edge: .bottom) {
            startButton
        }
        #if os(iOS)
        .fullScreenCover(isPresented: $showingTrainingView) {
            print("🎬 FullScreenCover presenting ExerciseTrainingView for \(exercise.displayName)")
            return ExerciseTrainingView(
                exerciseType: exercise,
                onCompletion: { completionData in
                    // トレーニング完了時の処理
                    sessionCompletionData = completionData
                    showingResultView = true
                }
            )
        }
        #else
        .sheet(isPresented: $showingTrainingView) {
            print("🎬 Sheet presenting ExerciseTrainingView for \(exercise.displayName)")
            return ExerciseTrainingView(
                exerciseType: exercise,
                onCompletion: { completionData in
                    // トレーニング完了時の処理
                    sessionCompletionData = completionData
                    showingResultView = true
                }
            )
        }
        #endif
        .sheet(isPresented: $showingResultView) {
            if let completionData = sessionCompletionData {
                SessionResultView(completionData: completionData)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: Notification.Name("ReturnToHome"))) { _ in
            // ホームに戻る
            dismiss()
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

            VStack(alignment: .leading, spacing: 12) {
                infoRow(
                    icon: "target",
                    title: "目標",
                    value: exercise.targetDisplayText,
                    color: .blue
                )

                infoRow(
                    icon: "lightbulb.fill",
                    title: "ガイダンス",
                    value: exercise.guidanceText,
                    color: .orange
                )
            }
        }
    }

    private func infoRow(icon: String, title: String, value: String, color: Color) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text(value)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }
    
    private var detailInfoSection: some View {
        VStack(spacing: 16) {
            Text("トレーニング効果")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            detailRow(icon: "figure.strengthtraining.traditional", title: "主な効果", description: exerciseEffects)
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
            .background(Color.systemGray6)
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
        .background(Color.systemBackground.opacity(0.95))
    }
    
    // MARK: - ヘルパービュー

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

        // 通知からセッションへの紐付け
        linkNotificationToSession()

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
        case .sideRaise:
            return "肩（三角筋側部）、僧帽筋、上腕"
        case .squat:
            return "大腿四頭筋、ハムストリング、臀筋、体幹"
        case .pushUp:
            return "胸筋、上腕三頭筋、肩、体幹"
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
        case .sideRaise:
            return [
                "肘を軽く曲げて固定する",
                "肩の高さまで上げる（それ以上は上げない）",
                "ゆっくりと制御して下ろす",
                "体の反動を使わない"
            ]
        case .squat:
            return [
                "膝がつま先より前に出ないよう注意",
                "背中をまっすぐ保つ",
                "太ももが床と平行になるまで下げる",
                "かかとに体重をかける"
            ]
        case .pushUp:
            return [
                "体を一直線に保つ",
                "胸がしっかりと床に近づくまで下げる",
                "手は肩幅より少し広めに",
                "腰を落とさない"
            ]
        }
    }

    // MARK: - Notification Linking

    /// 通知からセッションへの紐付け
    private func linkNotificationToSession() {
        // 最後にタップされた通知IDを取得
        guard let notificationId = UserDefaults.standard.string(forKey: "lastTappedNotificationId") else {
            print("📱 No notification ID found, session started without notification")
            return
        }

        // セッションIDを生成（開始時刻を使用）
        let sessionId = "\(Date().timeIntervalSince1970)"

        // 通知とセッションを紐付け
        let analyticsService = NotificationAnalyticsService()
        Task {
            do {
                try await analyticsService.linkNotificationToSession(
                    notificationId: notificationId,
                    sessionId: sessionId
                )
                print("✅ Linked notification \(notificationId) to session \(sessionId)")

                // 使用済み通知IDをクリア
                UserDefaults.standard.removeObject(forKey: "lastTappedNotificationId")
            } catch {
                print("❌ Failed to link notification to session: \(error)")
            }
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

#Preview("サイドレイズ") {
    NavigationView {
        ExerciseDetailView(exercise: .sideRaise)
    }
    .preferredColorScheme(.light)
}