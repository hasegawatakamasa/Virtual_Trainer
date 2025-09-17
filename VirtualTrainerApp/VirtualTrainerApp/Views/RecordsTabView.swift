import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct RecordsTabView: View {
    @StateObject private var trainingSessionService = TrainingSessionService.shared
    @StateObject private var achievementSystem = AchievementSystem.shared
    @StateObject private var oshiReactionManager = OshiReactionManager.shared
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // 進捗・チャート表示タブ
            ProgressVisualizationView()
                .tabItem {
                    Label("進捗", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(0)
            
            // 週間分析タブ
            WeeklyChartView()
                .tabItem {
                    Label("分析", systemImage: "chart.bar")
                }
                .tag(1)
            
            // カレンダータブ
            CalendarView()
                .tabItem {
                    Label("カレンダー", systemImage: "calendar")
                }
                .tag(2)
            
            // アチーブメント・推しレベルタブ
            AchievementListView()
                .tabItem {
                    Label("実績", systemImage: "trophy")
                }
                .tag(3)
            
            // 推しレベル・反応履歴タブ
            OshiLevelView()
                .tabItem {
                    Label("推しレベル", systemImage: "heart.fill")
                }
                .tag(4)
        }
        .onAppear {
            setupInitialData()
        }
        #if os(iOS)
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
            // アプリがフォアグラウンドに戻った時にデータをリフレッシュ
            setupInitialData()
        }
        #endif
    }
    
    private func setupInitialData() {
        // 初期データロードやリフレッシュ処理
        trainingSessionService.loadRecentSessions()
    }
}

// MARK: - Calendar View
struct CalendarView: View {
    @StateObject private var trainingSessionService = TrainingSessionService.shared
    @State private var selectedDate = Date()
    @State private var selectedDaySessions: [TrainingSession] = []
    @State private var monthlyStats: MonthlyStats?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                // 月間統計サマリー
                if let stats = monthlyStats {
                    monthlyStatsCard(stats)
                }
                
                // カレンダー表示（iOS 16+）
                if #available(iOS 16.0, *) {
                    CalendarViewRepresentable(
                        selectedDate: $selectedDate,
                        sessions: trainingSessionService.recentSessions
                    )
                } else {
                    // iOS 15以下の場合はリスト表示
                    Text("カレンダー機能はiOS 16以上で利用可能です")
                        .foregroundColor(.secondary)
                }
                
                // 選択日の詳細
                if !selectedDaySessions.isEmpty {
                    selectedDayDetails
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("トレーニングカレンダー")
            .onAppear {
                loadMonthlyStats()
                loadSelectedDaySessions()
            }
            .onChange(of: selectedDate) {
                loadSelectedDaySessions()
            }
        }
    }
    
    private func monthlyStatsCard(_ stats: MonthlyStats) -> some View {
        VStack(spacing: 12) {
            HStack {
                Text("今月の実績")
                    .font(.headline)
                Spacer()
                Text(stats.monthYear)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            HStack(spacing: 20) {
                StatItem(title: "トレーニング日数", value: "\(stats.trainingDays)")
                StatItem(title: "総セッション", value: "\(stats.totalSessions)")
                StatItem(title: "総レップ数", value: "\(stats.totalReps)")
                StatItem(title: "連続記録", value: "\(stats.currentStreak)日")
            }
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }
    
    private var selectedDayDetails: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("選択日の詳細")
                    .font(.headline)
                Spacer()
                Text(DateFormatter.dayDetail.string(from: selectedDate))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            ForEach(selectedDaySessions, id: \.objectID) { session in
                DayDetailCard(session: session)
            }
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private func loadMonthlyStats() {
        let calendar = Calendar.current
        let now = Date()
        let startOfMonth = calendar.dateInterval(of: .month, for: now)?.start ?? now
        let sessions = trainingSessionService.fetchSessionsForDateRange(from: startOfMonth, to: now)
        
        let trainingDays = Set(sessions.compactMap { session -> Date? in
            guard let startTime = session.startTime else { return nil }
            return calendar.startOfDay(for: startTime)
        }).count
        
        let totalSessions = sessions.count
        let totalReps = sessions.reduce(0) { $0 + Int($1.totalReps) }
        let currentStreak = calculateCurrentStreak()
        
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy年M月"
        
        monthlyStats = MonthlyStats(
            monthYear: formatter.string(from: now),
            trainingDays: trainingDays,
            totalSessions: totalSessions,
            totalReps: totalReps,
            currentStreak: currentStreak
        )
    }
    
    private func loadSelectedDaySessions() {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: selectedDate)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay) ?? startOfDay
        
        selectedDaySessions = trainingSessionService.fetchSessionsForDateRange(from: startOfDay, to: endOfDay)
    }
    
    private func calculateCurrentStreak() -> Int {
        // 簡易版のストリーク計算
        let calendar = Calendar.current
        let sessions = trainingSessionService.fetchSessionsForDateRange(
            from: calendar.date(byAdding: .month, value: -1, to: Date()) ?? Date(),
            to: Date()
        )
        
        let sessionDates = Set(sessions.compactMap { session -> Date? in
            guard let startTime = session.startTime else { return nil }
            return calendar.startOfDay(for: startTime)
        })
        
        var streak = 0
        var checkDate = calendar.startOfDay(for: Date())
        
        while sessionDates.contains(checkDate) {
            streak += 1
            checkDate = calendar.date(byAdding: .day, value: -1, to: checkDate) ?? checkDate
        }
        
        return streak
    }
}

// MARK: - Achievement List View
struct AchievementListView: View {
    @StateObject private var achievementSystem = AchievementSystem.shared
    @State private var achievementProgress: [AchievementProgress] = []
    
    var body: some View {
        NavigationView {
            List {
                // 最近の獲得アチーブメント
                if !achievementSystem.recentUnlocks.isEmpty {
                    Section("最近の獲得") {
                        ForEach(achievementSystem.recentUnlocks.prefix(3), id: \.objectID) { achievement in
                            RecentAchievementRow(achievement: achievement)
                        }
                    }
                }
                
                // 進捗状況
                Section("アチーブメント進捗") {
                    ForEach(achievementProgress) { progress in
                        AchievementProgressRow(progress: progress)
                    }
                }
                
                // すべての獲得済みアチーブメント
                Section("獲得済み") {
                    ForEach(achievementSystem.unlockedAchievements, id: \.objectID) { achievement in
                        UnlockedAchievementRow(achievement: achievement)
                    }
                }
            }
            .navigationTitle("アチーブメント")
            .onAppear {
                loadAchievementProgress()
            }
        }
    }
    
    private func loadAchievementProgress() {
        achievementProgress = achievementSystem.getAchievementProgress()
    }
}

// MARK: - Oshi Level View
struct OshiLevelView: View {
    @StateObject private var achievementSystem = AchievementSystem.shared
    @StateObject private var oshiReactionManager = OshiReactionManager.shared
    @State private var selectedCharacterLevel: OshiBondLevel?
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // キャラクター選択とレベル表示
                    characterLevelSection
                    
                    // レベルアップ進捗
                    if let level = selectedCharacterLevel {
                        levelProgressSection(level)
                    }
                    
                    // 最近の推し反応
                    recentReactionsSection
                    
                    // 絆ポイント履歴
                    bondPointHistorySection
                }
                .padding()
            }
            .navigationTitle("推しレベル")
            .onAppear {
                loadCharacterLevel()
            }
        }
    }
    
    private var characterLevelSection: some View {
        VStack(spacing: 12) {
            Text("推し絆レベル")
                .font(.title2)
                .fontWeight(.bold)
            
            if let level = selectedCharacterLevel {
                VStack(spacing: 8) {
                    Text(level.characterName)
                        .font(.title3)
                        .fontWeight(.semibold)
                    
                    HStack {
                        Text("Lv.\(level.currentLevel)")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.primary)
                        
                        Spacer()
                        
                        VStack(alignment: .trailing) {
                            Text("\(level.bondPoints) BP")
                                .font(.headline)
                            Text("総獲得: \(level.totalBondPoints) BP")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }
    
    private func levelProgressSection(_ level: OshiBondLevel) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("次のレベルまで")
                    .font(.headline)
                Spacer()
                Text("\(level.nextLevelPoints - level.bondPoints) BP")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            ProgressView(value: level.progressToNextLevel)
                .progressViewStyle(LinearProgressViewStyle(tint: .blue))
            
            HStack {
                Text("Lv.\(level.currentLevel)")
                    .font(.caption)
                Spacer()
                Text("Lv.\(level.currentLevel + 1)")
                    .font(.caption)
            }
            .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var recentReactionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("最近の推し反応")
                .font(.headline)
            
            if oshiReactionManager.recentReactions.isEmpty {
                Text("まだ反応がありません")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(oshiReactionManager.recentReactions.prefix(5)) { reaction in
                    ReactionCard(reaction: reaction)
                }
            }
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var bondPointHistorySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("絆ポイント獲得履歴")
                .font(.headline)
            
            // 簡易版の履歴表示
            Text("実装予定: 詳細な獲得履歴")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding()
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private func loadCharacterLevel() {
        let characterName = UserDefaults.standard.string(forKey: "selectedCharacter") ?? "ずんだもん"
        
        if let existingLevel = achievementSystem.oshiBondLevels.first(where: { $0.characterName == characterName }) {
            selectedCharacterLevel = existingLevel
        } else {
            selectedCharacterLevel = OshiBondLevel(characterName: characterName)
        }
    }
}

// MARK: - Supporting Views
struct DayDetailCard: View {
    let session: TrainingSession
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(session.exerciseType ?? "Unknown")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text("\(session.totalReps)レップ・\(String(format: "%.1f", session.sessionDuration/60))分")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(session.characterName ?? "")
                    .font(.caption)
                    .foregroundColor(.blue)
                
                if session.formErrors > 0 {
                    Text("エラー: \(session.formErrors)")
                        .font(.caption2)
                        .foregroundColor(.red)
                }
            }
        }
        .padding(.vertical, 8)
    }
}

struct RecentAchievementRow: View {
    let achievement: Achievement
    
    var body: some View {
        HStack {
            Image(systemName: "trophy.fill")
                .foregroundColor(.orange)
            
            VStack(alignment: .leading) {
                Text(achievement.type ?? "Achievement")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text("\(achievement.bondPointsAwarded) BP獲得")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(RelativeDateTimeFormatter().localizedString(for: achievement.unlockedAt ?? Date(), relativeTo: Date()))
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

struct AchievementProgressRow: View {
    let progress: AchievementProgress
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: progress.iconName)
                    .foregroundColor(progress.isUnlocked ? .green : .gray)
                
                VStack(alignment: .leading) {
                    Text(progress.title)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    Text(progress.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                if progress.isUnlocked {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                }
            }
            
            if !progress.isUnlocked {
                ProgressView(value: progress.progress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .blue))
            }
        }
    }
}

struct UnlockedAchievementRow: View {
    let achievement: Achievement
    
    var body: some View {
        HStack {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
            
            VStack(alignment: .leading) {
                Text(achievement.type ?? "Achievement")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(DateFormatter.dayDetail.string(from: achievement.unlockedAt ?? Date()))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text("+\(achievement.bondPointsAwarded) BP")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.blue)
        }
    }
}

struct ReactionCard: View {
    let reaction: OshiReaction
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: reaction.type.iconName)
                    .foregroundColor(.pink)
                
                Text(reaction.type.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text(RelativeDateTimeFormatter().localizedString(for: reaction.triggeredAt, relativeTo: Date()))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Text(reaction.message)
                .font(.subheadline)
                .multilineTextAlignment(.leading)
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(8)
    }
}

// MARK: - Calendar Representative (iOS 16+)
@available(iOS 16.0, *)
struct CalendarViewRepresentable: View {
    @Binding var selectedDate: Date
    let sessions: [TrainingSession]
    
    var body: some View {
        // iOS 16のCalendarView実装は複雑なため、プレースホルダー
        VStack {
            Text("カレンダーUI")
                .font(.headline)
            
            Text("ここにiOS 16のCalendarViewが表示されます")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(height: 300)
        .frame(maxWidth: .infinity)
        .background(Color.systemGray6)
        .cornerRadius(12)
    }
}

// MARK: - Data Models
struct MonthlyStats {
    let monthYear: String
    let trainingDays: Int
    let totalSessions: Int
    let totalReps: Int
    let currentStreak: Int
}

// MARK: - Date Formatters
extension DateFormatter {
    static let dayDetail: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        return formatter
    }()
}

#Preview {
    RecordsTabView()
}