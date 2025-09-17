import SwiftUI
import Charts
#if canImport(UIKit)
import UIKit
#endif

struct ProgressVisualizationView: View {
    @StateObject private var trainingSessionService = TrainingSessionService.shared
    @State private var timeRange: TimeRange = .week
    @State private var selectedExerciseType: String? = nil
    @State private var chartData: [ChartDataPoint] = []
    @State private var weeklyStats: WeeklyStats?
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // 時間範囲セレクタ
                    timeRangeSelector
                    
                    // 週間統計サマリー
                    if let stats = weeklyStats {
                        weeklyStatsCard(stats)
                    }
                    
                    // メインチャート
                    mainChart
                    
                    // フォーム正確率チャート
                    formAccuracyChart
                    
                    // 新記録バッジ
                    personalRecordsSection
                }
                .padding()
            }
            .navigationTitle("進捗データ")
            .onAppear {
                loadChartData()
                loadWeeklyStats()
            }
            .onChange(of: timeRange) {
                loadChartData()
            }
            #if os(iOS)
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
                // アプリがフォアグラウンドに戻った時にデータをリフレッシュ
                loadChartData()
                loadWeeklyStats()
            }
            #endif
        }
    }
    
    // MARK: - Time Range Selector
    private var timeRangeSelector: some View {
        Picker("期間", selection: $timeRange) {
            ForEach(TimeRange.allCases, id: \.self) { range in
                Text(range.displayName).tag(range)
            }
        }
        .pickerStyle(SegmentedPickerStyle())
    }
    
    // MARK: - Weekly Stats Card
    private func weeklyStatsCard(_ stats: WeeklyStats) -> some View {
        VStack(spacing: 12) {
            HStack {
                Text("今週の実績")
                    .font(.headline)
                Spacer()
            }
            
            HStack(spacing: 20) {
                StatItem(title: "セッション", value: "\(stats.totalSessions)")
                StatItem(title: "総レップ数", value: "\(stats.totalReps)")
                StatItem(title: "総時間", value: formatDuration(stats.totalDuration))
                StatItem(title: "フォーム精度", value: "\(Int(stats.averageFormAccuracy * 100))%")
            }
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }
    
    // MARK: - Main Chart
    private var mainChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("レップ数推移")
                    .font(.headline)
                Spacer()
            }
            
            if chartData.isEmpty {
                EmptyChartView()
            } else {
                Chart(chartData) { dataPoint in
                    LineMark(
                        x: .value("日付", dataPoint.date),
                        y: .value("レップ数", dataPoint.reps)
                    )
                    .foregroundStyle(.blue)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    
                    AreaMark(
                        x: .value("日付", dataPoint.date),
                        y: .value("レップ数", dataPoint.reps)
                    )
                    .foregroundStyle(.blue.opacity(0.2))
                }
                .frame(height: 200)
                .chartXAxis {
                    AxisMarks(values: .automatic) { _ in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel(format: .dateTime.month(.abbreviated).day())
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel()
                    }
                }
            }
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    // MARK: - Form Accuracy Chart
    private var formAccuracyChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("フォーム正確率")
                    .font(.headline)
                Spacer()
            }
            
            if chartData.isEmpty {
                EmptyChartView()
            } else {
                Chart(chartData) { dataPoint in
                    AreaMark(
                        x: .value("日付", dataPoint.date),
                        y: .value("正確率", dataPoint.formAccuracy * 100)
                    )
                    .foregroundStyle(.green.opacity(0.6))
                    
                    LineMark(
                        x: .value("日付", dataPoint.date),
                        y: .value("正確率", dataPoint.formAccuracy * 100)
                    )
                    .foregroundStyle(.green)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                }
                .frame(height: 150)
                .chartYScale(domain: 0...100)
                .chartXAxis {
                    AxisMarks(values: .automatic) { _ in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel(format: .dateTime.month(.abbreviated).day())
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel()
                    }
                }
            }
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    // MARK: - Personal Records Section
    private var personalRecordsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("個人記録")
                    .font(.headline)
                Spacer()
                Button("すべて見る") {
                    // Navigate to full records view
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                PersonalRecordCard(
                    title: "最高レップ数",
                    value: getPersonalBest(.maxReps),
                    icon: "trophy.fill",
                    color: .orange
                )
                
                PersonalRecordCard(
                    title: "完璧フォーム",
                    value: getPersonalBest(.perfectForm),
                    icon: "checkmark.seal.fill",
                    color: .green
                )
            }
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    // MARK: - Data Loading
    private func loadChartData() {
        let calendar = Calendar.current
        let endDate = Date()
        let startDate: Date
        
        switch timeRange {
        case .week:
            startDate = calendar.date(byAdding: .day, value: -7, to: endDate) ?? endDate
        case .month:
            startDate = calendar.date(byAdding: .day, value: -30, to: endDate) ?? endDate
        case .threeMonths:
            startDate = calendar.date(byAdding: .day, value: -90, to: endDate) ?? endDate
        }
        
        let sessions = trainingSessionService.fetchSessionsForDateRange(from: startDate, to: endDate)
            .filter { $0.totalReps > 0 } // 空のセッションを除外
        
        print("📊 Chart data loading: Found \(sessions.count) valid sessions for date range")
        for session in sessions.prefix(3) {
            print("📊 Session: \(session.totalReps) reps, \(session.formErrors) errors, started: \(session.startTime ?? Date())")
        }
        
        // 個別セッションをチャートポイントとして使用（日付ごとの集計をやめる）
        chartData = sessions.map { session in
            let reps = Int(session.totalReps)
            let errors = min(Int(session.formErrors), reps) // エラーがレップ数を超えないように制限
            let formAccuracy = reps > 0 ? max(0.0, Double(reps - errors) / Double(reps)) : 0.0
            
            let dataPoint = ChartDataPoint(
                date: session.startTime ?? Date(),
                reps: reps,
                formAccuracy: formAccuracy,
                sessions: 1
            )
            
            print("📊 Chart point: \(session.startTime ?? Date()) - \(reps) reps, \(String(format: "%.1f", formAccuracy * 100))% accuracy")
            return dataPoint
        }.sorted { $0.date < $1.date }
        
        print("📊 Final chart data: \(chartData.count) points")
    }
    
    private func loadWeeklyStats() {
        weeklyStats = CoreDataManager.shared.calculateWeeklyStats()
        
        if let stats = weeklyStats {
            print("📊 Weekly stats loaded: \(stats.totalSessions) sessions, \(stats.totalReps) reps, \(String(format: "%.1f", stats.averageFormAccuracy * 100))% accuracy")
        } else {
            print("📊 No weekly stats available")
        }
    }
    
    private func getPersonalBest(_ recordType: RecordType) -> String {
        let personalBests = trainingSessionService.calculatePersonalBests()
        let record = personalBests.first { $0.recordType == recordType }
        return record?.value.description ?? "0"
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let hours = Int(duration) / 3600
        let minutes = (Int(duration) % 3600) / 60
        
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else {
            return "\(minutes)m"
        }
    }
}

// MARK: - Supporting Views
struct StatItem: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct PersonalRecordCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(value)
                    .font(.title3)
                    .fontWeight(.bold)
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
        .padding(12)
        .background(color.opacity(0.1))
        .cornerRadius(8)
    }
}

struct EmptyChartView: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.largeTitle)
                .foregroundColor(.gray)
            
            Text("データがありません")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("トレーニングを開始してデータを蓄積しましょう！")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(height: 150)
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Data Models
struct ChartDataPoint: Identifiable {
    let id = UUID()
    let date: Date
    let reps: Int
    let formAccuracy: Double
    let sessions: Int
}

enum TimeRange: CaseIterable {
    case week, month, threeMonths
    
    var displayName: String {
        switch self {
        case .week: return "7日間"
        case .month: return "30日間"
        case .threeMonths: return "90日間"
        }
    }
}

#Preview {
    ProgressVisualizationView()
}