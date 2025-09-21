import SwiftUI
import Charts

struct WeeklyChartView: View {
    @StateObject private var trainingSessionService = TrainingSessionService.shared
    @State private var chartData: [WeeklyChartData] = []
    @State private var exerciseFilter: String? = nil
    @State private var showingNewRecordBadge = false
    @State private var latestRecord: PersonalRecord?
    
    var body: some View {
        VStack(spacing: 16) {
            // Header with exercise filter
            headerSection
            
            // Weekly trend comparison
            weeklyTrendSection
            
            // Exercise type breakdown
            exerciseBreakdownChart
            
            // New record badge (animated)
            if showingNewRecordBadge, let record = latestRecord {
                NewRecordBadge(record: record)
                    .transition(.scale.combined(with: .opacity))
                    .onTapGesture {
                        withAnimation {
                            showingNewRecordBadge = false
                        }
                    }
            }
        }
        .onAppear {
            loadWeeklyData()
            checkForNewRecords()
        }
    }
    
    // MARK: - Header Section
    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("週間分析")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("過去4週間のトレーニング傾向")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Menu {
                Button("すべての種目") {
                    exerciseFilter = nil
                    loadWeeklyData()
                }
                
                ForEach(availableExerciseTypes, id: \.self) { exerciseType in
                    Button(exerciseType) {
                        exerciseFilter = exerciseType
                        loadWeeklyData()
                    }
                }
            } label: {
                HStack {
                    Text(exerciseFilter ?? "すべて")
                        .font(.caption)
                    Image(systemName: "chevron.down")
                        .font(.caption2)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.systemGray5)
                .cornerRadius(8)
            }
        }
    }
    
    // MARK: - Weekly Trend Section
    private var weeklyTrendSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("週間レップ数推移")
                    .font(.headline)
                Spacer()
                Text(trendDescription)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(trendColor.opacity(0.2))
                    .foregroundColor(trendColor)
                    .cornerRadius(6)
            }
            
            if chartData.isEmpty {
                EmptyWeeklyChartView()
            } else {
                Chart(chartData) { data in
                    LineMark(
                        x: .value("週", data.weekLabel),
                        y: .value("レップ数", data.totalReps)
                    )
                    .foregroundStyle(.blue)
                    .lineStyle(StrokeStyle(lineWidth: 3))
                    .symbol(.circle)
                    .symbolSize(60)
                    
                    AreaMark(
                        x: .value("週", data.weekLabel),
                        y: .value("レップ数", data.totalReps)
                    )
                    .foregroundStyle(.blue.opacity(0.1))
                }
                .frame(height: 180)
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel()
                    }
                }
                .chartXAxis {
                    AxisMarks { value in
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
    
    // MARK: - Exercise Breakdown Chart
    private var exerciseBreakdownChart: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("種目別比較")
                .font(.headline)
            
            if chartData.isEmpty {
                EmptyWeeklyChartView()
            } else {
                Chart(exerciseBreakdownData) { data in
                    BarMark(
                        x: .value("種目", data.exerciseType),
                        y: .value("レップ数", data.reps)
                    )
                    .foregroundStyle(by: .value("種目", data.exerciseType))
                    .cornerRadius(6)
                }
                .frame(height: 200)
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel()
                    }
                }
                .chartXAxis {
                    AxisMarks { value in
                        AxisTick()
                        AxisValueLabel()
                    }
                }
                .chartLegend(.hidden)
            }
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    // MARK: - Data Loading
    private func loadWeeklyData() {
        let calendar = Calendar.current
        let now = Date()
        var weeklyData: [WeeklyChartData] = []
        
        for weekOffset in (0..<4).reversed() {
            let weekStart = calendar.date(byAdding: .weekOfYear, value: -weekOffset, to: now) ?? now
            let weekEnd = calendar.date(byAdding: .day, value: 6, to: weekStart) ?? weekStart
            
            var sessions: [TrainingSession]
            if let filter = exerciseFilter {
                sessions = trainingSessionService.fetchSessionsForDateRange(from: weekStart, to: weekEnd)
                    .filter { $0.exerciseType == filter }
            } else {
                sessions = trainingSessionService.fetchSessionsForDateRange(from: weekStart, to: weekEnd)
            }
            
            let totalReps = sessions.reduce(0) { $0 + Int($1.totalReps) }
            let totalErrors = sessions.reduce(0) { $0 + Int($1.formErrors) }
            let formAccuracy = totalReps > 0 ? Double(totalReps - totalErrors) / Double(totalReps) : 0.0
            
            let weekFormatter = DateFormatter()
            weekFormatter.dateFormat = "M/d"
            let weekLabel = weekFormatter.string(from: weekStart)
            
            weeklyData.append(WeeklyChartData(
                weekStart: weekStart,
                weekLabel: weekLabel,
                totalReps: totalReps,
                formAccuracy: formAccuracy,
                sessionCount: sessions.count
            ))
        }
        
        chartData = weeklyData
    }
    
    private func checkForNewRecords() {
        let personalBests = trainingSessionService.calculatePersonalBests()
        if let newRecord = personalBests.first(where: { $0.isNewRecord }) {
            latestRecord = newRecord
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) {
                    showingNewRecordBadge = true
                }
                
                // Auto-hide after 3 seconds
                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                    withAnimation {
                        showingNewRecordBadge = false
                    }
                }
            }
        }
    }
    
    // MARK: - Computed Properties
    private var availableExerciseTypes: [String] {
        let allSessions = trainingSessionService.fetchSessionsForDateRange(
            from: Calendar.current.date(byAdding: .month, value: -1, to: Date()) ?? Date(),
            to: Date()
        )
        return Array(Set(allSessions.compactMap { $0.exerciseType }))
    }
    
    private var exerciseBreakdownData: [ExerciseBreakdownData] {
        let calendar = Calendar.current
        let weekAgo = calendar.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        let sessions = trainingSessionService.fetchSessionsForDateRange(from: weekAgo, to: Date())
        
        let grouped = Dictionary(grouping: sessions, by: { $0.exerciseType ?? "Unknown" })
        
        return grouped.map { (exerciseType, sessions) in
            let totalReps = sessions.reduce(0) { $0 + Int($1.totalReps) }
            return ExerciseBreakdownData(exerciseType: exerciseType, reps: totalReps)
        }.sorted { $0.reps > $1.reps }
    }
    
    private var trendDescription: String {
        guard chartData.count >= 2 else { return "データ不足" }
        
        let thisWeek = chartData.last?.totalReps ?? 0
        let lastWeek = chartData[chartData.count - 2].totalReps
        
        if thisWeek > lastWeek {
            return "📈 向上中"
        } else if thisWeek < lastWeek {
            return "📉 要注意"
        } else {
            return "➡️ 維持"
        }
    }
    
    private var trendColor: Color {
        guard chartData.count >= 2 else { return .gray }
        
        let thisWeek = chartData.last?.totalReps ?? 0
        let lastWeek = chartData[chartData.count - 2].totalReps
        
        if thisWeek > lastWeek {
            return .green
        } else if thisWeek < lastWeek {
            return .red
        } else {
            return .blue
        }
    }
}

// MARK: - New Record Badge
struct NewRecordBadge: View {
    let record: PersonalRecord
    @State private var animate = false
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "trophy.fill")
                .font(.largeTitle)
                .foregroundColor(.orange)
                .scaleEffect(animate ? 1.2 : 1.0)
                .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: animate)
            
            Text("新記録達成！")
                .font(.headline)
                .fontWeight(.bold)
            
            Text("\(record.recordType.displayName): \(record.value)")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            if let improvement = record.improvement {
                Text("前回より +\(improvement)")
                    .font(.caption)
                    .foregroundColor(.green)
                    .fontWeight(.semibold)
            }
        }
        .padding(20)
        .background(Color.systemBackground)
        .cornerRadius(16)
        .shadow(radius: 8)
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(.orange, lineWidth: 2)
        )
        .onAppear {
            animate = true
        }
    }
}

// MARK: - Empty Chart View
struct EmptyWeeklyChartView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "chart.bar")
                .font(.system(size: 40))
                .foregroundColor(.gray)
            
            Text("まだデータがありません")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("週単位でのデータを表示するには\nもう少しトレーニングを継続してください")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(height: 150)
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Data Models
struct WeeklyChartData: Identifiable {
    let id = UUID()
    let weekStart: Date
    let weekLabel: String
    let totalReps: Int
    let formAccuracy: Double
    let sessionCount: Int
}

struct ExerciseBreakdownData: Identifiable {
    let id = UUID()
    let exerciseType: String
    let reps: Int
}

#Preview {
    WeeklyChartView()
}