import SwiftUI

/// 通知効果統計画面
struct NotificationStatsView: View {
    private let analyticsService = NotificationAnalyticsService()
    @State private var weeklyEffectiveness: NotificationEffectiveness?
    @State private var weeklyAnalytics: NotificationAnalytics?
    @State private var optimalTimeSlots: [NotificationAnalytics.TimeSlot: Double] = [:]
    @State private var frequencyAdjustment: FrequencyAdjustment?
    @State private var isLoading = false
    @State private var errorMessage: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                if isLoading {
                    ProgressView("データを読み込み中...")
                        .padding()
                } else if let error = errorMessage {
                    ErrorView(message: error)
                } else {
                    // 今週のサマリー
                    if let effectiveness = weeklyEffectiveness {
                        WeeklySummarySection(effectiveness: effectiveness)
                    }

                    // 推奨設定アドバイス
                    if let adjustment = frequencyAdjustment {
                        FrequencyAdjustmentCard(adjustment: adjustment)
                    }

                    // 時間帯別効果
                    if !optimalTimeSlots.isEmpty {
                        TimeSlotEffectivenessSection(timeSlots: optimalTimeSlots)
                    }

                    // 詳細分析
                    if let analytics = weeklyAnalytics {
                        DetailedAnalyticsSection(analytics: analytics)
                    }
                }
            }
            .padding()
        }
        .navigationTitle("通知効果")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await loadAnalytics()
        }
        .refreshable {
            await loadAnalytics()
        }
    }

    // MARK: - Data Loading

    private func loadAnalytics() async {
        isLoading = true
        errorMessage = nil

        // 週次効果を取得
        do {
            weeklyEffectiveness = try await analyticsService.getWeeklyEffectiveness()
        } catch {
            errorMessage = "データの読み込みに失敗しました: \(error.localizedDescription)"
        }

        // 週次分析を取得
        do {
            weeklyAnalytics = try await analyticsService.performWeeklyAnalysis()
        } catch {
            errorMessage = "データの読み込みに失敗しました: \(error.localizedDescription)"
        }

        // 最適時間帯を取得
        do {
            optimalTimeSlots = try await analyticsService.calculateOptimalTimeSlots()
        } catch {
            errorMessage = "データの読み込みに失敗しました: \(error.localizedDescription)"
        }

        // 頻度調整提案を取得
        if let analytics = weeklyAnalytics {
            frequencyAdjustment = analyticsService.suggestFrequencyAdjustment(analytics: analytics)
        }

        isLoading = false
    }
}

// MARK: - Weekly Summary Section

struct WeeklySummarySection: View {
    let effectiveness: NotificationEffectiveness

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("今週の通知効果")
                .font(.headline)

            VStack(spacing: 12) {
                SummaryRow(
                    icon: "bell.fill",
                    label: "配信数",
                    value: "\(effectiveness.deliveredCount)",
                    color: .blue
                )

                SummaryRow(
                    icon: "hand.tap.fill",
                    label: "タップ数",
                    value: "\(effectiveness.tappedCount)",
                    color: .green
                )

                SummaryRow(
                    icon: "figure.walk",
                    label: "トレーニング実施",
                    value: "\(effectiveness.sessionCount)",
                    color: .orange
                )

                Divider()

                HStack {
                    Label("実施率", systemImage: "chart.line.uptrend.xyaxis")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    Spacer()

                    Text(effectiveness.effectivenessPercentage)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(effectiveness.effectivenessColor)
                }
            }
            .padding()
            .background(Color(UIColor.secondarySystemBackground))
            .cornerRadius(12)
        }
    }
}

// MARK: - Summary Row

struct SummaryRow: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        HStack {
            Label(label, systemImage: icon)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundColor(color)
        }
    }
}

// MARK: - Frequency Adjustment Card

struct FrequencyAdjustmentCard: View {
    let adjustment: FrequencyAdjustment

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(.yellow)
                Text("おすすめ設定")
                    .font(.headline)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("推奨頻度: \(adjustment.suggestedFrequency.displayName)")
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Text(adjustment.reason)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.yellow.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Time Slot Effectiveness Section

struct TimeSlotEffectivenessSection: View {
    let timeSlots: [NotificationAnalytics.TimeSlot: Double]

    private var sortedSlots: [(NotificationAnalytics.TimeSlot, Double)] {
        timeSlots.sorted { (slot1, slot2) in
            slot1.key.hour < slot2.key.hour
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("時間帯別効果")
                .font(.headline)

            VStack(spacing: 8) {
                ForEach(sortedSlots, id: \.0) { slot, effectiveness in
                    TimeSlotRow(slot: slot, effectiveness: effectiveness)
                }
            }
            .padding()
            .background(Color(UIColor.secondarySystemBackground))
            .cornerRadius(12)
        }
    }
}

// MARK: - Time Slot Row

struct TimeSlotRow: View {
    let slot: NotificationAnalytics.TimeSlot
    let effectiveness: Double

    private var effectivenessPercentage: String {
        String(format: "%.0f%%", effectiveness * 100)
    }

    private var barWidth: CGFloat {
        CGFloat(effectiveness)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(slot.displayName)
                    .font(.subheadline)

                Spacer()

                Text(effectivenessPercentage)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(effectivenessColor)
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 8)
                        .cornerRadius(4)

                    Rectangle()
                        .fill(effectivenessColor)
                        .frame(width: geometry.size.width * barWidth, height: 8)
                        .cornerRadius(4)
                }
            }
            .frame(height: 8)
        }
    }

    private var effectivenessColor: Color {
        if effectiveness >= 0.5 {
            return .green
        } else if effectiveness >= 0.3 {
            return .orange
        } else {
            return .red
        }
    }
}

// MARK: - Detailed Analytics Section

struct DetailedAnalyticsSection: View {
    let analytics: NotificationAnalytics

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("詳細分析")
                .font(.headline)

            VStack(spacing: 12) {
                DetailRow(
                    label: "分析期間",
                    value: formatDateRange(analytics.period)
                )

                DetailRow(
                    label: "タップ率",
                    value: String(format: "%.1f%%", analytics.tapRate * 100)
                )

                DetailRow(
                    label: "実施率",
                    value: String(format: "%.1f%%", analytics.conversionRate * 100)
                )
            }
            .padding()
            .background(Color(UIColor.secondarySystemBackground))
            .cornerRadius(12)
        }
    }

    private func formatDateRange(_ interval: DateInterval) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        return "\(formatter.string(from: interval.start)) - \(formatter.string(from: interval.end))"
    }
}

// MARK: - Detail Row

struct DetailRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
        }
    }
}

// MARK: - Error View

struct ErrorView: View {
    let message: String

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundColor(.red)

            Text(message)
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
        }
        .padding()
    }
}

// MARK: - Extensions

extension NotificationEffectiveness {
    var effectivenessPercentage: String {
        String(format: "%.0f%%", effectiveness * 100)
    }

    var effectivenessColor: Color {
        if effectiveness >= 0.5 {
            return .green
        } else if effectiveness >= 0.3 {
            return .orange
        } else {
            return .red
        }
    }
}

extension NotificationAnalytics.TimeSlot {
    var displayName: String {
        let endHour = (hour + 1) % 24
        return "\(hour):00 - \(endHour):00"
    }
}

#Preview {
    NavigationStack {
        NotificationStatsView()
    }
}