// DEBUG: Calendar data inspector - remove in production

import SwiftUI

/// カレンダーデータ表示
/// 要件対応: 4.1, 4.2, 4.3, 4.5, 4.6, 4.7
struct CalendarDataView: View {
    @StateObject private var dataProvider = CalendarDebugDataProvider()
    @State private var showError = false

    var body: some View {
        List {
            // サマリーセクション
            Section {
                HStack {
                    Text("カレンダー連携")
                    Spacer()
                    if dataProvider.isCalendarConnected {
                        Label("接続済み", systemImage: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    } else {
                        Label("未接続", systemImage: "xmark.circle.fill")
                            .foregroundColor(.red)
                    }
                }

                HStack {
                    Text("イベント数")
                    Spacer()
                    Text("\(dataProvider.eventCount)件")
                        .foregroundColor(.secondary)
                }

                if let lastFetch = dataProvider.lastFetchDate {
                    HStack {
                        Text("最終取得")
                        Spacer()
                        Text(formatDateTime(lastFetch))
                            .foregroundColor(.secondary)
                    }
                }

                HStack {
                    Text("取得期間")
                    Spacer()
                    Text("今日から7日間")
                        .foregroundColor(.secondary)
                }
            } header: {
                Text("サマリー")
            }

            // 同期ボタンセクション
            if dataProvider.isCalendarConnected {
                Section {
                    Button(action: {
                        Task {
                            do {
                                try await dataProvider.manualSync()
                            } catch {
                                showError = true
                            }
                        }
                    }) {
                        HStack {
                            if dataProvider.isLoading {
                                ProgressView()
                            } else {
                                Image(systemName: "arrow.clockwise")
                            }
                            Text("最新データを取得")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .disabled(dataProvider.isLoading)
                }
            }

            // エラー表示
            if let error = dataProvider.fetchError {
                Section {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            // イベント一覧
            if !dataProvider.events.isEmpty {
                let groupedEvents = Dictionary(grouping: dataProvider.events) { event in
                    Calendar.current.startOfDay(for: event.startTime)
                }

                ForEach(groupedEvents.keys.sorted(), id: \.self) { date in
                    Section {
                        ForEach(groupedEvents[date] ?? [], id: \.id) { event in
                            NavigationLink(destination: EventDetailView(event: event)) {
                                EventRow(event: event)
                            }
                        }
                    } header: {
                        Text(formatDate(date))
                    }
                }
            } else if !dataProvider.isCalendarConnected {
                Section {
                    VStack(spacing: 12) {
                        Image(systemName: "calendar.badge.exclamationmark")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("Googleカレンダーと連携していません")
                            .font(.headline)
                        Text("設定 → カレンダー連携から接続してください")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
            }
        }
        .navigationTitle("カレンダーデータ")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            dataProvider.loadRecentData()
        }
        .alert("エラー", isPresented: $showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(dataProvider.fetchError ?? "不明なエラーが発生しました")
        }
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "M月d日(E)"
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }

    private func formatDateTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }
}

struct EventRow: View {
    let event: CalendarEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(formatTime(event.startTime))
                    .font(.headline)
                Text("-")
                    .foregroundColor(.secondary)
                Text(formatTime(event.endTime))
                    .font(.headline)

                Spacer()

                StatusBadge(status: event.status)
            }

            Text(durationText)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }

    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }

    private var durationText: String {
        let duration = event.endTime.timeIntervalSince(event.startTime)
        let minutes = Int(duration / 60)
        return "\(minutes)分"
    }
}

struct StatusBadge: View {
    let status: CalendarEvent.EventStatus

    var body: some View {
        Text(statusText)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(statusColor.opacity(0.2))
            .foregroundColor(statusColor)
            .cornerRadius(4)
    }

    private var statusText: String {
        switch status {
        case .confirmed:
            return "確定"
        case .tentative:
            return "仮"
        }
    }

    private var statusColor: Color {
        switch status {
        case .confirmed:
            return .green
        case .tentative:
            return .orange
        }
    }
}

#Preview {
    NavigationStack {
        CalendarDataView()
    }
}
