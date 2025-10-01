// DEBUG: Event detail inspector - remove in production

import SwiftUI

/// イベント詳細表示
/// 要件対応: 4.3
struct EventDetailView: View {
    let event: CalendarEvent

    var body: some View {
        List {
            Section("時刻情報") {
                DebugDetailRow(label: "開始時刻", value: formatDateTime(event.startTime))
                DebugDetailRow(label: "終了時刻", value: formatDateTime(event.endTime))
                DebugDetailRow(label: "長さ", value: durationText)
            }

            Section("ステータス") {
                HStack {
                    Text("確定状況")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    StatusBadge(status: event.status)
                }
            }

            Section("イベントID") {
                Text(event.id)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .navigationTitle("イベント詳細")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func formatDateTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "M月d日(E) HH:mm"
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }

    private var durationText: String {
        let duration = event.endTime.timeIntervalSince(event.startTime)
        let minutes = Int(duration / 60)
        let hours = minutes / 60
        let remainingMinutes = minutes % 60

        if hours > 0 {
            return "\(hours)時間\(remainingMinutes)分"
        } else {
            return "\(minutes)分"
        }
    }
}

#Preview {
    let event = CalendarEvent(
        id: "test-123",
        startTime: Date(),
        endTime: Date().addingTimeInterval(3600),
        status: .confirmed
    )

    return NavigationStack {
        EventDetailView(event: event)
    }
}
