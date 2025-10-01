// DEBUG: Gap time detail inspector - remove in production

import SwiftUI

/// 隙間時間詳細表示
/// 要件対応: 5.6
struct GapTimeDetailView: View {
    let gapTime: AvailableTimeSlot
    let dataProvider: GapTimeDebugProvider

    @State private var surroundingEvents: (before: CalendarEvent?, after: CalendarEvent?) = (nil, nil)

    var body: some View {
        List {
            Section("隙間時間情報") {
                DebugDetailRow(label: "開始時刻", value: formatDateTime(gapTime.startTime))
                DebugDetailRow(label: "終了時刻", value: formatDateTime(gapTime.endTime))
                DebugDetailRow(label: "長さ", value: durationText)

                HStack {
                    Text("種類")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    SlotTypeBadge(slotType: gapTime.slotType)
                }
            }

            if let beforeEvent = surroundingEvents.before {
                Section("直前のイベント") {
                    DebugDetailRow(label: "開始", value: formatDateTime(beforeEvent.startTime))
                    DebugDetailRow(label: "終了", value: formatDateTime(beforeEvent.endTime))

                    HStack {
                        Text("ステータス")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        StatusBadge(status: beforeEvent.status)
                    }
                }
            }

            if let afterEvent = surroundingEvents.after {
                Section("直後のイベント") {
                    DebugDetailRow(label: "開始", value: formatDateTime(afterEvent.startTime))
                    DebugDetailRow(label: "終了", value: formatDateTime(afterEvent.endTime))

                    HStack {
                        Text("ステータス")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        StatusBadge(status: afterEvent.status)
                    }
                }
            }

            if surroundingEvents.before == nil && surroundingEvents.after == nil {
                Section {
                    Text("前後のイベント情報がありません")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .navigationTitle("隙間時間詳細")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            surroundingEvents = dataProvider.getSurroundingEvents(for: gapTime)
        }
    }

    private func formatDateTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "M月d日(E) HH:mm"
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }

    private var durationText: String {
        let minutes = Int(gapTime.duration / 60)
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
    let slot = AvailableTimeSlot(
        startTime: Date(),
        endTime: Date().addingTimeInterval(3600),
        duration: 3600,
        slotType: .gapTime
    )
    let provider = GapTimeDebugProvider()

    return NavigationStack {
        GapTimeDetailView(gapTime: slot, dataProvider: provider)
    }
}
