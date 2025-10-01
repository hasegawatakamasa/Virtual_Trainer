// DEBUG: Gap time detection inspector - remove in production

import SwiftUI

/// 隙間時間検出結果表示
/// 要件対応: 5.1, 5.2, 5.3, 5.4, 5.5, 5.7
struct GapTimeDetectionView: View {
    @StateObject private var dataProvider = GapTimeDebugProvider()
    @State private var isLoading = false

    var body: some View {
        List {
            // サマリーセクション
            Section {
                HStack {
                    Text("検出された隙間時間")
                    Spacer()
                    Text("\(dataProvider.gapTimes.count)件")
                        .foregroundColor(.secondary)
                }

                if !dataProvider.excludedSlots.isEmpty {
                    HStack {
                        Text("除外された短い隙間")
                        Spacer()
                        Text("\(dataProvider.excludedSlots.count)件")
                            .foregroundColor(.secondary)
                    }
                }
            } header: {
                Text("サマリー")
            }

            // 隙間時間一覧
            if !dataProvider.gapTimes.isEmpty {
                let groupedSlots = Dictionary(grouping: dataProvider.gapTimes) { slot in
                    Calendar.current.startOfDay(for: slot.startTime)
                }

                ForEach(groupedSlots.keys.sorted(), id: \.self) { date in
                    Section {
                        ForEach(groupedSlots[date] ?? [], id: \.id) { slot in
                            NavigationLink(destination: GapTimeDetailView(gapTime: slot, dataProvider: dataProvider)) {
                                GapTimeRow(slot: slot)
                            }
                        }
                    } header: {
                        Text(formatDate(date))
                    }
                }
            } else if !isLoading {
                Section {
                    VStack(spacing: 12) {
                        Image(systemName: "clock.badge.checkmark")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("今後7日間に隙間時間は検出されませんでした")
                            .font(.headline)
                        Text("カレンダーを更新するか、予定を追加してください")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
            }

            // 除外された隙間時間
            if !dataProvider.excludedSlots.isEmpty {
                Section {
                    ForEach(dataProvider.excludedSlots, id: \.id) { slot in
                        GapTimeRow(slot: slot)
                    }
                } header: {
                    Text("検出されたが通知候補から除外")
                } footer: {
                    Text("30分未満の隙間時間は通知候補に含まれません")
                }
            }
        }
        .navigationTitle("隙間時間検出")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadData()
        }
        .refreshable {
            loadData()
        }
        .overlay {
            if isLoading {
                ProgressView("読み込み中...")
            }
        }
    }

    private func loadData() {
        isLoading = true
        Task {
            await dataProvider.loadGapTimes()
            isLoading = false
        }
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "M月d日(E)"
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }
}

struct GapTimeRow: View {
    let slot: AvailableTimeSlot

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(formatTime(slot.startTime))
                    .font(.headline)
                Text("-")
                    .foregroundColor(.secondary)
                Text(formatTime(slot.endTime))
                    .font(.headline)

                Spacer()

                SlotTypeBadge(slotType: slot.slotType)
            }

            HStack {
                Text("\(durationMinutes)分")
                    .font(.caption)
                    .foregroundColor(.secondary)

                if let notificationTime = suggestedNotificationTime {
                    Spacer()
                    Label(formatTime(notificationTime), systemImage: "bell.fill")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }

    private var durationMinutes: Int {
        Int(slot.duration / 60)
    }

    private var suggestedNotificationTime: Date? {
        // 隙間時間の開始10分前を通知時刻として提案
        slot.startTime.addingTimeInterval(-10 * 60)
    }
}

struct SlotTypeBadge: View {
    let slotType: AvailableTimeSlot.TimeSlotType

    var body: some View {
        Text(slotTypeText)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(slotTypeColor.opacity(0.2))
            .foregroundColor(slotTypeColor)
            .cornerRadius(4)
    }

    private var slotTypeText: String {
        switch slotType {
        case .gapTime:
            return "隙間時間"
        case .morningSlot:
            return "朝の空き"
        case .eveningSlot:
            return "夜の空き"
        case .freeDay:
            return "予定なし日"
        }
    }

    private var slotTypeColor: Color {
        switch slotType {
        case .gapTime:
            return .blue
        case .morningSlot:
            return .orange
        case .eveningSlot:
            return .purple
        case .freeDay:
            return .green
        }
    }
}

#Preview {
    NavigationStack {
        GapTimeDetectionView()
    }
}
