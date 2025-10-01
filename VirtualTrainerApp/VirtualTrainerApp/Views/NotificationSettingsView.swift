import SwiftUI

/// 通知設定画面
struct NotificationSettingsView: View {
    @StateObject private var settingsManager: NotificationSettingsManager
    @State private var notificationEnabled: Bool
    @State private var selectedFrequency: NotificationSettings.NotificationFrequency
    @State private var timeRangeStart: Date
    @State private var timeRangeEnd: Date
    @State private var weekendOnly: Bool
    @State private var nextNotifications: [NotificationPreview] = []
    @State private var errorMessage: String?

    init(settingsManager: NotificationSettingsManager) {
        _settingsManager = StateObject(wrappedValue: settingsManager)
        _notificationEnabled = State(initialValue: settingsManager.settings.enabled)
        _selectedFrequency = State(initialValue: settingsManager.settings.frequency)
        _timeRangeStart = State(initialValue: settingsManager.settings.timeRangeStart)
        _timeRangeEnd = State(initialValue: settingsManager.settings.timeRangeEnd)
        _weekendOnly = State(initialValue: settingsManager.settings.weekendOnly)
    }

    var body: some View {
        Form {
            // MARK: - 通知ON/OFFセクション
            Section {
                Toggle("通知を有効化", isOn: $notificationEnabled)
                    .onChange(of: notificationEnabled) { _, newValue in
                        toggleNotifications(enabled: newValue)
                    }
            } header: {
                Text("通知設定")
            } footer: {
                Text("トレーニング通知を受け取るかどうかを設定します。")
            }

            if notificationEnabled {
                // MARK: - 通知頻度セクション
                Section {
                    Picker("通知頻度", selection: $selectedFrequency) {
                        ForEach(NotificationSettings.NotificationFrequency.allCases, id: \.self) { frequency in
                            Text(frequency.displayName).tag(frequency)
                        }
                    }
                    .onChange(of: selectedFrequency) { _, newValue in
                        updateSettings()
                    }
                } header: {
                    Text("通知頻度")
                } footer: {
                    Text("1日あたりの最大通知回数を設定します。")
                }

                // MARK: - 通知時間帯セクション
                Section {
                    DatePicker("開始時刻", selection: $timeRangeStart, displayedComponents: .hourAndMinute)
                        .onChange(of: timeRangeStart) { _, _ in
                            updateSettings()
                        }

                    DatePicker("終了時刻", selection: $timeRangeEnd, displayedComponents: .hourAndMinute)
                        .onChange(of: timeRangeEnd) { _, _ in
                            updateSettings()
                        }
                } header: {
                    Text("通知時間帯")
                } footer: {
                    Text("通知を受け取る時間帯を設定します。")
                }

                // MARK: - 週末のみセクション
                Section {
                    Toggle("週末のみ通知", isOn: $weekendOnly)
                        .onChange(of: weekendOnly) { _, _ in
                            updateSettings()
                        }
                } footer: {
                    Text("土日のみ通知を受け取る場合に有効化してください。")
                }

                // MARK: - 次回通知プレビュー
                Section {
                    if nextNotifications.isEmpty {
                        Text("スケジュールされた通知はありません")
                            .foregroundColor(.secondary)
                    } else {
                        ForEach(nextNotifications) { notification in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(notification.scheduledTime, style: .date)
                                    .font(.headline)
                                Text(notification.scheduledTime, style: .time)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Text(notification.body)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                } header: {
                    Text("次回の通知")
                }

                // MARK: - 統計画面へのリンク
                Section {
                    NavigationLink {
                        NotificationStatsView()
                    } label: {
                        Label("通知効果を見る", systemImage: "chart.bar.fill")
                    }
                }
            }

            // MARK: - エラー表示
            if let errorMessage = errorMessage {
                Section {
                    Text(errorMessage)
                        .foregroundColor(.red)
                        .font(.caption)
                }
            }
        }
        .navigationTitle("通知設定")
        .task {
            await loadNextNotifications()
        }
    }

    // MARK: - Actions

    private func toggleNotifications(enabled: Bool) {
        Task {
            do {
                if enabled {
                    try await settingsManager.enableNotifications()
                } else {
                    await settingsManager.disableNotifications()
                }
                await loadNextNotifications()
            } catch {
                errorMessage = (error as? NotificationSchedulingError)?.userFriendlyMessage ?? error.localizedDescription
                notificationEnabled = false
            }
        }
    }

    private func updateSettings() {
        var newSettings = settingsManager.settings
        newSettings.frequency = selectedFrequency
        newSettings.timeRangeStart = timeRangeStart
        newSettings.timeRangeEnd = timeRangeEnd
        newSettings.weekendOnly = weekendOnly

        Task {
            await settingsManager.saveSettings(newSettings)
            await loadNextNotifications()
        }
    }

    private func loadNextNotifications() async {
        // TODO: NotificationServiceから次回通知を取得
        // nextNotifications = await settingsManager.notificationService.getNextNotifications(limit: 3)
    }
}

#Preview {
    let notificationService = OshiTrainerNotificationService()
    let settingsManager = NotificationSettingsManager(
        notificationService: notificationService
    )

    return NavigationView {
        NotificationSettingsView(settingsManager: settingsManager)
    }
}