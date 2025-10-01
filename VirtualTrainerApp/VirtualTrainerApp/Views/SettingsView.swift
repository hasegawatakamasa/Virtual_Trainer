import SwiftUI

/// アプリ設定画面
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var showingCalendarSettings = false
    @State private var showingNotificationSettings = false

    var body: some View {
        NavigationStack {
            List {
                // MARK: - 通知設定セクション
                Section {
                    NavigationLink {
                        CalendarSettingsView()
                    } label: {
                        Label {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("カレンダー連携")
                                    .font(.body)
                                Text("Googleカレンダーと連携して最適なタイミングで通知を受け取る")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } icon: {
                            Image(systemName: "calendar.badge.clock")
                                .foregroundColor(.blue)
                        }
                    }

                    NavigationLink {
                        let notificationService = OshiTrainerNotificationService()
                        let settingsManager = NotificationSettingsManager(
                            notificationService: notificationService
                        )
                        NotificationSettingsView(settingsManager: settingsManager)
                    } label: {
                        Label {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("通知設定")
                                    .font(.body)
                                Text("トレーニング通知の頻度や時間帯を設定")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } icon: {
                            Image(systemName: "bell.fill")
                                .foregroundColor(.orange)
                        }
                    }
                } header: {
                    Text("通知")
                } footer: {
                    Text("推しトレーナーからのトレーニング通知を管理します。")
                }

                // MARK: - アプリ情報セクション
                Section {
                    HStack {
                        Text("バージョン")
                        Spacer()
                        Text(getAppVersion())
                            .foregroundColor(.secondary)
                    }

                    HStack {
                        Text("ビルド")
                        Spacer()
                        Text(getBuildNumber())
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text("アプリ情報")
                }
            }
            .navigationTitle("設定")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完了") {
                        dismiss()
                    }
                }
            }
        }
    }

    // MARK: - Helper Methods

    private func getAppVersion() -> String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "不明"
    }

    private func getBuildNumber() -> String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "不明"
    }
}

#Preview {
    SettingsView()
}