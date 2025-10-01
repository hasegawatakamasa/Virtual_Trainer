import SwiftUI

/// アプリ設定画面
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var showingCalendarSettings = false
    @State private var showingNotificationSettings = false
    @ObservedObject var debugMode: DebugModeManager = .shared

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

                // DEBUG: Debug section - remove entire section before production
                #if DEBUG
                if debugMode.isDebugVisible {
                    // MARK: - デバッグセクション
                    Section {
                        Toggle(isOn: $debugMode.isDebugModeEnabled) {
                            Label {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("デバッグモード")
                                        .font(.body)
                                    Text("開発者向け機能を有効化")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            } icon: {
                                Image(systemName: "wrench.and.screwdriver")
                                    .foregroundColor(.gray)
                            }
                        }

                        if debugMode.isDebugModeEnabled {
                            NavigationLink {
                                DebugDashboardView(debugMode: debugMode)
                            } label: {
                                Label {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("デバッグダッシュボード")
                                            .font(.body)
                                        Text("通知・カレンダー機能のデバッグツール")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                } icon: {
                                    Image(systemName: "gear.badge")
                                        .foregroundColor(.purple)
                                }
                            }
                        }
                    } header: {
                        Text("デバッグ")
                    } footer: {
                        Text("開発・デバッグ用の機能です。本番環境では表示されません。")
                    }
                }
                #endif
                // DEBUG: End of debug section

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