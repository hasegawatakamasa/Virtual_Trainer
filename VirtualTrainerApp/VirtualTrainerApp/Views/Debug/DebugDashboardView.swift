// DEBUG: Main debug dashboard - remove in production

import SwiftUI

/// デバッグダッシュボードメイン画面
/// 要件対応: 8.2
struct DebugDashboardView: View {
    @ObservedObject var debugMode: DebugModeManager

    var body: some View {
        List {
            // テスト通知セクション
            TestNotificationSection()

            // 予約通知管理セクション
            Section {
                NavigationLink {
                    ScheduledNotificationsView()
                } label: {
                    HStack {
                        Image(systemName: "clock.badge")
                            .foregroundColor(.orange)
                        VStack(alignment: .leading) {
                            Text("予約通知一覧")
                                .font(.headline)
                            Text("スケジュール済み通知の確認・キャンセル")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            } header: {
                Text("通知管理")
            }

            // Googleカレンダー連携セクション
            Section {
                NavigationLink {
                    CalendarDataView()
                } label: {
                    HStack {
                        Image(systemName: "calendar")
                            .foregroundColor(.green)
                        VStack(alignment: .leading) {
                            Text("カレンダーデータ")
                                .font(.headline)
                            Text("取得したイベントの確認")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                NavigationLink {
                    GapTimeDetectionView()
                } label: {
                    HStack {
                        Image(systemName: "clock.arrow.2.circlepath")
                            .foregroundColor(.purple)
                        VStack(alignment: .leading) {
                            Text("隙間時間検出")
                                .font(.headline)
                            Text("検出された隙間時間の確認")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

            } header: {
                Text("カレンダー連携")
            }
        }
        .navigationTitle("デバッグダッシュボード")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    NavigationStack {
        DebugDashboardView(debugMode: .shared)
    }
}
