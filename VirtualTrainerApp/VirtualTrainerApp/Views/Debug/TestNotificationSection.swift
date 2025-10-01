// DEBUG: Test notification UI component - remove in production

import SwiftUI
import UserNotifications

/// テスト通知送信セクション
/// 要件対応: 1.1, 1.5, 1.6, 1.7
struct TestNotificationSection: View {
    @StateObject private var testService = TestNotificationService()
    @State private var showError = false
    @State private var errorMessage = ""

    var body: some View {
        Section {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "bell.badge")
                        .foregroundColor(.blue)
                    Text("テスト通知")
                        .font(.headline)
                }

                Text("通知機能が正常に動作するか確認できます。")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Button(action: {
                    Task {
                        do {
                            try await testService.sendTestNotification()
                        } catch {
                            errorMessage = error.localizedDescription
                            showError = true
                        }
                    }
                }) {
                    HStack {
                        if case .sending = testService.sendingStatus {
                            ProgressView()
                                .progressViewStyle(.circular)
                        } else {
                            Image(systemName: "paperplane.fill")
                        }
                        Text("テスト通知を送信")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(isSending)

                // ステータス表示
                if case .success = testService.sendingStatus {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("テスト通知を送信しました（5秒後に配信されます）")
                            .font(.caption)
                    }
                    .padding(.vertical, 4)
                    .onAppear {
                        // 3秒後にステータスをリセット
                        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                            testService.resetStatus()
                        }
                    }
                }

                if case .failure(let error) = testService.sendingStatus {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text(error.localizedDescription)
                            .font(.caption)
                    }
                    .padding(.vertical, 4)
                }
            }
            .padding(.vertical, 4)
        } header: {
            Text("テスト通知")
        }
        .alert("エラー", isPresented: $showError) {
            Button("設定アプリを開く", role: .cancel) {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            Button("キャンセル", role: .destructive) {}
        } message: {
            Text(errorMessage)
        }
    }

    private var isSending: Bool {
        if case .sending = testService.sendingStatus {
            return true
        }
        return false
    }
}

#Preview {
    List {
        TestNotificationSection()
    }
}
