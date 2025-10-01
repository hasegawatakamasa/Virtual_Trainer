import SwiftUI

/// カレンダー連携設定画面
struct CalendarSettingsView: View {
    @StateObject private var authService: GoogleCalendarAuthService
    @State private var syncCoordinator: CalendarSyncCoordinator?
    @State private var showingConsentSheet = false
    @State private var showingDisconnectAlert = false
    @State private var errorMessage: String?
    @State private var isLoading = false

    init() {
        let auth = GoogleCalendarAuthService()
        _authService = StateObject(wrappedValue: auth)

        // TODO: syncCoordinatorは実際のAPIClient実装後に初期化
        // _syncCoordinator = State(initialValue: nil)
    }

    var body: some View {
        Form {
            // MARK: - 連携状態セクション
            Section {
                if authService.isAuthenticated {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("連携済み")
                            .fontWeight(.semibold)
                    }

                    if let email = authService.authenticatedUserEmail {
                        HStack {
                            Text("アカウント")
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(email)
                                .font(.caption)
                        }
                    }

                    if let lastSyncTime = syncCoordinator?.getLastSyncTime() {
                        HStack {
                            Text("最終同期")
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(lastSyncTime, style: .relative)
                                .font(.caption)
                        }
                    }
                } else {
                    HStack {
                        Image(systemName: "exclamationmark.circle")
                            .foregroundColor(.orange)
                        Text("未連携")
                    }
                }
            } header: {
                Text("連携状態")
            }

            // MARK: - アクションセクション
            Section {
                if authService.isAuthenticated {
                    Button(role: .destructive) {
                        showingDisconnectAlert = true
                    } label: {
                        Label("連携を解除", systemImage: "xmark.circle")
                    }
                } else {
                    Button {
                        connectCalendar()
                    } label: {
                        if isLoading {
                            ProgressView()
                        } else {
                            Label("Googleカレンダーと連携", systemImage: "calendar.badge.plus")
                        }
                    }
                    .disabled(isLoading)
                }
            } header: {
                Text("操作")
            } footer: {
                if !authService.isAuthenticated {
                    Text("カレンダーと連携すると、あなたの予定に合わせて最適なタイミングで通知が届きます。")
                }
            }

            // MARK: - プライバシーセクション
            Section {
                NavigationLink {
                    PrivacyPolicyView()
                } label: {
                    Label("プライバシーポリシー", systemImage: "hand.raised.fill")
                }
            } header: {
                Text("プライバシー")
            }

            // MARK: - エラー表示
            if let errorMessage = errorMessage {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Label {
                            Text("エラー")
                                .fontWeight(.semibold)
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                        }

                        Text(errorMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if errorMessage.contains("GoogleSignIn SDK") {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("この機能を使用するには:")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .padding(.top, 4)

                                Text("1. GoogleSignIn SDK を追加")
                                    .font(.caption2)
                                Text("2. Google Cloud Console でプロジェクト作成")
                                    .font(.caption2)
                                Text("3. OAuth 2.0 クライアントID を取得")
                                    .font(.caption2)
                                Text("4. Info.plist を設定")
                                    .font(.caption2)
                            }
                            .foregroundColor(.secondary)
                        }
                    }
                }
            }
        }
        .navigationTitle("カレンダー連携")
        .alert("連携を解除しますか？", isPresented: $showingDisconnectAlert) {
            Button("キャンセル", role: .cancel) {}
            Button("解除", role: .destructive) {
                disconnectCalendar()
            }
        } message: {
            Text("カレンダー連携を解除すると、予定に基づく通知が停止されます。")
        }
        .sheet(isPresented: $showingConsentSheet) {
            ConsentSheetView(onAccept: {
                showingConsentSheet = false
                performConnection()
            })
        }
    }

    // MARK: - Actions

    private func connectCalendar() {
        // 初回の場合は同意画面を表示
        if CalendarPrivacyManager.shared.shouldShowConsentScreen() {
            showingConsentSheet = true
        } else {
            performConnection()
        }
    }

    private func performConnection() {
        isLoading = true
        errorMessage = nil

        Task {
            do {
                _ = try await authService.signIn()
                CalendarPrivacyManager.shared.recordUserConsent()
                isLoading = false
            } catch {
                errorMessage = (error as? GoogleCalendarError)?.userFriendlyMessage ?? error.localizedDescription
                isLoading = false
            }
        }
    }

    private func disconnectCalendar() {
        Task {
            do {
                try await authService.signOut()
                CalendarPrivacyManager.shared.clearEventCache()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}

// MARK: - ConsentSheetView

struct ConsentSheetView: View {
    let onAccept: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        PrivacyPolicyContent()
                    }
                    .padding()
                }

                // 固定ボタン
                VStack(spacing: 12) {
                    Button {
                        onAccept()
                    } label: {
                        Text("同意して連携する")
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.horizontal)
                }
                .padding(.bottom)
                .background(Color(UIColor.systemBackground))
            }
            .navigationTitle("データ使用について")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("キャンセル") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - PrivacyPolicyView

struct PrivacyPolicyView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PrivacyPolicyContent()
            }
            .padding()
        }
        .navigationTitle("プライバシーポリシー")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - PrivacyPolicyContent

struct PrivacyPolicyContent: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // イントロダクション
            VStack(alignment: .leading, spacing: 8) {
                Text("Virtual Trainerは、あなたのプライバシーを最優先に考えています。")
                    .font(.body)

                Text("カレンダー連携により、以下の情報のみを取得します：")
                    .font(.body)
                    .fontWeight(.semibold)
                    .padding(.top, 8)
            }

            Divider()

            // 取得する情報
            VStack(alignment: .leading, spacing: 8) {
                Text("取得する情報")
                    .font(.headline)

                BulletPoint(text: "カレンダーイベントの開始時刻と終了時刻のみ")
            }

            // 取得しない情報
            VStack(alignment: .leading, spacing: 8) {
                Text("取得しない情報")
                    .font(.headline)

                BulletPoint(text: "イベントのタイトル")
                BulletPoint(text: "イベントの詳細・説明")
                BulletPoint(text: "イベントの参加者")
                BulletPoint(text: "イベントの場所")
            }

            Divider()

            // データの保存
            VStack(alignment: .leading, spacing: 8) {
                Text("データの保存")
                    .font(.headline)

                BulletPoint(text: "カレンダー情報はデバイス内でのみ処理され、外部サーバーに送信されません", isBold: true)
                BulletPoint(text: "時刻情報のみを一時的にメモリ内で使用します")
                BulletPoint(text: "イベントの詳細情報は一切保存されません")
            }

            // データの利用目的
            VStack(alignment: .leading, spacing: 8) {
                Text("データの利用目的")
                    .font(.headline)

                BulletPoint(text: "トレーニング通知の最適なタイミングを計算するため")
                BulletPoint(text: "あなたの予定に合わせた空き時間を検出するため")
            }

            Divider()

            // セキュリティ
            VStack(alignment: .leading, spacing: 8) {
                Text("セキュリティ")
                    .font(.headline)

                BulletPoint(text: "OAuth認証トークンは、iOSの最高セキュリティレベル（Keychain）で暗号化保存されます")
                BulletPoint(text: "アプリをアンインストールすると、全てのデータが自動的に削除されます")
            }

            Text("安心してご利用ください。")
                .font(.body)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
                .padding(.top, 8)
        }
    }
}

// MARK: - BulletPoint

struct BulletPoint: View {
    let text: String
    var isBold: Bool = false

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•")
                .font(.body)
                .foregroundColor(.secondary)

            Text(text)
                .font(.body)
                .fontWeight(isBold ? .semibold : .regular)
        }
    }
}

#Preview {
    NavigationView {
        CalendarSettingsView()
    }
}