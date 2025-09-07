import SwiftUI
import AVFoundation

/// カメラ権限要求画面
struct PermissionView: View {
    @ObservedObject var cameraManager: CameraManager
    @State private var isRequestingPermission = false
    @State private var showingSettings = false
    
    let onPermissionGranted: () -> Void
    
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 32) {
                Spacer()
                
                // アプリアイコンとタイトル
                VStack(spacing: 16) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 80))
                        .foregroundColor(.blue)
                    
                    Text("推しトレ")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("AIエクササイズコーチ")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                
                // 説明文
                VStack(spacing: 12) {
                    Text("カメラアクセスが必要です")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .multilineTextAlignment(.center)
                    
                    Text("リアルタイムでフォームを分析し、\nエクササイズの正確性をチェックします")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .lineLimit(nil)
                }
                
                // 権限の説明
                privacyInfoView
                
                Spacer()
                
                // アクションボタン
                VStack(spacing: 16) {
                    actionButton
                    
                    if cameraManager.permissionStatus == .denied {
                        settingsButton
                    }
                }
                
                Spacer()
            }
            .padding(.horizontal, 32)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Color(.systemBackground))
        .sheet(isPresented: $showingSettings) {
            SettingsSheetView()
        }
    }
    
    // MARK: - Private Views
    
    private var privacyInfoView: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: "shield.fill")
                    .foregroundColor(.green)
                    .font(.headline)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("プライバシー保護")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text("すべての処理はデバイス上で実行されます")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            HStack(spacing: 12) {
                Image(systemName: "wifi.slash")
                    .foregroundColor(.blue)
                    .font(.headline)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("データ送信なし")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text("カメラ映像は外部に送信されません")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
        }
        .padding(20)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
    }
    
    private var actionButton: some View {
        Group {
            switch cameraManager.permissionStatus {
            case .notDetermined:
                Button(action: requestPermission) {
                    HStack {
                        if isRequestingPermission {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.9)
                        }
                        
                        Text(isRequestingPermission ? "許可を確認中..." : "カメラアクセスを許可")
                            .font(.headline)
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity, minHeight: 50)
                    .foregroundColor(.white)
                    .background(isRequestingPermission ? Color.gray : Color.blue)
                    .cornerRadius(25)
                }
                .disabled(isRequestingPermission)
                
            case .denied, .restricted:
                Button("設定で許可する") {
                    openSettings()
                }
                .font(.headline)
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity, minHeight: 50)
                .foregroundColor(.white)
                .background(Color.orange)
                .cornerRadius(25)
                
            case .authorized:
                Button("続行") {
                    onPermissionGranted()
                }
                .font(.headline)
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity, minHeight: 50)
                .foregroundColor(.white)
                .background(Color.green)
                .cornerRadius(25)
                
            @unknown default:
                Button("再試行") {
                    requestPermission()
                }
                .font(.headline)
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity, minHeight: 50)
                .foregroundColor(.white)
                .background(Color.gray)
                .cornerRadius(25)
            }
        }
    }
    
    private var settingsButton: some View {
        Button("アプリについて") {
            showingSettings = true
        }
        .font(.subheadline)
        .foregroundColor(.blue)
    }
    
    // MARK: - Actions
    
    private func requestPermission() {
        isRequestingPermission = true
        
        Task {
            let granted = await cameraManager.requestCameraPermission()
            
            await MainActor.run {
                isRequestingPermission = false
                if granted {
                    // 権限が許可された場合、少し遅延を入れてから画面遷移
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        onPermissionGranted()
                    }
                }
            }
        }
    }
    
    private func openSettings() {
        guard let settingsURL = URL(string: UIApplication.openSettingsURLString) else {
            return
        }
        
        if UIApplication.shared.canOpenURL(settingsURL) {
            UIApplication.shared.open(settingsURL)
        }
    }
}

// MARK: - Settings Sheet
private struct SettingsSheetView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // アプリ情報
                    VStack(spacing: 8) {
                        Image(systemName: "dumbbell.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.blue)
                        
                        Text("推しトレ")
                            .font(.title)
                            .fontWeight(.bold)
                        
                        Text("バージョン \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0")")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    // 機能説明
                    VStack(alignment: .leading, spacing: 16) {
                        FeatureRow(
                            icon: "camera.fill",
                            title: "リアルタイム姿勢検出",
                            description: "AIがエクササイズフォームを即座に分析"
                        )
                        
                        FeatureRow(
                            icon: "chart.line.uptrend.xyaxis",
                            title: "自動回数カウント",
                            description: "正確な動作のみをカウントして記録"
                        )
                        
                        FeatureRow(
                            icon: "shield.fill",
                            title: "完全プライバシー保護",
                            description: "すべての処理はデバイス内で完結"
                        )
                    }
                    
                    // サポート情報
                    VStack(alignment: .leading, spacing: 8) {
                        Text("サポート")
                            .font(.headline)
                            .padding(.bottom, 4)
                        
                        Text("ご不明な点やフィードバックがございましたら、アプリストアのレビューにてお知らせください。")
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                }
                .padding(24)
            }
            .navigationTitle("アプリについて")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("閉じる") {
                        dismiss()
                    }
                }
            }
        }
    }
}

private struct FeatureRow: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 32)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                
                Text(description)
                    .font(.body)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
    }
}

// MARK: - Preview
#Preview {
    PermissionView(
        cameraManager: {
            let manager = CameraManager()
            return manager
        }(),
        onPermissionGranted: {}
    )
}