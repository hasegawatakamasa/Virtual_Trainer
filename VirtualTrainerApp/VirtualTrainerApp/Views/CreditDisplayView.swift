import SwiftUI
#if os(iOS)
import UIKit
#endif

/// VOICEVOX クレジット表示ビュー
struct CreditDisplayView: View {
    let credits: [String]
    @State private var showingLicenseSheet = false
    
    init(credits: [String] = []) {
        // デフォルトでは全キャラクターのクレジットを表示
        if credits.isEmpty {
            self.credits = VoiceCharacter.allCases.map { $0.creditText }
        } else {
            self.credits = credits
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("音声提供")
                .font(.caption2)
                .foregroundColor(.secondary)
            
            VStack(alignment: .leading, spacing: 4) {
                ForEach(credits, id: \.self) { credit in
                    Button(action: {
                        showingLicenseSheet = true
                    }) {
                        Text(credit)
                            .font(.caption)
                            .foregroundColor(.primary)
                            .underline()
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
        }
        .sheet(isPresented: $showingLicenseSheet) {
            LicenseDetailView()
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("音声提供: \(credits.joined(separator: ", "))")
        .accessibilityHint("タップしてライセンス情報を表示")
    }
}

/// ライセンス詳細表示ビュー
struct LicenseDetailView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    headerSection
                    
                    characterCreditsSection
                    
                    licenseSection
                    
                    Spacer(minLength: 20)
                }
                .padding()
            }
            .navigationTitle("ライセンス情報")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                #if os(iOS)
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("閉じる") {
                        dismiss()
                    }
                }
                #else
                ToolbarItem(placement: .primaryAction) {
                    Button("閉じる") {
                        dismiss()
                    }
                }
                #endif
            }
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: "doc.text.fill")
                .font(.largeTitle)
                .foregroundColor(.accentColor)
            
            Text("音声合成ライブラリ")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("このアプリで使用されている音声は、VOICEVOX音声合成エンジンを使用して生成されています。")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
    
    private var characterCreditsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("使用キャラクター")
                .font(.headline)
            
            ForEach(VoiceCharacter.allCases) { character in
                HStack(spacing: 12) {
                    Image(systemName: character.iconName)
                        .font(.title2)
                        .foregroundColor(.accentColor)
                        .frame(width: 32, height: 32)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text(character.displayName)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        Text(character.creditText)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                }
                .padding(.vertical, 4)
            }
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }
    
    private var licenseSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("利用規約")
                .font(.headline)
            
            Text("四国めたんの音声ライブラリを用いて生成した音声は、適切なクレジット表記により、商用・非商用で利用可能です。")
                .font(.subheadline)
            
            Button(action: {
                if let url = URL(string: "https://zunko.jp/con_ongen_kiyaku.html") {
                    #if os(iOS)
                    UIApplication.shared.open(url)
                    #elseif os(macOS)
                    NSWorkspace.shared.open(url)
                    #endif
                }
            }) {
                HStack {
                    Image(systemName: "safari")
                        .font(.headline)
                    Text("詳細な利用規約を確認")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Spacer()
                    Image(systemName: "arrow.up.right")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color.accentColor.opacity(0.1))
                .foregroundColor(.accentColor)
                .cornerRadius(8)
            }
            
            Text("VOICEVOX プロジェクト")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.top)
            
            Text("オープンソースの音声合成エンジン「VOICEVOX」を使用しています。")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.systemBackground)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.systemGray4, lineWidth: 1)
        )
        .cornerRadius(12)
    }
}

#Preview("Credit Display") {
    VStack {
        Spacer()
        CreditDisplayView()
            .padding()
        Spacer()
    }
    #if os(iOS)
    .background(Color.systemGroupedBackground)
    #else
    .background(Color(NSColor.windowBackgroundColor))
    #endif
}

#Preview("License Detail") {
    LicenseDetailView()
}