//
//  SessionResultView.swift
//  VirtualTrainerApp
//
//  トレーニングセッション終了後のリザルト画面
//

import SwiftUI

struct SessionResultView: View {
    // MARK: - Properties

    /// セッション完了データ
    let completionData: SessionCompletionData

    /// 画面を閉じる
    @Environment(\.dismiss) private var dismiss

    /// ルートに戻るための環境変数
    @Environment(\.presentationMode) var presentationMode

    /// トレーニングセッションサービス
    @StateObject private var sessionService = TrainingSessionService.shared

    // MARK: - Body

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // ヘッダー（完了メッセージ）
                    resultHeader

                    // メイン統計
                    mainStatsSection

                    // 詳細統計
                    detailStatsSection

                    // パフォーマンス評価
                    performanceSection

                    // アクションボタン
                    actionButtons
                }
                .padding()
            }
            .background(Color.systemGroupedBackground)
            .navigationTitle("トレーニング結果")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: {
                    #if os(iOS)
                    return .navigationBarTrailing
                    #else
                    return .automatic
                    #endif
                }()) {
                    Button("閉じる") {
                        dismiss()
                    }
                }
            }
        }
    }

    // MARK: - Subviews

    /// 結果ヘッダー
    private var resultHeader: some View {
        VStack(spacing: 12) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 60))
                .foregroundColor(.green)

            Text("トレーニング完了！")
                .font(.title)
                .fontWeight(.bold)

            Text("お疲れ様でした")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.vertical)
    }

    /// メイン統計セクション
    private var mainStatsSection: some View {
        HStack(spacing: 20) {
            // 完了レップ数
            statCard(
                title: "レップ数",
                value: "\(completionData.completedReps)",
                unit: "回",
                icon: "number.circle.fill",
                color: .blue
            )

            // トレーニング時間
            statCard(
                title: "時間",
                value: formatTime(completionData.actualDuration),
                unit: "",
                icon: "timer",
                color: .orange
            )
        }
    }

    /// 詳細統計セクション
    private var detailStatsSection: some View {
        VStack(spacing: 16) {
            Text("詳細データ")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            VStack(spacing: 12) {
                // フォームエラー
                detailRow(
                    label: "フォームエラー",
                    value: "\(completionData.formErrorCount) 回",
                    icon: "exclamationmark.triangle",
                    color: completionData.formErrorCount == 0 ? .green : .orange
                )

                // 速度警告
                detailRow(
                    label: "速度警告",
                    value: "\(completionData.speedWarningCount) 回",
                    icon: "speedometer",
                    color: completionData.speedWarningCount < 3 ? .green : .orange
                )

                // 平均ペース
                if let avgReps = completionData.averageRepsPerMinute {
                    detailRow(
                        label: "平均ペース",
                        value: String(format: "%.1f 回/分", avgReps),
                        icon: "chart.line.uptrend.xyaxis",
                        color: .blue
                    )
                }

                // 最大連続正確レップ
                detailRow(
                    label: "最大連続",
                    value: "\(completionData.maxConsecutiveCorrectReps) 回",
                    icon: "arrow.up.right.circle",
                    color: .purple
                )
            }
            .padding()
            .background(Color.systemBackground)
            .cornerRadius(12)
        }
    }

    /// パフォーマンス評価セクション
    private var performanceSection: some View {
        VStack(spacing: 16) {
            Text("パフォーマンス評価")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            VStack(spacing: 12) {
                // フォーム精度
                performanceBar(
                    label: "フォーム精度",
                    percentage: calculateFormAccuracy(),
                    color: .green
                )

                // 速度安定性
                performanceBar(
                    label: "速度安定性",
                    percentage: calculateSpeedStability(),
                    color: .blue
                )

                // 総合評価
                HStack {
                    Text("総合評価")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    Spacer()

                    Text(overallGrade)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(gradeColor)
                }
                .padding(.top, 8)
            }
            .padding()
            .background(Color.systemBackground)
            .cornerRadius(12)
        }
    }

    /// アクションボタン
    private var actionButtons: some View {
        VStack(spacing: 12) {
            // もう一度トレーニング
            Button(action: {
                // TODO: 新しいトレーニングセッションを開始
                dismiss()
            }) {
                Label("もう一度トレーニング", systemImage: "arrow.clockwise")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }

            // ホームに戻る
            Button(action: {
                // SessionResultViewを閉じる
                dismiss()
                // ExerciseDetailViewも閉じるための通知
                NotificationCenter.default.post(name: Notification.Name("ReturnToHome"), object: nil)
            }) {
                Label("ホームに戻る", systemImage: "house")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.systemGray5)
                    .foregroundColor(.primary)
                    .cornerRadius(12)
            }
        }
        .padding(.top)
    }

    // MARK: - Helper Views

    /// 統計カード
    private func statCard(title: String, value: String, unit: String, icon: String, color: Color) -> some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)

            HStack(alignment: .bottom, spacing: 2) {
                Text(value)
                    .font(.title)
                    .fontWeight(.bold)

                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.bottom, 4)
            }
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
    }

    /// 詳細行
    private func detailRow(label: String, value: String, icon: String, color: Color) -> some View {
        HStack {
            Label(label, systemImage: icon)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(color)
        }
    }

    /// パフォーマンスバー
    private func performanceBar(label: String, percentage: Double, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(label)
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Spacer()

                Text("\(Int(percentage))%")
                    .font(.subheadline)
                    .fontWeight(.medium)
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.systemGray5)
                        .frame(height: 8)

                    RoundedRectangle(cornerRadius: 4)
                        .fill(color)
                        .frame(width: geometry.size.width * (percentage / 100), height: 8)
                }
            }
            .frame(height: 8)
        }
    }

    // MARK: - Helper Methods

    /// 時間をフォーマット
    private func formatTime(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60

        if minutes > 0 {
            return String(format: "%d:%02d", minutes, remainingSeconds)
        } else {
            return "\(remainingSeconds)秒"
        }
    }

    /// フォーム精度を計算
    private func calculateFormAccuracy() -> Double {
        guard completionData.completedReps > 0 else { return 0 }
        let errorRate = Double(completionData.formErrorCount) / Double(completionData.completedReps)
        return max(0, (1.0 - errorRate) * 100)
    }

    /// 速度安定性を計算
    private func calculateSpeedStability() -> Double {
        guard completionData.completedReps > 0 else { return 0 }
        let warningRate = Double(completionData.speedWarningCount) / Double(completionData.completedReps)
        return max(0, (1.0 - warningRate * 2) * 100) // 警告が多いほど減点
    }

    /// 総合評価
    private var overallGrade: String {
        let formScore = calculateFormAccuracy()
        let speedScore = calculateSpeedStability()
        let average = (formScore + speedScore) / 2

        switch average {
        case 90...100:
            return "S"
        case 80..<90:
            return "A"
        case 70..<80:
            return "B"
        case 60..<70:
            return "C"
        default:
            return "D"
        }
    }

    /// 評価の色
    private var gradeColor: Color {
        switch overallGrade {
        case "S":
            return .yellow
        case "A":
            return .green
        case "B":
            return .blue
        case "C":
            return .orange
        default:
            return .red
        }
    }
}

// MARK: - Preview

struct SessionResultView_Previews: PreviewProvider {
    static var previews: some View {
        SessionResultView(
            completionData: SessionCompletionData(
                startTime: Date().addingTimeInterval(-65),
                endTime: Date(),
                configuredDuration: 60,
                actualDuration: 65,
                completedReps: 25,
                completionReason: .timerCompleted,
                formErrorCount: 2,
                speedWarningCount: 3,
                averageRepsPerMinute: 23.0,
                maxConsecutiveCorrectReps: 15,
                voiceCharacter: "ずんだもん",
                exerciseType: "オーバーヘッドプレス"
            )
        )
    }
}