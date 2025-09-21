//
//  TimerDisplayView.swift
//  VirtualTrainerApp
//
//  Created on 2025/09/15.
//

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

/// タイマー表示ビューコンポーネント
struct TimerDisplayView: View {
    /// 残り時間（秒）
    let remainingTime: Int

    /// 最後の10秒かどうか
    let isLastTenSeconds: Bool

    /// タイマー状態
    let timerState: TimerState

    /// 手動開始可能かどうか
    let canStartManually: Bool

    /// 手動開始ボタンのアクション
    let onManualStart: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            // タイマー表示
            if timerState.isActive || timerState == .completed {
                timerDisplay
            }

            // 待機メッセージまたは手動開始ボタン
            if timerState == .waitingForRep {
                waitingContent
            }
        }
        .padding()
        .background(backgroundStyle)
        .cornerRadius(12)
        .shadow(radius: 4)
        .animation(.easeInOut(duration: 0.3), value: timerState)
        .animation(.easeInOut(duration: 0.3), value: isLastTenSeconds)
    }

    /// タイマー表示部分
    private var timerDisplay: some View {
        HStack(spacing: 8) {
            Image(systemName: timerIconName)
                .font(.title2)
                .foregroundColor(timerColor)
                .animation(.spring(), value: remainingTime)

            Text("残り時間: ")
                .font(.headline)
                .foregroundColor(.primary)

            Text("\(remainingTime)秒")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(timerColor)
                .monospacedDigit()
                .animation(.none, value: remainingTime)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    /// 待機中のコンテンツ
    private var waitingContent: some View {
        VStack(spacing: 12) {
            // 準備プロンプト
            HStack(spacing: 8) {
                Image(systemName: "figure.strengthtraining.traditional")
                    .font(.title3)
                    .foregroundColor(.blue)

                Text("準備してください")
                    .font(.headline)
                    .foregroundColor(.primary)
            }

            // 手動開始ボタン（30秒後に表示）
            if canStartManually {
                Button(action: onManualStart) {
                    HStack(spacing: 6) {
                        Image(systemName: "play.fill")
                            .font(.caption)

                        Text("手動で開始")
                            .font(.subheadline)
                            .fontWeight(.medium)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .transition(.scale.combined(with: .opacity))
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    /// タイマーの色
    private var timerColor: Color {
        if timerState == .completed {
            return .green
        } else if isLastTenSeconds {
            return .red
        } else if remainingTime <= 30 {
            return .orange
        } else {
            return .primary
        }
    }

    /// タイマーのアイコン名
    private var timerIconName: String {
        if timerState == .completed {
            return "checkmark.circle.fill"
        } else if isLastTenSeconds {
            return "timer"
        } else {
            return "timer"
        }
    }

    /// 背景スタイル
    private var backgroundStyle: some View {
        Group {
            if timerState == .completed {
                Color.green.opacity(0.1)
            } else if isLastTenSeconds {
                Color.red.opacity(0.1)
            } else if timerState == .waitingForRep {
                Color.blue.opacity(0.05)
            } else {
                #if os(iOS)
                Color(UIColor.secondarySystemBackground)
                #else
                Color(NSColor.controlBackgroundColor)
                #endif
            }
        }
    }
}

/// 待機プロンプトビュー
struct WaitingPromptView: View {
    /// 表示するメッセージ
    let message: String

    /// アイコン名（SF Symbols）
    let iconName: String

    /// プライマリカラー
    let primaryColor: Color

    init(
        message: String = "準備してください",
        iconName: String = "figure.strengthtraining.traditional",
        primaryColor: Color = .blue
    ) {
        self.message = message
        self.iconName = iconName
        self.primaryColor = primaryColor
    }

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: iconName)
                .font(.title)
                .foregroundColor(primaryColor)
                .symbolEffect(.pulse, value: true)

            VStack(alignment: .leading, spacing: 4) {
                Text(message)
                    .font(.headline)
                    .foregroundColor(.primary)

                Text("最初のレップを検出するとタイマーが開始されます")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(primaryColor.opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(primaryColor.opacity(0.3), lineWidth: 1)
        )
    }
}

// MARK: - Preview
struct TimerDisplayView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            // 通常のタイマー表示
            TimerDisplayView(
                remainingTime: 45,
                isLastTenSeconds: false,
                timerState: .running,
                canStartManually: false,
                onManualStart: {}
            )

            // 最後の10秒
            TimerDisplayView(
                remainingTime: 8,
                isLastTenSeconds: true,
                timerState: .running,
                canStartManually: false,
                onManualStart: {}
            )

            // 待機中（手動開始ボタンなし）
            TimerDisplayView(
                remainingTime: 60,
                isLastTenSeconds: false,
                timerState: .waitingForRep,
                canStartManually: false,
                onManualStart: {}
            )

            // 待機中（手動開始ボタンあり）
            TimerDisplayView(
                remainingTime: 60,
                isLastTenSeconds: false,
                timerState: .waitingForRep,
                canStartManually: true,
                onManualStart: {
                    print("Manual start tapped")
                }
            )

            // 完了状態
            TimerDisplayView(
                remainingTime: 0,
                isLastTenSeconds: false,
                timerState: .completed,
                canStartManually: false,
                onManualStart: {}
            )

            // 待機プロンプト
            WaitingPromptView()
        }
        .padding()
        #if os(iOS)
        .background(Color(UIColor.systemBackground))
        #else
        .background(Color(NSColor.windowBackgroundColor))
        #endif
    }
}