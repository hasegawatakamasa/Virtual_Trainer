// DEBUG: Calendar data provider for debugging - remove in production

import Foundation
import SwiftUI

/// カレンダーデバッグデータ提供サービス
/// 要件対応: 4.1, 4.2, 4.4
@MainActor
class CalendarDebugDataProvider: ObservableObject {
    @Published var events: [CalendarEvent] = []
    @Published var lastFetchDate: Date?
    @Published var fetchError: String?
    @Published var isLoading = false
    @Published var isCalendarConnected = false

    private let apiClient: GoogleCalendarAPIClient?
    private let analyzer: CalendarEventAnalyzer

    init() {
        // カレンダー連携状態を確認
        let isConnected = UserDefaults.standard.bool(forKey: UserDefaultsKeys.isCalendarConnected)
        self.isCalendarConnected = isConnected

        if isConnected {
            let authService = GoogleCalendarAuthService()
            self.apiClient = GoogleCalendarAPIClient(authService: authService)
        } else {
            self.apiClient = nil
        }

        self.analyzer = CalendarEventAnalyzer()
    }

    /// 最新のカレンダーデータを取得
    func loadRecentData() {
        guard isCalendarConnected else {
            fetchError = "カレンダーが連携されていません"
            return
        }

        // UserDefaultsから最終取得日時を読み込み
        if let lastSync = UserDefaults.standard.object(forKey: UserDefaultsKeys.calendarLastSyncTime) as? Date {
            lastFetchDate = lastSync
        }

        // キャッシュされたイベントがあれば表示（実際の実装に依存）
        // ここではダミーデータを使用
        fetchError = "データを取得するには「最新データを取得」をタップしてください"
    }

    /// カレンダーを手動で同期
    func manualSync() async throws {
        guard isCalendarConnected, let apiClient = apiClient else {
            fetchError = "カレンダーが連携されていません"
            throw CalendarDebugError.notConnected
        }

        isLoading = true
        fetchError = nil

        do {
            let calendar = Calendar.current
            let now = Date()
            guard let endDate = calendar.date(byAdding: .day, value: 7, to: now) else {
                throw CalendarDebugError.invalidDateRange
            }

            // APIからイベントを取得
            let fetchedEvents = try await apiClient.fetchEvents(from: now, to: endDate)

            events = fetchedEvents
            lastFetchDate = Date()

            // 最終同期時刻を保存
            UserDefaults.standard.set(lastFetchDate, forKey: UserDefaultsKeys.calendarLastSyncTime)

            isLoading = false
        } catch {
            fetchError = error.localizedDescription
            isLoading = false
            throw error
        }
    }

    /// イベント数を取得
    var eventCount: Int {
        events.count
    }

    /// 取得期間（今日から7日間）を取得
    var fetchPeriod: DateInterval {
        let calendar = Calendar.current
        let now = Date()
        let endDate = calendar.date(byAdding: .day, value: 7, to: now) ?? now
        return DateInterval(start: now, end: endDate)
    }
}

enum CalendarDebugError: LocalizedError {
    case notConnected
    case invalidDateRange

    var errorDescription: String? {
        switch self {
        case .notConnected:
            return "カレンダーが連携されていません"
        case .invalidDateRange:
            return "日付範囲が不正です"
        }
    }
}
