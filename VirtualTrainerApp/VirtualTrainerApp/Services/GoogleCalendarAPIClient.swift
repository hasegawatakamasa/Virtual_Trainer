import Foundation

/// Google Calendar APIクライアント
/// 要件対応: 2.1, 2.2, 2.7
class GoogleCalendarAPIClient {
    private let session: URLSession
    private let authService: GoogleCalendarAuthService

    private let baseURL = "https://www.googleapis.com/calendar/v3"

    init(session: URLSession = .shared, authService: GoogleCalendarAuthService) {
        self.session = session
        self.authService = authService
    }

    // MARK: - API Methods

    /// 指定期間のカレンダーイベントを取得
    /// - Parameters:
    ///   - startDate: 取得開始日
    ///   - endDate: 取得終了日
    /// - Returns: カレンダーイベントの配列
    func fetchEvents(from startDate: Date, to endDate: Date) async throws -> [CalendarEvent] {
        // 日付範囲のバリデーション
        try validateDateRange(start: startDate, end: endDate)

        let accessToken = try await authService.getAccessToken()

        // RFC3339形式で日付をフォーマット
        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withInternetDateTime]

        let timeMin = dateFormatter.string(from: startDate)
        let timeMax = dateFormatter.string(from: endDate)

        var components = URLComponents(string: "\(baseURL)/calendars/primary/events")!
        components.queryItems = [
            URLQueryItem(name: "timeMin", value: timeMin),
            URLQueryItem(name: "timeMax", value: timeMax),
            URLQueryItem(name: "singleEvents", value: "true"),
            URLQueryItem(name: "orderBy", value: "startTime")
        ]

        guard let url = components.url else {
            throw GoogleCalendarError.invalidResponse
        }

        var request = URLRequest(url: url)
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // リトライ付きでAPIを呼び出し
        return try await executeWithRetry(maxRetries: 3) {
            let (data, response) = try await self.session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw GoogleCalendarError.invalidResponse
            }

            // ステータスコードのチェック
            switch httpResponse.statusCode {
            case 200:
                return try self.parseEventsResponse(data)
            case 401:
                throw GoogleCalendarError.tokenExpired
            case 403:
                throw GoogleCalendarError.permissionDenied
            case 404:
                throw GoogleCalendarError.calendarNotFound
            case 429:
                // Rate limit exceeded
                let retryAfter = httpResponse.value(forHTTPHeaderField: "Retry-After").flatMap(Double.init) ?? 60
                throw GoogleCalendarError.rateLimitExceeded(retryAfter: retryAfter)
            case 500...599:
                throw GoogleCalendarError.apiError(statusCode: httpResponse.statusCode, message: "Server error")
            default:
                throw GoogleCalendarError.apiError(
                    statusCode: httpResponse.statusCode,
                    message: "Unexpected status code"
                )
            }
        }
    }

    /// プライマリカレンダーの情報を取得
    func fetchPrimaryCalendar() async throws -> CalendarInfo {
        let accessToken = try await authService.getAccessToken()
        let url = URL(string: "\(baseURL)/calendars/primary")!

        var request = URLRequest(url: url)
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw GoogleCalendarError.invalidResponse
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let id = json?["id"] as? String,
              let summary = json?["summary"] as? String,
              let timeZone = json?["timeZone"] as? String else {
            throw GoogleCalendarError.invalidResponse
        }

        return CalendarInfo(id: id, summary: summary, timeZone: timeZone)
    }

    // MARK: - Helper Methods

    /// API呼び出しをリトライ付きで実行
    /// - Parameters:
    ///   - maxRetries: 最大リトライ回数
    ///   - operation: 実行する非同期操作
    func executeWithRetry<T>(maxRetries: Int = 3, operation: () async throws -> T) async throws -> T {
        var lastError: Error?

        for attempt in 1...maxRetries {
            do {
                return try await operation()
            } catch let error as GoogleCalendarError {
                lastError = error

                switch error {
                case .networkError, .apiError(500...599, _):
                    // ネットワークエラーとサーバーエラーはリトライ
                    let delay = TimeInterval(attempt * 2)  // Exponential backoff
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                case .rateLimitExceeded(let retryAfter):
                    // レート制限はリトライ時間を考慮
                    try await Task.sleep(nanoseconds: UInt64(retryAfter * 1_000_000_000))
                    continue
                default:
                    // その他のエラーは即座に失敗
                    throw error
                }
            } catch {
                throw GoogleCalendarError.networkError(underlying: error)
            }
        }

        throw lastError ?? GoogleCalendarError.networkError(underlying: NSError(domain: "RetryFailed", code: -1))
    }

    /// イベントレスポンスをパース
    private func parseEventsResponse(_ data: Data) throws -> [CalendarEvent] {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["items"] as? [[String: Any]] else {
            throw GoogleCalendarError.invalidResponse
        }

        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        var events: [CalendarEvent] = []

        for item in items {
            guard let id = item["id"] as? String,
                  let start = item["start"] as? [String: Any],
                  let end = item["end"] as? [String: Any] else {
                continue
            }

            // 開始時刻と終了時刻を取得（dateTime or date）
            let startTimeString = start["dateTime"] as? String ?? start["date"] as? String
            let endTimeString = end["dateTime"] as? String ?? end["date"] as? String

            guard let startTimeString = startTimeString,
                  let endTimeString = endTimeString,
                  let startTime = dateFormatter.date(from: startTimeString) ?? ISO8601DateFormatter().date(from: startTimeString),
                  let endTime = dateFormatter.date(from: endTimeString) ?? ISO8601DateFormatter().date(from: endTimeString) else {
                continue
            }

            // ステータス取得（デフォルトはconfirmed）
            let statusString = item["status"] as? String ?? "confirmed"
            let status = CalendarEvent.EventStatus(rawValue: statusString) ?? .confirmed

            let event = CalendarEvent(
                id: id,
                startTime: startTime,
                endTime: endTime,
                status: status
            )

            events.append(event)
        }

        return events
    }

    /// 日付範囲を検証
    private func validateDateRange(start: Date, end: Date) throws {
        guard start < end else {
            throw ValidationError.invalidDateRange
        }
        guard end.timeIntervalSince(start) <= 7 * 24 * 3600 else {
            throw ValidationError.dateRangeTooLarge
        }
    }
}