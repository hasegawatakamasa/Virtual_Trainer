// DEBUG: Debug-only model for API logging

import Foundation
import SwiftUI

/// APIログデータモデル
/// 要件対応: 7.1, 7.2, 7.3, 7.4
struct DebugAPILog: Identifiable {
    let id: UUID
    let timestamp: Date
    let endpoint: String
    let httpStatusCode: Int?
    let responseTime: TimeInterval?
    let requestHeaders: [String: String]?
    let responseBody: String?
    let error: String?

    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        endpoint: String,
        httpStatusCode: Int? = nil,
        responseTime: TimeInterval? = nil,
        requestHeaders: [String: String]? = nil,
        responseBody: String? = nil,
        error: String? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.endpoint = endpoint
        self.httpStatusCode = httpStatusCode
        self.responseTime = responseTime
        self.requestHeaders = requestHeaders
        self.responseBody = responseBody
        self.error = error
    }

    var isSuccess: Bool {
        guard let statusCode = httpStatusCode else { return false }
        return (200...299).contains(statusCode)
    }

    var statusColor: Color {
        isSuccess ? .green : .red
    }

    /// 認証トークンをマスキングしたリクエストヘッダー
    var maskedRequestHeaders: [String: String]? {
        guard var headers = requestHeaders else { return nil }

        if let authHeader = headers["Authorization"] {
            // "Bearer abc123..." → "Bearer ****"
            let masked = authHeader.replacingOccurrences(
                of: #"Bearer\s+\S+"#,
                with: "Bearer ****",
                options: .regularExpression
            )
            headers["Authorization"] = masked
        }

        return headers
    }
}
