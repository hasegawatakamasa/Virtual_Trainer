import Foundation

// MARK: - GoogleCalendarError

enum GoogleCalendarError: LocalizedError {
    case authenticationFailed(reason: String)
    case tokenExpired
    case tokenRefreshFailed
    case networkError(underlying: Error)
    case apiError(statusCode: Int, message: String)
    case invalidResponse
    case permissionDenied
    case calendarNotFound
    case rateLimitExceeded(retryAfter: TimeInterval)

    var errorDescription: String? {
        switch self {
        case .authenticationFailed(let reason):
            return "認証に失敗しました: \(reason)"
        case .tokenExpired:
            return "認証トークンの有効期限が切れました"
        case .tokenRefreshFailed:
            return "認証トークンの更新に失敗しました"
        case .networkError(let error):
            return "ネットワークエラー: \(error.localizedDescription)"
        case .apiError(let code, let message):
            return "APIエラー (\(code)): \(message)"
        case .invalidResponse:
            return "無効なレスポンスを受信しました"
        case .permissionDenied:
            return "カレンダーへのアクセスが拒否されました"
        case .calendarNotFound:
            return "カレンダーが見つかりません"
        case .rateLimitExceeded(let retryAfter):
            return "リクエスト制限を超過しました。\(Int(retryAfter))秒後に再試行してください"
        }
    }
}

// MARK: - NotificationSchedulingError

enum NotificationSchedulingError: LocalizedError {
    case notificationPermissionDenied
    case schedulingFailed(reason: String)
    case invalidTimeSlot
    case noAvailableSlots
    case trainerNotFound
    case imageAttachmentFailed(underlying: Error)

    var errorDescription: String? {
        switch self {
        case .notificationPermissionDenied:
            return "通知権限が許可されていません。設定アプリで権限を有効にしてください"
        case .schedulingFailed(let reason):
            return "通知のスケジュールに失敗しました: \(reason)"
        case .invalidTimeSlot:
            return "無効な時間帯です"
        case .noAvailableSlots:
            return "通知可能な時間帯が見つかりませんでした"
        case .trainerNotFound:
            return "トレーナー情報が見つかりません"
        case .imageAttachmentFailed(let error):
            return "画像の添付に失敗しました: \(error.localizedDescription)"
        }
    }
}

// MARK: - User-Facing Error Messages

extension GoogleCalendarError {
    var userFriendlyMessage: String {
        switch self {
        case .authenticationFailed:
            return "Googleアカウントとの連携に失敗しました。もう一度お試しください。"
        case .tokenExpired, .tokenRefreshFailed:
            return "認証の有効期限が切れました。再度ログインしてください。"
        case .networkError:
            return "インターネット接続を確認して、再度お試しください。"
        case .permissionDenied:
            return "カレンダーへのアクセスが拒否されました。Google設定で権限を確認してください。"
        case .rateLimitExceeded(let retryAfter):
            return "一時的にアクセスが制限されています。\(Int(retryAfter))秒後に再度お試しください。"
        default:
            return "エラーが発生しました。しばらくしてから再度お試しください。"
        }
    }
}

extension NotificationSchedulingError {
    var userFriendlyMessage: String {
        switch self {
        case .notificationPermissionDenied:
            return "通知を送信するには、設定アプリで通知を許可してください。"
        case .noAvailableSlots:
            return "現在、通知可能な時間帯が見つかりませんでした。カレンダーの予定を確認してください。"
        case .trainerNotFound:
            return "推しトレーナーが選択されていません。設定から選択してください。"
        default:
            return "通知の設定中にエラーが発生しました。再度お試しください。"
        }
    }
}

// MARK: - ValidationError

enum ValidationError: LocalizedError {
    case invalidDateRange
    case dateRangeTooLarge

    var errorDescription: String? {
        switch self {
        case .invalidDateRange:
            return "無効な日付範囲です"
        case .dateRangeTooLarge:
            return "日付範囲が大きすぎます（最大7日間）"
        }
    }
}