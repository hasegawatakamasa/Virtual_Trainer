import Foundation

/// カレンダープライバシー管理
/// 要件対応: 7.1, 7.3, 7.5, 7.6, 7.7
class CalendarPrivacyManager {
    static let shared = CalendarPrivacyManager()

    private init() {}

    // MARK: - Event Sanitization

    /// イベント詳細を除去（時刻のみ抽出）
    /// - Parameter event: 生のカレンダーイベント
    /// - Returns: 時刻情報のみのイベント
    func sanitizeEvent(_ rawEvent: RawCalendarEvent) -> CalendarEvent {
        // タイトルや詳細を保存せず、時刻情報のみを抽出
        return CalendarEvent(
            id: rawEvent.id,
            startTime: rawEvent.startTime,
            endTime: rawEvent.endTime,
            status: rawEvent.status
        )
    }

    /// メモリキャッシュをクリア
    func clearEventCache() {
        // メモリキャッシュのクリア（CalendarSyncCoordinator経由）
        UserDefaults.standard.removeObject(forKey: "com.virtualtrainer.cachedCalendarEvents")
        print("[CalendarPrivacyManager] Event cache cleared")
    }

    /// プライバシーポリシーテキストを取得
    func getPrivacyPolicyText() -> String {
        return """
        ## カレンダー情報の取り扱いについて

        推しトレは、あなたのプライバシーを最優先に考えています。

        ### 取得する情報
        - カレンダーイベントの開始時刻と終了時刻のみ

        ### 取得しない情報
        - イベントのタイトル
        - イベントの詳細・説明
        - イベントの参加者
        - イベントの場所

        ### データの保存
        - カレンダー情報は**デバイス内でのみ処理**され、外部サーバーに送信されません
        - 時刻情報のみを一時的にメモリ内で使用します
        - イベントの詳細情報は一切保存されません

        ### データの利用目的
        - トレーニング通知の最適なタイミングを計算するため
        - あなたの予定に合わせた空き時間を検出するため

        ### セキュリティ
        - OAuth認証トークンは、iOSの最高セキュリティレベル（Keychain）で暗号化保存されます
        - アプリをアンインストールすると、全てのデータが自動的に削除されます

        安心してご利用ください。
        """
    }

    /// データ使用同意画面を表示すべきか判定
    func shouldShowConsentScreen() -> Bool {
        return !UserDefaults.standard.bool(forKey: UserDefaultsKeys.calendarConsentGiven)
    }

    /// ユーザー同意を記録
    func recordUserConsent() {
        UserDefaults.standard.set(true, forKey: UserDefaultsKeys.calendarConsentGiven)
        print("[CalendarPrivacyManager] User consent recorded")
    }
}

// MARK: - RawCalendarEvent

/// 生のカレンダーイベント（APIレスポンス用）
struct RawCalendarEvent {
    let id: String
    let startTime: Date
    let endTime: Date
    let status: CalendarEvent.EventStatus
    let title: String?  // 内部処理のみ、保存しない
    let description: String?  // 内部処理のみ、保存しない
}