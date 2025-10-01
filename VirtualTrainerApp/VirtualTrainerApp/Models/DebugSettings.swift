// DEBUG: Debug-only data model - remove before production

import Foundation

/// デバッグ設定データモデル
/// 要件対応: 8.1, 8.6
struct DebugSettings: Codable {
    var isEnabled: Bool
    var logLevel: LogLevel

    enum LogLevel: String, Codable {
        case verbose
        case debug
        case info
        case warning
        case error
    }

    static var `default`: DebugSettings {
        DebugSettings(
            isEnabled: isDebugBuild(),
            logLevel: .debug
        )
    }

    private static func isDebugBuild() -> Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }
}
