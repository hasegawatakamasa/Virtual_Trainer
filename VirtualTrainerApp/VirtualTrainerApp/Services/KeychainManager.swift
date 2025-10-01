import Foundation
import Security

/// Keychainアクセスマネージャー
class KeychainManager {
    static let shared = KeychainManager()

    private init() {}

    // MARK: - Generic Save/Get/Delete

    /// Keychainに文字列を保存
    func save(_ string: String, forKey key: String) throws {
        guard let data = string.data(using: .utf8) else {
            throw KeychainError.encodingFailed
        }
        try save(data, forKey: key)
    }

    /// Keychainにデータを保存
    func save(_ data: Data, forKey key: String) throws {
        // 既存のアイテムを削除
        let deleteQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        // 新しいアイテムを追加
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.saveFailed(status: status)
        }
    }

    /// Keychainから文字列を取得
    func getString(forKey key: String) throws -> String? {
        guard let data = try getData(forKey: key) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    /// Keychainからデータを取得
    func getData(forKey key: String) throws -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status != errSecItemNotFound else {
            return nil
        }

        guard status == errSecSuccess else {
            throw KeychainError.retrievalFailed(status: status)
        }

        return result as? Data
    }

    /// Keychainからアイテムを削除
    func delete(forKey key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deletionFailed(status: status)
        }
    }

    // MARK: - Google OAuth Token Management

    /// Google Access Tokenを保存
    func saveGoogleAccessToken(_ token: String) throws {
        try save(token, forKey: KeychainKeys.googleAccessToken)
    }

    /// Google Refresh Tokenを保存
    func saveGoogleRefreshToken(_ token: String) throws {
        try save(token, forKey: KeychainKeys.googleRefreshToken)
    }

    /// Google Token有効期限を保存
    func saveGoogleTokenExpiry(_ expiry: Date) throws {
        let data = try JSONEncoder().encode(expiry)
        try save(data, forKey: KeychainKeys.googleTokenExpiry)
    }

    /// Google Access Tokenを取得
    func getGoogleAccessToken() throws -> String? {
        return try getString(forKey: KeychainKeys.googleAccessToken)
    }

    /// Google Refresh Tokenを取得
    func getGoogleRefreshToken() throws -> String? {
        return try getString(forKey: KeychainKeys.googleRefreshToken)
    }

    /// Google Token有効期限を取得
    func getGoogleTokenExpiry() throws -> Date? {
        guard let data = try getData(forKey: KeychainKeys.googleTokenExpiry) else {
            return nil
        }
        return try JSONDecoder().decode(Date.self, from: data)
    }

    /// 全てのGoogle OAuthトークンを削除
    func deleteGoogleTokens() throws {
        try? delete(forKey: KeychainKeys.googleAccessToken)
        try? delete(forKey: KeychainKeys.googleRefreshToken)
        try? delete(forKey: KeychainKeys.googleTokenExpiry)
    }
}

// MARK: - KeychainError

enum KeychainError: LocalizedError {
    case encodingFailed
    case saveFailed(status: OSStatus)
    case retrievalFailed(status: OSStatus)
    case deletionFailed(status: OSStatus)

    var errorDescription: String? {
        switch self {
        case .encodingFailed:
            return "データのエンコードに失敗しました"
        case .saveFailed(let status):
            return "Keychainへの保存に失敗しました (status: \(status))"
        case .retrievalFailed(let status):
            return "Keychainからの取得に失敗しました (status: \(status))"
        case .deletionFailed(let status):
            return "Keychainからの削除に失敗しました (status: \(status))"
        }
    }
}