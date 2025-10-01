import Foundation
import UIKit
import GoogleSignIn

/// Googleカレンダー認証サービス
/// 要件対応: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
@MainActor
class GoogleCalendarAuthService: ObservableObject {
    @Published var isAuthenticated: Bool = false
    @Published var authenticatedUserEmail: String?
    @Published var authError: GoogleCalendarError?

    private let keychainManager: KeychainManager
    private let calendarScope = "https://www.googleapis.com/auth/calendar.readonly"

    init(keychainManager: KeychainManager = .shared) {
        self.keychainManager = keychainManager
        Task {
            await checkAuthenticationStatus()
        }
    }

    // MARK: - Authentication

    /// OAuth認証フローを開始
    /// - Returns: 認証成功したユーザー情報
    func signIn() async throws -> GoogleUser {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let presentingViewController = windowScene.windows.first?.rootViewController else {
            throw GoogleCalendarError.authenticationFailed(reason: "View controller not found")
        }

        // クライアントIDを取得
        guard let clientID = Bundle.main.object(forInfoDictionaryKey: "GIDClientID") as? String else {
            throw GoogleCalendarError.authenticationFailed(reason: "GIDClientID not found in Info.plist")
        }

        let config = GIDConfiguration(clientID: clientID)
        GIDSignIn.sharedInstance.configuration = config

        let result = try await GIDSignIn.sharedInstance.signIn(
            withPresenting: presentingViewController,
            hint: nil,
            additionalScopes: [calendarScope]
        )

        let user = result.user
        let accessToken = user.accessToken.tokenString

        // トークンをKeychainに保存
        try keychainManager.saveGoogleAccessToken(accessToken)
        try keychainManager.saveGoogleRefreshToken(user.refreshToken.tokenString)
        try keychainManager.saveGoogleTokenExpiry(user.accessToken.expirationDate ?? Date().addingTimeInterval(3600))

        let googleUser = GoogleUser(
            userId: user.userID ?? "",
            email: user.profile?.email ?? "",
            fullName: user.profile?.name,
            profileImageURL: user.profile?.imageURL(withDimension: 200)
        )

        self.isAuthenticated = true
        self.authenticatedUserEmail = googleUser.email

        // UserDefaultsに連携状態を保存
        UserDefaults.standard.set(true, forKey: UserDefaultsKeys.isCalendarConnected)
        UserDefaults.standard.set(googleUser.email, forKey: UserDefaultsKeys.connectedGoogleEmail)

        return googleUser
    }

    /// 現在の認証状態を確認
    func checkAuthenticationStatus() async -> Bool {
        do {
            guard let _ = try keychainManager.getGoogleAccessToken(),
                  let expiry = try keychainManager.getGoogleTokenExpiry() else {
                isAuthenticated = false
                return false
            }

            // トークンが有効期限内かチェック
            if expiry > Date() {
                isAuthenticated = true
                authenticatedUserEmail = UserDefaults.standard.string(forKey: UserDefaultsKeys.connectedGoogleEmail)
                return true
            } else {
                // トークンが期限切れの場合、リフレッシュを試みる
                do {
                    _ = try await refreshAccessToken()
                    return true
                } catch {
                    isAuthenticated = false
                    return false
                }
            }
        } catch {
            isAuthenticated = false
            return false
        }
    }

    /// アクセストークンを取得（自動リフレッシュ付き）
    /// - Returns: 有効なアクセストークン
    func getAccessToken() async throws -> String {
        // トークンの有効性をチェック
        guard let expiry = try keychainManager.getGoogleTokenExpiry() else {
            throw GoogleCalendarError.tokenExpired
        }

        // 有効期限が切れている場合はリフレッシュ
        if expiry <= Date() {
            return try await refreshAccessToken()
        }

        // 有効なトークンを返す
        guard let validToken = try keychainManager.getGoogleAccessToken() else {
            throw GoogleCalendarError.tokenExpired
        }

        return validToken
    }

    /// トークンをリフレッシュ
    private func refreshAccessToken() async throws -> String {
        guard let _ = try keychainManager.getGoogleRefreshToken() else {
            throw GoogleCalendarError.tokenRefreshFailed
        }

        let result = try await GIDSignIn.sharedInstance.restorePreviousSignIn()
        let accessToken = result.accessToken.tokenString

        try keychainManager.saveGoogleAccessToken(accessToken)
        try keychainManager.saveGoogleTokenExpiry(result.accessToken.expirationDate ?? Date().addingTimeInterval(3600))

        return accessToken
    }

    /// カレンダー連携を解除
    func signOut() async throws {
        GIDSignIn.sharedInstance.signOut()

        // Keychainからトークンを削除
        try keychainManager.deleteGoogleTokens()

        // UserDefaultsの連携状態をクリア
        UserDefaults.standard.set(false, forKey: UserDefaultsKeys.isCalendarConnected)
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.connectedGoogleEmail)

        isAuthenticated = false
        authenticatedUserEmail = nil
    }

    /// トークンの有効期限を確認
    func isTokenValid() async -> Bool {
        do {
            guard let expiry = try keychainManager.getGoogleTokenExpiry() else {
                return false
            }
            return expiry > Date()
        } catch {
            return false
        }
    }
}