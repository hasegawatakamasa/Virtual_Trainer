// DEBUG: Temporary debug mode manager - remove before production release

import Foundation
import SwiftUI

/// デバッグモード管理サービス
/// 要件対応: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
@MainActor
class DebugModeManager: ObservableObject {
    @Published var isDebugModeEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isDebugModeEnabled, forKey: UserDefaultsKeys.debugModeEnabled)
        }
    }

    static let shared = DebugModeManager()

    private init() {
        // UserDefaultsから初期値を読み込み
        self.isDebugModeEnabled = UserDefaults.standard.bool(forKey: UserDefaultsKeys.debugModeEnabled)
    }

    /// デバッグセクションが表示可能かチェック
    /// Debug buildでは常にtrue、Release buildでは常にfalse
    var isDebugVisible: Bool {
        #if DEBUG
        return true
        #else
        // Release buildではUserDefaultsの設定に関わらず常にfalse
        return false
        #endif
    }

    /// デバッグモードを切り替え
    func toggleDebugMode() {
        isDebugModeEnabled.toggle()
    }
}
