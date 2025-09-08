import Foundation

// MARK: - Display State Management

/// UI状態表示の種類と優先度を管理するenum
enum DisplayState: CaseIterable, Equatable {
    case exerciseZone
    case formMonitoring
    case liveAudioText
    case none
    
    /// 表示優先度（数値が小さいほど高優先度）
    var priority: Int {
        switch self {
        case .liveAudioText: return 1      // 最高優先度: 音声指導中のテキスト表示
        case .formMonitoring: return 2     // 高優先度: フォーム監視中表示
        case .exerciseZone: return 3       // 中優先度: エクササイズゾーン表示
        case .none: return 0               // 表示なし
        }
    }
    
    /// 表示名
    var displayName: String {
        switch self {
        case .exerciseZone:
            return "エクササイズゾーン"
        case .formMonitoring:
            return "フォーム監視中"
        case .liveAudioText:
            return "音声指導"
        case .none:
            return ""
        }
    }
    
    /// 表示期間の種類
    var durationType: DisplayDurationType {
        switch self {
        case .exerciseZone:
            return .persistent      // 継続表示
        case .formMonitoring:
            return .conditional     // 条件付き表示
        case .liveAudioText:
            return .temporary       // 一時表示
        case .none:
            return .none
        }
    }
    
    /// この状態が他の状態を非表示にするかどうか
    func shouldHide(_ other: DisplayState) -> Bool {
        switch (self, other) {
        case (.liveAudioText, .exerciseZone), (.liveAudioText, .formMonitoring):
            return true // ライブテキスト表示中は他を非表示
        case (.formMonitoring, .exerciseZone):
            return true // フォーム監視中はエクササイズゾーン表示を非表示
        default:
            return false
        }
    }
    
    /// 状態の互換性をチェック
    func isCompatibleWith(_ other: DisplayState) -> Bool {
        return !shouldHide(other) && !other.shouldHide(self)
    }
}

// MARK: - Display Duration Type

/// 表示期間の種類
enum DisplayDurationType {
    case persistent     // 継続表示（手動で変更されるまで表示）
    case conditional    // 条件付き表示（特定の条件下でのみ表示）
    case temporary      // 一時表示（一定時間後に自動非表示）
    case none           // 表示なし
}

// MARK: - Display State Transition

/// 状態遷移の設定
struct DisplayStateTransition {
    let from: DisplayState
    let to: DisplayState
    let animationDuration: Double
    let shouldAnimate: Bool
    
    static let `default` = DisplayStateTransition(
        from: .none,
        to: .none,
        animationDuration: 0.3,
        shouldAnimate: true
    )
    
    /// ライブテキスト表示への遷移
    static func toLiveText(from: DisplayState) -> DisplayStateTransition {
        DisplayStateTransition(
            from: from,
            to: .liveAudioText,
            animationDuration: 0.2,
            shouldAnimate: true
        )
    }
    
    /// ライブテキストからの遷移
    static func fromLiveText(to: DisplayState) -> DisplayStateTransition {
        DisplayStateTransition(
            from: .liveAudioText,
            to: to,
            animationDuration: 0.3,
            shouldAnimate: true
        )
    }
    
    /// フォーム監視への遷移
    static func toFormMonitoring(from: DisplayState) -> DisplayStateTransition {
        DisplayStateTransition(
            from: from,
            to: .formMonitoring,
            animationDuration: 0.2,
            shouldAnimate: true
        )
    }
}

// MARK: - Display State Extensions

extension DisplayState: CustomStringConvertible {
    var description: String {
        return "DisplayState.\(String(describing: self))(priority: \(priority), name: \"\(displayName)\")"
    }
}

extension DisplayState: Comparable {
    static func < (lhs: DisplayState, rhs: DisplayState) -> Bool {
        return lhs.priority < rhs.priority
    }
}

// MARK: - Display State Collection Extensions

extension Collection where Element == DisplayState {
    /// 最も優先度の高い状態を取得
    var highestPriority: DisplayState? {
        return self.filter { $0 != .none }.min()
    }
    
    /// アクティブな状態（.none以外）を取得
    var activeStates: [DisplayState] {
        return self.filter { $0 != .none }
    }
    
    /// 競合する状態を解決して最適な表示状態を決定
    func resolveConflicts() -> DisplayState {
        let activeStates = self.activeStates
        
        // アクティブな状態がない場合
        guard !activeStates.isEmpty else {
            return .none
        }
        
        // 最高優先度の状態を取得
        guard let highest = activeStates.min() else {
            return .none
        }
        
        return highest
    }
}