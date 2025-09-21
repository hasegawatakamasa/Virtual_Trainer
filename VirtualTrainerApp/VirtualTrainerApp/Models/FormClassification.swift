import Foundation

/// エクササイズフォームの分類結果
enum FormClassification: String, Codable, CaseIterable {
    case ready = "Ready"
    case normal = "Normal"
    case elbowError = "Elbow Error"
    case tooFast = "Too Fast"
    
    /// 表示用の日本語ラベル
    var displayName: String {
        switch self {
        case .ready:
            return "準備完了"
        case .normal:
            return "正常"
        case .elbowError:
            return "肘エラー"
        case .tooFast:
            return "速すぎます"
        }
    }
    
    /// UIの色分け用カラー
    var color: (red: Double, green: Double, blue: Double) {
        switch self {
        case .ready:
            return (red: 1.0, green: 1.0, blue: 1.0) // 白
        case .normal:
            return (red: 0.0, green: 1.0, blue: 0.0) // 緑
        case .elbowError:
            return (red: 1.0, green: 0.0, blue: 0.0) // 赤
        case .tooFast:
            return (red: 1.0, green: 1.0, blue: 0.0) // 黄
        }
    }
    
    /// フォーム分類結果の詳細情報
    struct Result: Codable, Equatable {
        let classification: FormClassification
        let confidence: Float
        let timestamp: Date
        
        init(classification: FormClassification, confidence: Float, timestamp: Date = Date()) {
            self.classification = classification
            self.confidence = confidence
            self.timestamp = timestamp
        }
        
        /// 信頼できる結果かどうか
        var isReliable: Bool {
            switch classification {
            case .ready, .tooFast:
                return true // システム判定なので常に信頼できる
            case .normal, .elbowError:
                return confidence >= 0.6 // MLモデルの信頼度閾値
            }
        }
    }
}

/// フォーム分類器の状態
enum FormClassifierState {
    case notReady
    case ready
    case analyzing
    case error(String)
    
    var description: String {
        switch self {
        case .notReady:
            return "モデル未準備"
        case .ready:
            return "準備完了"
        case .analyzing:
            return "分析中"
        case .error(let message):
            return "エラー: \(message)"
        }
    }
}