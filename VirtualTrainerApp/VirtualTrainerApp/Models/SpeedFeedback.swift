import Foundation

// MARK: - Exercise Speed Classification
enum ExerciseSpeed: String, CaseIterable {
    case fast = "fast"
    case slow = "slow" 
    case normal = "normal"
    
    var displayName: String {
        switch self {
        case .fast:
            return "Fast"
        case .slow:
            return "Slow"
        case .normal:
            return "Normal"
        }
    }
    
    var needsFeedback: Bool {
        switch self {
        case .fast, .slow:
            return true
        case .normal:
            return false
        }
    }
}

// MARK: - Speed Feedback State
struct SpeedFeedbackState {
    let currentSpeed: ExerciseSpeed
    let keypointsCount: Int
    let lastFeedbackTime: Date
    let feedbackCount: Int
    let isInCooldown: Bool
    
    init(currentSpeed: ExerciseSpeed = .normal,
         keypointsCount: Int = 0,
         lastFeedbackTime: Date = Date.distantPast,
         feedbackCount: Int = 0) {
        self.currentSpeed = currentSpeed
        self.keypointsCount = keypointsCount
        self.lastFeedbackTime = lastFeedbackTime
        self.feedbackCount = feedbackCount
        
        // Calculate cooldown status (3 seconds)
        let cooldownInterval: TimeInterval = 3.0
        self.isInCooldown = Date().timeIntervalSince(lastFeedbackTime) < cooldownInterval
    }
    
    // Helper to create updated state
    func updated(speed: ExerciseSpeed? = nil,
                keypointsCount: Int? = nil,
                lastFeedbackTime: Date? = nil,
                feedbackCount: Int? = nil) -> SpeedFeedbackState {
        return SpeedFeedbackState(
            currentSpeed: speed ?? self.currentSpeed,
            keypointsCount: keypointsCount ?? self.keypointsCount,
            lastFeedbackTime: lastFeedbackTime ?? self.lastFeedbackTime,
            feedbackCount: feedbackCount ?? self.feedbackCount
        )
    }
}

// MARK: - Speed Analysis Configuration
struct SpeedAnalysisConfig {
    let fastThreshold: Int
    let slowThreshold: Int
    let cooldownSeconds: TimeInterval
    
    static let `default` = SpeedAnalysisConfig(
        fastThreshold: 8,      // Less than 8 keypoints = too fast
        slowThreshold: 15,     // More than 15 keypoints = too slow  
        cooldownSeconds: 3.0   // 3 second cooldown between feedback
    )
}