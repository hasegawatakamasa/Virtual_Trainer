import Foundation
import Combine

class SpeedAnalyzer: ObservableObject {
    
    // MARK: - Published Properties
    @Published private(set) var currentState = SpeedFeedbackState()
    @Published private(set) var isAnalysisActive = false
    
    // MARK: - Configuration
    private let config: SpeedAnalysisConfig
    
    // MARK: - Private Properties  
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    init(config: SpeedAnalysisConfig = .default) {
        self.config = config
    }
    
    // MARK: - Public Methods
    
    /// Analyze exercise speed based on keypoints count and exercise state
    /// - Parameters:
    ///   - keypointsCount: Number of keypoints collected during the rep
    ///   - isExerciseActive: Whether user is currently in exercise zone
    /// - Returns: Exercise speed classification
    func analyzeSpeed(keypointsCount: Int, isExerciseActive: Bool) -> ExerciseSpeed {
        // Only analyze if exercise is active
        guard isExerciseActive else {
            return .normal
        }
        
        // Determine speed based on keypoints count thresholds
        let speed: ExerciseSpeed
        if keypointsCount < config.fastThreshold {
            speed = .fast
        } else if keypointsCount > config.slowThreshold {
            speed = .slow
        } else {
            speed = .normal
        }
        
        // Update current state
        DispatchQueue.main.async { [weak self] in
            self?.updateState(speed: speed, keypointsCount: keypointsCount)
        }
        
        return speed
    }
    
    /// Check if speed feedback should be played
    /// - Parameters:
    ///   - speed: Current exercise speed
    ///   - isExerciseActive: Whether exercise is currently active
    /// - Returns: Whether feedback should be played
    func shouldPlayFeedback(for speed: ExerciseSpeed, isExerciseActive: Bool) -> Bool {
        // No feedback if exercise is not active
        guard isExerciseActive else { return false }
        
        // No feedback if speed is normal
        guard speed.needsFeedback else { return false }
        
        // No feedback if in cooldown period
        guard !currentState.isInCooldown else { return false }
        
        return true
    }
    
    /// Start speed analysis session
    func startAnalysis() {
        DispatchQueue.main.async { [weak self] in
            self?.isAnalysisActive = true
            self?.resetState()
        }
    }
    
    /// Stop speed analysis session
    func stopAnalysis() {
        DispatchQueue.main.async { [weak self] in
            self?.isAnalysisActive = false
        }
    }
    
    /// Record that feedback was played
    func recordFeedbackPlayed() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            let newFeedbackCount = self.currentState.feedbackCount + 1
            self.currentState = self.currentState.updated(
                lastFeedbackTime: Date(),
                feedbackCount: newFeedbackCount
            )
        }
    }
    
    // MARK: - Private Methods
    
    private func updateState(speed: ExerciseSpeed, keypointsCount: Int) {
        currentState = currentState.updated(
            speed: speed,
            keypointsCount: keypointsCount
        )
    }
    
    private func resetState() {
        currentState = SpeedFeedbackState()
    }
}

// MARK: - Debug Helpers
extension SpeedAnalyzer {
    var debugInfo: String {
        return """
        Speed: \(currentState.currentSpeed.displayName)
        Keypoints: \(currentState.keypointsCount)
        Feedback Count: \(currentState.feedbackCount)
        In Cooldown: \(currentState.isInCooldown)
        Active: \(isAnalysisActive)
        """
    }
}