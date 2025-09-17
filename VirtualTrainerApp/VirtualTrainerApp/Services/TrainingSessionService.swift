import Foundation
import Combine
import CoreData

class TrainingSessionService: ObservableObject {
    static let shared = TrainingSessionService()
    
    @Published var currentSession: TrainingSession?
    @Published var sessionStats: SessionStats = SessionStats()
    @Published var recentSessions: [TrainingSession] = []
    @Published var isSessionActive: Bool = false
    
    private let coreDataManager = CoreDataManager.shared
    private var cancellables = Set<AnyCancellable>()
    private var sessionTimer: Timer?
    
    private init() {
        loadRecentSessions()
    }
    
    // MARK: - Session Management
    func startSession(exerciseType: ExerciseType, voiceCharacter: VoiceCharacter) {
        guard currentSession == nil else {
            print("Warning: Attempting to start session while another is active")
            return
        }
        
        Task { @MainActor in
            let session = coreDataManager.createTrainingSession(
                exerciseType: exerciseType.rawValue,
                characterName: voiceCharacter.displayName,
                startTime: Date()
            )
            
            currentSession = session
            isSessionActive = true
            sessionStats = SessionStats()
            
            // Start real-time session tracking
            startSessionTimer()
            
            print("✅ Training session started: \(exerciseType.displayName) with \(voiceCharacter.displayName)")
        }
    }
    
    func recordRep(formQuality: FormClassification, keypointConfidence: Double) {
        guard let session = currentSession else {
            print("Warning: Attempting to record rep without active session")
            return
        }
        
        Task { @MainActor in
            // Add rep to Core Data
            coreDataManager.addSessionRep(
                to: session,
                formQuality: formQuality.rawValue,
                keypointConfidence: keypointConfidence,
                timestamp: Date()
            )
            
            // Update session stats
            sessionStats.totalReps += 1
            session.totalReps = Int32(sessionStats.totalReps)
            
            if formQuality != .normal {
                sessionStats.formErrors += 1
                session.formErrors = Int32(sessionStats.formErrors)
            }
            
            // Update UI
            objectWillChange.send()
            
            print("📊 Rep recorded: \(formQuality.rawValue), Total: \(sessionStats.totalReps)")
        }
    }
    
    func recordFormError() {
        guard let session = currentSession else { return }
        
        Task { @MainActor in
            sessionStats.formErrors += 1
            session.formErrors = Int32(sessionStats.formErrors)
            
            coreDataManager.save()
            objectWillChange.send()
            
            print("⚠️ Form error recorded. Total errors: \(sessionStats.formErrors)")
        }
    }
    
    func recordSpeedFeedback() {
        guard let session = currentSession else { return }
        
        Task { @MainActor in
            sessionStats.speedWarnings += 1
            session.speedWarnings = Int32(sessionStats.speedWarnings)
            
            coreDataManager.save()
            objectWillChange.send()
            
            print("🏃‍♂️ Speed feedback recorded. Total warnings: \(sessionStats.speedWarnings)")
        }
    }
    
    @MainActor
    func endSession() -> SessionSummary? {
        guard let session = currentSession else {
            print("Warning: Attempting to end session when none is active")
            return nil
        }
        
        let endTime = Date()
        
        // Update session with final stats - 同期的に実行
        coreDataManager.updateTrainingSession(
            session,
            endTime: endTime,
            totalReps: Int32(sessionStats.totalReps),
            formErrors: Int32(sessionStats.formErrors),
            speedWarnings: Int32(sessionStats.speedWarnings)
        )
        
        // Stop session timer
        stopSessionTimer()
        
        // Generate session summary - 更新後のセッションから生成
        let summary = session.toSessionSummary()
        
        // Reset current session
        currentSession = nil
        isSessionActive = false
        sessionStats = SessionStats()
        
        // Refresh recent sessions
        loadRecentSessions()
        
        print("✅ Session ended. Duration: \(summary.sessionDuration)s, Reps: \(summary.totalReps)")
        print("📊 Session saved: \(session.totalReps) reps, \(session.formErrors) errors, ended: \(session.endTime ?? Date())")
        
        return summary
    }
    
    func cancelSession() {
        guard let session = currentSession else { return }
        
        Task { @MainActor in
            // Delete the incomplete session
            coreDataManager.deleteTrainingSession(session)
        }
        
        // Reset state
        currentSession = nil
        isSessionActive = false
        sessionStats = SessionStats()
        stopSessionTimer()
        
        print("❌ Session cancelled and deleted")
    }
    
    // MARK: - Session Timer
    private func startSessionTimer() {
        sessionTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateSessionDuration()
        }
    }
    
    private func stopSessionTimer() {
        sessionTimer?.invalidate()
        sessionTimer = nil
    }
    
    private func updateSessionDuration() {
        guard let session = currentSession,
              let startTime = session.startTime else { return }
        
        let currentDuration = Date().timeIntervalSince(startTime)
        sessionStats.sessionDuration = currentDuration
        
        // Update every 10 seconds to avoid excessive Core Data writes
        if Int(currentDuration) % 10 == 0 {
            session.sessionDuration = currentDuration
            Task { @MainActor in
                coreDataManager.save()
            }
        }
        
        objectWillChange.send()
    }
    
    // MARK: - Data Loading
    func loadRecentSessions(limit: Int = 10) {
        let allSessions = coreDataManager.fetchTrainingSessions()
        recentSessions = Array(allSessions.prefix(limit))
        objectWillChange.send()
    }
    
    func fetchSessionsForDateRange(from startDate: Date, to endDate: Date) -> [TrainingSession] {
        return coreDataManager.fetchTrainingSessions(from: startDate, to: endDate)
    }
    
    func fetchSessionsForExercise(_ exerciseType: ExerciseType) -> [TrainingSession] {
        return coreDataManager.fetchTrainingSessions(exerciseType: exerciseType.rawValue)
    }
    
    // MARK: - Statistics
    func calculateFormAccuracy(for sessions: [TrainingSession]) -> Double {
        guard !sessions.isEmpty else { return 0.0 }
        
        let totalReps = sessions.reduce(0) { $0 + Int($1.totalReps) }
        let totalErrors = sessions.reduce(0) { $0 + Int($1.formErrors) }
        
        guard totalReps > 0 else { return 0.0 }
        
        return max(0.0, Double(totalReps - totalErrors) / Double(totalReps))
    }
    
    func calculateWeeklyTrends() -> WeeklyTrends {
        let calendar = Calendar.current
        let now = Date()
        let weekAgo = calendar.date(byAdding: .day, value: -7, to: now) ?? now
        let twoWeeksAgo = calendar.date(byAdding: .day, value: -14, to: now) ?? now
        
        let thisWeekSessions = fetchSessionsForDateRange(from: weekAgo, to: now)
        let lastWeekSessions = fetchSessionsForDateRange(from: twoWeeksAgo, to: weekAgo)
        
        let thisWeekReps = thisWeekSessions.reduce(0) { $0 + Int($1.totalReps) }
        let lastWeekReps = lastWeekSessions.reduce(0) { $0 + Int($1.totalReps) }
        
        let repsChange = thisWeekReps - lastWeekReps
        let repsChangePercentage = lastWeekReps > 0 ? Double(repsChange) / Double(lastWeekReps) : 0.0
        
        return WeeklyTrends(
            thisWeekReps: thisWeekReps,
            lastWeekReps: lastWeekReps,
            repsChange: repsChange,
            repsChangePercentage: repsChangePercentage,
            thisWeekSessions: thisWeekSessions.count,
            lastWeekSessions: lastWeekSessions.count
        )
    }
    
    func calculatePersonalBests() -> [PersonalRecord] {
        let allSessions = coreDataManager.fetchTrainingSessions()
        var personalBests: [String: PersonalRecord] = [:]
        
        for session in allSessions {
            let exerciseType = session.exerciseType ?? ""
            let characterName = session.characterName ?? ""
            let key = "\(exerciseType)_\(characterName)_maxReps"
            
            let currentReps = Int(session.totalReps)
            
            if let existingRecord = personalBests[key] {
                if currentReps > existingRecord.value {
                    personalBests[key] = PersonalRecord(
                        recordType: .maxReps,
                        value: currentReps,
                        exerciseType: exerciseType,
                        characterName: characterName,
                        achievedAt: session.startTime ?? Date(),
                        previousBest: existingRecord.value
                    )
                }
            } else {
                personalBests[key] = PersonalRecord(
                    recordType: .maxReps,
                    value: currentReps,
                    exerciseType: exerciseType,
                    characterName: characterName,
                    achievedAt: session.startTime ?? Date(),
                    previousBest: nil
                )
            }
        }
        
        return Array(personalBests.values)
    }
    
    // MARK: - Background Session Persistence
    func saveCurrentSessionState() {
        guard let session = currentSession else { return }
        
        Task { @MainActor in
            // Save intermediate state for background preservation
            coreDataManager.updateTrainingSession(
                session,
                totalReps: Int32(sessionStats.totalReps),
                formErrors: Int32(sessionStats.formErrors),
                speedWarnings: Int32(sessionStats.speedWarnings),
                sessionDuration: sessionStats.sessionDuration
            )
            
            print("💾 Session state saved for background preservation")
        }
    }
    
    func restoreSessionFromBackground() {
        // If app was backgrounded during session, restore incomplete session
        let incompleteSessions = coreDataManager.fetchTrainingSessions()
            .filter { $0.endTime == nil }
            .sorted { ($0.startTime ?? Date.distantPast) > ($1.startTime ?? Date.distantPast) }
        
        if let incompleteSession = incompleteSessions.first,
           let startTime = incompleteSession.startTime,
           Date().timeIntervalSince(startTime) < 3600 { // Within 1 hour
            
            currentSession = incompleteSession
            isSessionActive = true
            sessionStats = SessionStats(
                totalReps: Int(incompleteSession.totalReps),
                formErrors: Int(incompleteSession.formErrors),
                speedWarnings: Int(incompleteSession.speedWarnings),
                sessionDuration: incompleteSession.sessionDuration
            )
            
            startSessionTimer()
            print("🔄 Session restored from background")
        }
    }
}

// MARK: - Supporting Data Structures
struct SessionStats {
    var totalReps: Int = 0
    var formErrors: Int = 0
    var speedWarnings: Int = 0
    var sessionDuration: TimeInterval = 0
    
    var formAccuracy: Double {
        guard totalReps > 0 else { return 0.0 }
        return Double(totalReps - formErrors) / Double(totalReps)
    }
    
    var averageRepTime: Double {
        guard totalReps > 0, sessionDuration > 0 else { return 0.0 }
        return sessionDuration / Double(totalReps)
    }
}

struct WeeklyTrends {
    let thisWeekReps: Int
    let lastWeekReps: Int
    let repsChange: Int
    let repsChangePercentage: Double
    let thisWeekSessions: Int
    let lastWeekSessions: Int
    
    var isImproving: Bool {
        return repsChange > 0
    }
    
    var trendDescription: String {
        if repsChange > 0 {
            return "先週より\(repsChange)レップ増加！"
        } else if repsChange < 0 {
            return "先週より\(abs(repsChange))レップ減少"
        } else {
            return "先週と同じペース"
        }
    }
}

// MARK: - Integration Extensions
extension TrainingSessionService {
    func getSessionSummary() -> SessionSummary? {
        guard let session = currentSession else { return nil }
        return session.toSessionSummary()
    }
    
    func getRecentSessionSummaries(limit: Int = 5) -> [SessionSummary] {
        return recentSessions.prefix(limit).map { $0.toSessionSummary() }
    }
}