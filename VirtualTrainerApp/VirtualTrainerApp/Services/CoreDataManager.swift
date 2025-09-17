import Foundation
import CoreData

class CoreDataManager: ObservableObject {
    static let shared = CoreDataManager()
    
    // MARK: - Core Data Stack  
    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "VirtualTrainerApp")
        
        // Local Storage Configuration (without CloudKit)
        container.loadPersistentStores { _, error in
            if let error = error as NSError? {
                // Replace this implementation with code to handle the error appropriately
                print("Unresolved error \(error), \(error.userInfo)")
            }
        }
        
        container.viewContext.automaticallyMergesChangesFromParent = true
        return container
    }()
    
    var viewContext: NSManagedObjectContext {
        return persistentContainer.viewContext
    }
    
    var backgroundContext: NSManagedObjectContext {
        return persistentContainer.newBackgroundContext()
    }
    
    private init() {
        Task { @MainActor in
            cleanupIncompleteSessions()
        }
    }
    
    // MARK: - Data Cleanup
    @MainActor
    private func cleanupIncompleteSessions() {
        let request: NSFetchRequest<TrainingSession> = TrainingSession.fetchRequest()
        
        // 24時間以上前の未完了セッションを削除
        let oneDayAgo = Date().addingTimeInterval(-86400)
        request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: [
            NSPredicate(format: "endTime == nil"),
            NSPredicate(format: "startTime < %@", oneDayAgo as NSDate)
        ])
        
        do {
            let incompleteSessions = try viewContext.fetch(request)
            for session in incompleteSessions {
                print("🧹 Cleaning up incomplete session from \(session.startTime ?? Date())")
                viewContext.delete(session)
            }
            
            if !incompleteSessions.isEmpty {
                save()
                print("🧹 Cleaned up \(incompleteSessions.count) incomplete sessions")
            }
        } catch {
            print("❌ Failed to cleanup incomplete sessions: \(error)")
        }
        
        // 異常データの修正
        fixAnomalousData()
    }
    
    @MainActor
    func fixAnomalousData() {
        let request: NSFetchRequest<TrainingSession> = TrainingSession.fetchRequest()
        
        do {
            let allSessions = try viewContext.fetch(request)
            var fixedCount = 0
            
            for session in allSessions {
                var needsFix = false
                
                // formErrors > totalRepsの修正
                if session.formErrors > session.totalReps {
                    print("🔧 Fixing session \(session.id?.uuidString ?? ""): errors(\(session.formErrors)) > reps(\(session.totalReps))")
                    session.formErrors = max(0, session.totalReps / 2) // エラー率を50%に制限
                    needsFix = true
                }
                
                // 0レップだが終了しているセッションを削除
                if session.totalReps == 0 && session.endTime != nil {
                    print("🗑️ Deleting empty completed session \(session.id?.uuidString ?? "")")
                    viewContext.delete(session)
                    needsFix = true
                }
                
                if needsFix {
                    fixedCount += 1
                }
            }
            
            if fixedCount > 0 {
                save()
                print("✅ Fixed \(fixedCount) anomalous sessions")
            }
        } catch {
            print("❌ Failed to fix anomalous data: \(error)")
        }
    }
    
    // MARK: - Core Data Saving
    @MainActor
    func save() {
        let context = persistentContainer.viewContext
        
        guard context.hasChanges else { return }
        
        do {
            try context.save()
            print("💾 Core Data: Successfully saved to main context")
        } catch {
            let nsError = error as NSError
            print("❌ Core Data save error: \(nsError), \(nsError.userInfo)")
            
            // Reset context on save failure to prevent corruption
            context.rollback()
        }
    }
    
    func saveBackground(context: NSManagedObjectContext) {
        context.perform {
            guard context.hasChanges else { return }
            
            do {
                try context.save()
                print("💾 Core Data: Successfully saved to background context")
            } catch {
                let nsError = error as NSError
                print("❌ Background save error: \(nsError), \(nsError.userInfo)")
                context.rollback()
            }
        }
    }
    
    // MARK: - Training Session CRUD Operations
    @MainActor
    @discardableResult
    func createTrainingSession(
        exerciseType: String,
        characterName: String,
        startTime: Date = Date()
    ) -> TrainingSession {
        let context = viewContext
        let session = TrainingSession(context: context)
        
        session.id = UUID()
        session.exerciseType = exerciseType
        session.characterName = characterName
        session.startTime = startTime
        session.totalReps = 0
        session.formErrors = 0
        session.speedWarnings = 0
        session.sessionDuration = 0
        
        save()
        return session
    }
    
    @MainActor
    func updateTrainingSession(
        _ session: TrainingSession,
        endTime: Date? = nil,
        totalReps: Int32? = nil,
        formErrors: Int32? = nil,
        speedWarnings: Int32? = nil,
        sessionDuration: Double? = nil
    ) {
        if let endTime = endTime {
            session.endTime = endTime
            if let startTime = session.startTime {
                session.sessionDuration = endTime.timeIntervalSince(startTime)
            }
        }
        if let totalReps = totalReps {
            session.totalReps = totalReps
        }
        if let formErrors = formErrors {
            session.formErrors = formErrors
        }
        if let speedWarnings = speedWarnings {
            session.speedWarnings = speedWarnings
        }
        if let sessionDuration = sessionDuration {
            session.sessionDuration = sessionDuration
        }
        
        save()
    }
    
    func fetchTrainingSessions(
        from startDate: Date? = nil,
        to endDate: Date? = nil,
        exerciseType: String? = nil
    ) -> [TrainingSession] {
        let request: NSFetchRequest<TrainingSession> = TrainingSession.fetchRequest()
        var predicates: [NSPredicate] = []
        
        // 完了したセッションのみを取得（totalReps > 0 かつ endTime != nil）
        predicates.append(NSPredicate(format: "totalReps > 0"))
        predicates.append(NSPredicate(format: "endTime != nil"))
        
        if let startDate = startDate {
            predicates.append(NSPredicate(format: "startTime >= %@", startDate as NSDate))
        }
        if let endDate = endDate {
            predicates.append(NSPredicate(format: "startTime <= %@", endDate as NSDate))
        }
        if let exerciseType = exerciseType {
            predicates.append(NSPredicate(format: "exerciseType == %@", exerciseType))
        }
        
        request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TrainingSession.startTime, ascending: false)]
        
        do {
            let sessions = try viewContext.fetch(request)
            
            // データ整合性チェック（formErrors > totalRepsの異常データを修正）
            return sessions.map { session in
                if session.formErrors > session.totalReps {
                    print("⚠️ Data integrity fix: Session \(session.id?.uuidString ?? "") has more errors than reps. Capping errors.")
                    session.formErrors = max(0, session.totalReps - 1) // 最低1回は成功とする
                    Task { @MainActor in
                        self.save()
                    }
                }
                return session
            }
        } catch {
            print("Fetch error: \(error)")
            return []
        }
    }
    
    @MainActor
    func deleteTrainingSession(_ session: TrainingSession) {
        let context = viewContext
        context.delete(session)
        save()
    }
    
    // MARK: - Session Rep CRUD Operations
    @MainActor
    @discardableResult
    func addSessionRep(
        to session: TrainingSession,
        formQuality: String,
        keypointConfidence: Double,
        timestamp: Date = Date()
    ) -> SessionRep {
        let context = viewContext
        let rep = SessionRep(context: context)
        
        rep.id = UUID()
        rep.formQuality = formQuality
        rep.keypointConfidence = keypointConfidence
        rep.timestamp = timestamp
        rep.session = session
        
        save()
        return rep
    }
    
    // MARK: - Achievement CRUD Operations
    @MainActor
    @discardableResult
    func createAchievement(
        type: String,
        characterName: String,
        bondPointsAwarded: Int32,
        unlockedAt: Date = Date()
    ) -> Achievement {
        let context = viewContext
        let achievement = Achievement(context: context)
        
        achievement.id = UUID()
        achievement.type = type
        achievement.characterName = characterName
        achievement.bondPointsAwarded = bondPointsAwarded
        achievement.unlockedAt = unlockedAt
        
        save()
        return achievement
    }
    
    func fetchAchievements(for characterName: String? = nil) -> [Achievement] {
        let request: NSFetchRequest<Achievement> = Achievement.fetchRequest()
        
        if let characterName = characterName {
            request.predicate = NSPredicate(format: "characterName == %@", characterName)
        }
        
        request.sortDescriptors = [NSSortDescriptor(keyPath: \Achievement.unlockedAt, ascending: false)]
        
        do {
            return try viewContext.fetch(request)
        } catch {
            print("Fetch achievements error: \(error)")
            return []
        }
    }
    
    // MARK: - Statistics and Analytics
    func calculateWeeklyStats(for exerciseType: String? = nil) -> WeeklyStats {
        let calendar = Calendar.current
        let now = Date()
        let weekAgo = calendar.date(byAdding: .day, value: -7, to: now) ?? now
        
        let sessions = fetchTrainingSessions(from: weekAgo, to: now, exerciseType: exerciseType)
        
        let totalSessions = sessions.count
        let totalReps = sessions.reduce(0) { $0 + Int($1.totalReps) }
        let totalDuration = sessions.reduce(0) { $0 + $1.sessionDuration }
        let averageFormAccuracy = calculateFormAccuracy(sessions: sessions)
        
        return WeeklyStats(
            totalSessions: totalSessions,
            totalReps: totalReps,
            totalDuration: totalDuration,
            averageFormAccuracy: averageFormAccuracy
        )
    }
    
    private func calculateFormAccuracy(sessions: [TrainingSession]) -> Double {
        guard !sessions.isEmpty else { return 0.0 }
        
        let totalReps = sessions.reduce(0) { $0 + Int($1.totalReps) }
        let totalErrors = sessions.reduce(0) { $0 + min(Int($1.formErrors), Int($1.totalReps)) } // エラー数を制限
        
        guard totalReps > 0 else { return 0.0 }
        
        // 精度は0-100%の範囲内に制限
        let accuracy = Double(totalReps - totalErrors) / Double(totalReps)
        return min(1.0, max(0.0, accuracy))
    }
}

// MARK: - Data Models
struct WeeklyStats {
    let totalSessions: Int
    let totalReps: Int
    let totalDuration: Double
    let averageFormAccuracy: Double
    
    var averageDurationPerSession: Double {
        guard totalSessions > 0 else { return 0.0 }
        return totalDuration / Double(totalSessions)
    }
}

// MARK: - Error Handling
enum TrainingRecordError: Error, LocalizedError {
    case coreDataSaveError(NSError)
    case fetchError(NSError)
    case iCloudSyncError(NSError)
    case modelNotFound
    
    var errorDescription: String? {
        switch self {
        case .coreDataSaveError(let error):
            return "Core Data保存エラー: \(error.localizedDescription)"
        case .fetchError(let error):
            return "データ取得エラー: \(error.localizedDescription)"
        case .iCloudSyncError(let error):
            return "iCloud同期エラー: \(error.localizedDescription)"
        case .modelNotFound:
            return "データモデルが見つかりません"
        }
    }
}