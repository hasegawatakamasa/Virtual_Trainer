import SwiftUI

@main
struct VirtualTrainerAppApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @Environment(\.scenePhase) private var scenePhase

    init() {
        // UserDefaultsのデフォルト値を設定
        UserDefaultsKeys.registerDefaults()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .onChange(of: scenePhase) { oldPhase, newPhase in
            handleScenePhaseChange(from: oldPhase, to: newPhase)
        }
    }

    // MARK: - Scene Phase Handling

    private func handleScenePhaseChange(from oldPhase: ScenePhase, to newPhase: ScenePhase) {
        switch newPhase {
        case .background:
            print("[App] Entered background")
            appDelegate.scheduleBackgroundSync()
        case .active:
            print("[App] Became active")
        case .inactive:
            print("[App] Became inactive")
        @unknown default:
            break
        }
    }
}
