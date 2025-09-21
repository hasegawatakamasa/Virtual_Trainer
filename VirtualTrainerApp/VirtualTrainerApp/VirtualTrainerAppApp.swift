import SwiftUI

@main
struct VirtualTrainerAppApp: App {
    
    init() {
        // UserDefaultsのデフォルト値を設定
        UserDefaultsKeys.registerDefaults()
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
