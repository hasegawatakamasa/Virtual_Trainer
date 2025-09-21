import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            ExerciseSelectionView()
        }
        .preferredColorScheme(.dark) // カメラアプリは通常ダークモード
    }
}

#Preview {
    ContentView()
}
