import SwiftUI

struct ContentView: View {
    var body: some View {
        ExerciseTrainingView()
            .preferredColorScheme(.dark) // カメラアプリは通常ダークモード
    }
}

#Preview {
    ContentView()
}
