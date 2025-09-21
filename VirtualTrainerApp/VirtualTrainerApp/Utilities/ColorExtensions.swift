import SwiftUI
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

extension Color {
    #if os(iOS)
    static let systemBackground = Color(UIColor.systemBackground)
    static let systemGroupedBackground = Color(UIColor.systemGroupedBackground)
    static let systemGray4 = Color(UIColor.systemGray4)
    static let systemGray5 = Color(UIColor.systemGray5)
    static let systemGray6 = Color(UIColor.systemGray6)
    #else
    static let systemBackground = Color(NSColor.windowBackgroundColor)
    static let systemGroupedBackground = Color(NSColor.controlBackgroundColor)
    static let systemGray4 = Color(NSColor.systemGray.withAlphaComponent(0.7))
    static let systemGray5 = Color(NSColor.systemGray.withAlphaComponent(0.6))
    static let systemGray6 = Color(NSColor.systemGray.withAlphaComponent(0.5))
    #endif
}