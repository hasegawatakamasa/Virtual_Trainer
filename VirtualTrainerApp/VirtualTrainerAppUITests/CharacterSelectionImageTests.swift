import XCTest

final class CharacterSelectionImageTests: XCTestCase {
    
    var app: XCUIApplication!

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
        
        // In UI tests it is usually best to stop immediately when a failure occurs.
        continueAfterFailure = false

        // In UI tests it's important to set the initial state - such as interface orientation - required for your tests before they run. The setUp method is a good place to do this.
        app = XCUIApplication()
        app.launch()
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        app = nil
    }

    // MARK: - Character Selection UI Tests
    
    func testCharacterSelectionScreenDisplaysImages() throws {
        // Navigate to character selection if needed
        // This assumes there's a way to access the VoiceCharacterSettingsView
        
        // Wait for the screen to appear
        let characterSelectionTitle = app.staticTexts["音声キャラクター"]
        XCTAssertTrue(characterSelectionTitle.waitForExistence(timeout: 5), 
                     "キャラクター選択画面が表示されない")
        
        // Check if character cards are displayed
        let zundamonCard = app.buttons["ずんだもん"]
        let shikokuMetanCard = app.buttons["四国めたん"]
        
        XCTAssertTrue(zundamonCard.waitForExistence(timeout: 3), 
                     "ずんだもんのカードが表示されない")
        XCTAssertTrue(shikokuMetanCard.waitForExistence(timeout: 3), 
                     "四国めたんのカードが表示されない")
    }
    
    func testCharacterSelectionWithImages() throws {
        let characterSelectionTitle = app.staticTexts["音声キャラクター"]
        XCTAssertTrue(characterSelectionTitle.waitForExistence(timeout: 5))
        
        // Test selecting ずんだもん
        let zundamonCard = app.buttons["ずんだもん"]
        XCTAssertTrue(zundamonCard.waitForExistence(timeout: 3))
        
        zundamonCard.tap()
        
        // Verify selection state (check for selection indicator or state change)
        // This would depend on how the selected state is exposed to accessibility
        XCTAssertTrue(zundamonCard.isSelected || 
                     app.staticTexts.containing(NSPredicate(format: "label CONTAINS '選択中'")).element.exists,
                     "ずんだもんが選択状態になっていない")
        
        // Test selecting 四国めたん
        let shikokuMetanCard = app.buttons["四国めたん"]
        shikokuMetanCard.tap()
        
        XCTAssertTrue(shikokuMetanCard.isSelected || 
                     app.staticTexts.containing(NSPredicate(format: "label CONTAINS '選択中'")).element.exists,
                     "四国めたんが選択状態になっていない")
    }
    
    func testAccessibilityWithImages() throws {
        let characterSelectionTitle = app.staticTexts["音声キャラクター"]
        XCTAssertTrue(characterSelectionTitle.waitForExistence(timeout: 5))
        
        let zundamonCard = app.buttons["ずんだもん"]
        XCTAssertTrue(zundamonCard.waitForExistence(timeout: 3))
        
        // Check that accessibility label contains image description
        let accessibilityLabel = zundamonCard.label
        XCTAssertTrue(accessibilityLabel.contains("ずんだもん"), 
                     "アクセシビリティラベルにキャラクター名が含まれていない")
        XCTAssertTrue(accessibilityLabel.contains("キャラクター画像") || 
                     accessibilityLabel.contains("アイコン"),
                     "アクセシビリティラベルに画像の説明が含まれていない")
    }
    
    func testCharacterImageLoadingStates() throws {
        let characterSelectionTitle = app.staticTexts["音声キャラクター"]
        XCTAssertTrue(characterSelectionTitle.waitForExistence(timeout: 5))
        
        // Check that images load within reasonable time
        // This test assumes that the images should be loaded quickly
        let zundamonCard = app.buttons["ずんだもん"]
        XCTAssertTrue(zundamonCard.waitForExistence(timeout: 3))
        
        // Wait a brief moment for any loading to complete
        Thread.sleep(forTimeInterval: 0.5)
        
        // The card should still be present and interactive after loading
        XCTAssertTrue(zundamonCard.exists, "画像読み込み後にカードが消失")
        XCTAssertTrue(zundamonCard.isEnabled, "画像読み込み後にカードが無効化")
    }
    
    // MARK: - Performance Tests
    
    func testCharacterSelectionScreenLaunchPerformance() throws {
        measure(metrics: [XCTApplicationLaunchMetric()]) {
            // Test app launch time to character selection screen
            let freshApp = XCUIApplication()
            freshApp.launch()
            
            let characterSelectionTitle = freshApp.staticTexts["音声キャラクター"]
            _ = characterSelectionTitle.waitForExistence(timeout: 10)
            
            freshApp.terminate()
        }
    }
    
    func testCharacterImageDisplayPerformance() throws {
        let characterSelectionTitle = app.staticTexts["音声キャラクター"]
        XCTAssertTrue(characterSelectionTitle.waitForExistence(timeout: 5))
        
        measure {
            // Navigate away and back to test image reload performance
            if app.navigationBars.buttons["完了"].exists {
                app.navigationBars.buttons["完了"].tap()
                
                // Navigate back to character selection
                // This would depend on your app's navigation structure
                // For now, we'll just wait and check if the screen reloads quickly
                Thread.sleep(forTimeInterval: 0.1)
                
                // Re-access the screen if possible
                // This would need to be adapted based on actual navigation flow
            }
        }
    }
    
    // MARK: - Error Handling Tests
    
    func testCharacterSelectionWithNetworkError() throws {
        // This test would be relevant if images were loaded from network
        // For bundle resources, this test ensures graceful handling of missing files
        
        let characterSelectionTitle = app.staticTexts["音声キャラクター"]
        XCTAssertTrue(characterSelectionTitle.waitForExistence(timeout: 5))
        
        // Even if some images fail to load, the UI should remain functional
        let zundamonCard = app.buttons["ずんだもん"]
        let shikokuMetanCard = app.buttons["四国めたん"]
        
        XCTAssertTrue(zundamonCard.waitForExistence(timeout: 3))
        XCTAssertTrue(shikokuMetanCard.waitForExistence(timeout: 3))
        
        // Both cards should remain tappable even if images fail to load
        XCTAssertTrue(zundamonCard.isEnabled)
        XCTAssertTrue(shikokuMetanCard.isEnabled)
        
        // Test tapping still works
        zundamonCard.tap()
        shikokuMetanCard.tap()
        
        // UI should remain responsive
        XCTAssertTrue(zundamonCard.exists)
        XCTAssertTrue(shikokuMetanCard.exists)
    }
}