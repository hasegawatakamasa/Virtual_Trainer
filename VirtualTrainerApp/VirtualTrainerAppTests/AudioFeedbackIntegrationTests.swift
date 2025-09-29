import XCTest
import Combine
@testable import VirtualTrainerApp

/// AudioFeedbackService連携のIntegration Tests
final class AudioFeedbackIntegrationTests: XCTestCase {

    var oshiTrainerSettings: OshiTrainerSettings!
    var voiceSettings: VoiceSettings!
    var audioService: AudioFeedbackService!
    var cancellables: Set<AnyCancellable>!

    @MainActor
    override func setUp() {
        super.setUp()
        oshiTrainerSettings = OshiTrainerSettings.shared
        voiceSettings = VoiceSettings.shared
        audioService = AudioFeedbackService()
        cancellables = Set<AnyCancellable>()

        // UserDefaultsをクリーンアップ
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.selectedOshiTrainerId)
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.selectedVoiceCharacter)
    }

    @MainActor
    override func tearDown() {
        cancellables = nil
        audioService = nil
        voiceSettings = nil
        oshiTrainerSettings = nil

        // UserDefaultsをクリーンアップ
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.selectedOshiTrainerId)
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.selectedVoiceCharacter)

        super.tearDown()
    }

    @MainActor
    func testOshiTrainerVoiceCharacterIntegration() {
        // Given: 初期状態では推乃 藍（ずんだもんボイス）
        let initialTrainer = oshiTrainerSettings.selectedTrainer
        XCTAssertEqual(initialTrainer, .oshinoAi, "初期トレーナーは推乃 藍であること")

        // When: トレーナーを更新
        oshiTrainerSettings.updateTrainer(.oshinoAi)

        // Then: VoiceSettingsが同期されること
        let expectation = XCTestExpectation(description: "VoiceSettings synchronization")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            XCTAssertEqual(VoiceSettings.shared.selectedCharacter, .zundamon, "VoiceSettingsがずんだもんに同期されること")
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 2.0)
    }

    @MainActor
    func testOshiTrainerSettingsUpdatesVoiceSettings() {
        // Given: OshiTrainerSettingsのselectedTrainerを監視
        var receivedVoiceCharacter: VoiceCharacter?

        voiceSettings.$selectedCharacter
            .sink { character in
                receivedVoiceCharacter = character
            }
            .store(in: &cancellables)

        // When: トレーナーを推乃 藍に更新
        oshiTrainerSettings.updateTrainer(.oshinoAi)

        // Then: VoiceSettingsのselectedCharacterがずんだもんになること
        let expectation = XCTestExpectation(description: "Voice character update")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            XCTAssertEqual(receivedVoiceCharacter, .zundamon, "VoiceCharacterがずんだもんに更新されること")
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 2.0)
    }

    @MainActor
    func testCurrentVoiceCharacterGetter() {
        // Given: 推乃 藍を選択
        oshiTrainerSettings.updateTrainer(.oshinoAi)

        // When: currentVoiceCharacterを取得
        let voiceCharacter = oshiTrainerSettings.currentVoiceCharacter

        // Then: ずんだもんが返されること
        XCTAssertEqual(voiceCharacter, .zundamon, "currentVoiceCharacterがずんだもんであること")
    }

    @MainActor
    func testVoiceSettingsIntegrationWithAudioFeedbackService() {
        // Given: トレーナーを推乃 藍に設定
        oshiTrainerSettings.updateTrainer(.oshinoAi)

        // Wait for settings to propagate
        let expectation = XCTestExpectation(description: "Settings propagation")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            // Then: VoiceSettings経由でAudioFeedbackServiceが正しい音声キャラクターを使用
            let currentCharacter = VoiceSettings.shared.selectedCharacter
            XCTAssertEqual(currentCharacter, .zundamon, "AudioFeedbackServiceがずんだもんボイスを使用すること")

            // AudioFeedbackServiceが音声ファイルURLを正しく取得できることを確認
            if let audioURL = currentCharacter.audioFileURL(for: .formError) {
                XCTAssertTrue(audioURL.path.contains("zundamon"), "音声ファイルパスにずんだもんが含まれること")
            }

            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 2.0)
    }

    @MainActor
    func testMultipleTrainerUpdates() {
        // Given: 複数回のトレーナー更新を実行
        oshiTrainerSettings.updateTrainer(.oshinoAi)

        // Wait a bit
        let expectation1 = XCTestExpectation(description: "First update")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            expectation1.fulfill()
        }
        wait(for: [expectation1], timeout: 1.0)

        // When: 再度同じトレーナーを設定
        oshiTrainerSettings.updateTrainer(.oshinoAi)

        // Then: VoiceSettingsが一貫性を保つこと
        let expectation2 = XCTestExpectation(description: "Second update")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            XCTAssertEqual(VoiceSettings.shared.selectedCharacter, .zundamon, "複数回更新後もVoiceSettingsが一貫していること")
            expectation2.fulfill()
        }

        wait(for: [expectation2], timeout: 1.0)
    }
}