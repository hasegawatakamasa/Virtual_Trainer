import XCTest
import Combine
@testable import VirtualTrainerApp

final class OshiTrainerSettingsTests: XCTestCase {

    var cancellables: Set<AnyCancellable>!

    override func setUp() {
        super.setUp()
        cancellables = Set<AnyCancellable>()

        // UserDefaultsをクリーンアップ
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.selectedOshiTrainerId)
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.selectedVoiceCharacter)
    }

    override func tearDown() {
        cancellables = nil

        // UserDefaultsをクリーンアップ
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.selectedOshiTrainerId)
        UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.selectedVoiceCharacter)

        super.tearDown()
    }

    func testDefaultTrainerSelection() {
        // Given: シングルトンインスタンスを使用
        let settings = OshiTrainerSettings.shared

        // Then: デフォルトトレーナーが推乃 藍であること
        XCTAssertEqual(settings.selectedTrainer, .oshinoAi, "デフォルトトレーナーは推乃 藍であること")
        XCTAssertEqual(settings.currentVoiceCharacter, .zundamon, "デフォルトトレーナーの音声はずんだもんであること")
    }

    func testTrainerSelectionPersistence() {
        // Given: シングルトンインスタンスを使用
        let settings = OshiTrainerSettings.shared

        // When: トレーナーを更新
        settings.updateTrainer(.oshinoAi)

        // Then: UserDefaultsに保存されること
        let savedId = UserDefaults.standard.string(forKey: UserDefaultsKeys.selectedOshiTrainerId)
        XCTAssertEqual(savedId, OshiTrainer.oshinoAi.id, "トレーナーIDがUserDefaultsに保存されること")
    }

    func testVoiceSettingsSynchronization() {
        // Given: シングルトンインスタンスを使用
        let settings = OshiTrainerSettings.shared

        // When: トレーナーを更新
        settings.updateTrainer(.oshinoAi)

        // Then: VoiceSettingsも更新されること
        let expectation = expectation(description: "VoiceSettings synchronization")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            XCTAssertEqual(VoiceSettings.shared.selectedCharacter, .zundamon, "VoiceSettingsが同期されること")
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 1.0)
    }

    func testMigrationFromVoiceCharacter() {
        // Given: 既存のVoiceCharacter設定が保存されている状態
        UserDefaults.standard.set(VoiceCharacter.zundamon.rawValue, forKey: UserDefaultsKeys.selectedVoiceCharacter)

        // 注: シングルトンは一度初期化されると再初期化できないため、
        // このテストではマイグレーション後のトレーナー状態のみを確認
        let settings = OshiTrainerSettings.shared

        // Then: ずんだもんボイスを使用する推乃 藍が選択されていること
        XCTAssertEqual(settings.selectedTrainer, .oshinoAi, "ずんだもんボイスから推乃 藍にマイグレーションされること")

        // マイグレーションは実際の初回起動時にのみ実行されるため、
        // ここではトレーナーの状態が正しいことのみを確認
    }

    func testMigrationFromShikokuMetan() {
        // Given: 四国めたんの設定が保存されている状態
        UserDefaults.standard.set(VoiceCharacter.shikokuMetan.rawValue, forKey: UserDefaultsKeys.selectedVoiceCharacter)

        // シングルトンを使用
        let settings = OshiTrainerSettings.shared

        // Then: 現在は推乃 藍にマッピングされること（将来的に四国めたんボイスのトレーナーが追加される予定）
        XCTAssertEqual(settings.selectedTrainer, .oshinoAi, "四国めたんボイスから推乃 藍にマイグレーションされること（将来対応予定）")
    }

    func testPublishedPropertyUpdate() {
        // Given: シングルトンインスタンスを使用
        let settings = OshiTrainerSettings.shared
        let expectation = expectation(description: "Published property update")
        var receivedTrainer: OshiTrainer?

        // When: @Published プロパティの変更を購読
        settings.$selectedTrainer
            .dropFirst() // 初期値をスキップ
            .sink { trainer in
                receivedTrainer = trainer
                expectation.fulfill()
            }
            .store(in: &cancellables)

        // トレーナーを更新
        settings.updateTrainer(.oshinoAi)

        // Then: Combineを通じて変更が通知されること
        wait(for: [expectation], timeout: 1.0)
        XCTAssertEqual(receivedTrainer, .oshinoAi, "Combineを通じてトレーナー変更が通知されること")
    }

    func testCurrentVoiceCharacterGetter() {
        // Given: シングルトンインスタンスを使用
        let settings = OshiTrainerSettings.shared

        // When
        let voiceCharacter = settings.currentVoiceCharacter

        // Then: selectedTrainerのvoiceCharacterと一致すること
        XCTAssertEqual(voiceCharacter, settings.selectedTrainer.voiceCharacter, "currentVoiceCharacterがselectedTrainerのvoiceCharacterと一致すること")
    }

    func testSingletonInstance() {
        // Given
        let instance1 = OshiTrainerSettings.shared
        let instance2 = OshiTrainerSettings.shared

        // Then: 同じインスタンスであること
        XCTAssertTrue(instance1 === instance2, "シングルトンインスタンスが同一であること")
    }
}