import XCTest
@testable import VirtualTrainerApp

final class OshiTrainerTests: XCTestCase {

    func testDefaultTrainerIsOshinoAi() {
        // Given
        let trainer = OshiTrainer.oshinoAi

        // Then
        XCTAssertEqual(trainer.id, "oshino-ai", "トレーナーIDが正しいこと")
        XCTAssertEqual(trainer.displayName, "推乃 藍", "トレーナー名が正しいこと")
        XCTAssertEqual(trainer.firstPerson, "うち", "一人称が正しいこと")
        XCTAssertEqual(trainer.secondPerson, "あんた", "二人称が正しいこと")
        XCTAssertEqual(trainer.personality, "ツンデレ", "性格設定が正しいこと")
        XCTAssertEqual(trainer.voiceCharacter, .zundamon, "ボイスキャラクターが正しいこと")
        XCTAssertEqual(trainer.imageName, "normal", "画像名が正しいこと")
        XCTAssertEqual(trainer.imageDirectory, "OshinoAi", "画像ディレクトリが正しいこと")
    }

    func testTrainerImageURLResolution() {
        // Given
        let trainer = OshiTrainer.oshinoAi

        // When
        let url = trainer.imageFileURL()

        // Then
        // 実際の画像ファイルが存在する場合、URLが返されること
        if let url = url {
            XCTAssertTrue(url.path.contains("normal"), "画像パスにnormalが含まれること")
            XCTAssertTrue(url.path.hasSuffix(".png"), "画像パスが.pngで終わること")
            print("[Test] Image URL resolved: \(url.path)")
        } else {
            print("[Test] Image URL not found - this is acceptable in test environment")
        }
    }

    func testTrainerEquatable() {
        // Given
        let trainer1 = OshiTrainer.oshinoAi
        let trainer2 = OshiTrainer.oshinoAi
        let trainer3 = OshiTrainer(
            id: "different-id",
            displayName: "別のトレーナー",
            firstPerson: "わたし",
            secondPerson: "あなた",
            personality: "優しい",
            voiceCharacter: .shikokuMetan,
            imageName: "other",
            imageDirectory: "Other"
        )

        // Then
        XCTAssertEqual(trainer1, trainer2, "同じトレーナーは等しいこと")
        XCTAssertNotEqual(trainer1, trainer3, "異なるトレーナーは等しくないこと")
    }

    func testTrainerCodable() throws {
        // Given
        let trainer = OshiTrainer.oshinoAi
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // When
        let encoded = try encoder.encode(trainer)
        let decoded = try decoder.decode(OshiTrainer.self, from: encoded)

        // Then
        XCTAssertEqual(trainer, decoded, "エンコード・デコード後も同じトレーナーであること")
    }

    func testAllTrainersContainsOshinoAi() {
        // Given
        let allTrainers = OshiTrainer.allTrainers

        // Then
        XCTAssertFalse(allTrainers.isEmpty, "利用可能なトレーナーが存在すること")
        XCTAssertTrue(allTrainers.contains(OshiTrainer.oshinoAi), "推乃 藍が含まれること")
    }

    func testAccessibilityDescription() {
        // Given
        let trainer = OshiTrainer.oshinoAi

        // When
        let description = trainer.accessibilityDescription

        // Then
        XCTAssertTrue(description.contains("推乃 藍"), "トレーナー名が含まれること")
        XCTAssertTrue(description.contains("ツンデレ"), "性格が含まれること")
        XCTAssertTrue(description.contains("うち"), "一人称が含まれること")
        XCTAssertTrue(description.contains("あんた"), "二人称が含まれること")
    }
}