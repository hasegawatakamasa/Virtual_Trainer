import XCTest
import SwiftUI
@testable import VirtualTrainerApp

final class CharacterImageTests: XCTestCase {
    
    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    // MARK: - VoiceCharacter Image Properties Tests
    
    func testZundamonImageName() throws {
        let character = VoiceCharacter.zundamon
        XCTAssertEqual(character.imageName, "zundamon_1")
    }
    
    func testShikokuMetanImageName() throws {
        let character = VoiceCharacter.shikokuMetan
        XCTAssertEqual(character.imageName, "shikoku_metan_1")
    }
    
    func testZundamonHasImage() throws {
        let character = VoiceCharacter.zundamon
        // ずんだもんの画像は存在するはず
        XCTAssertTrue(character.hasImage, "ずんだもんの画像が見つからない")
    }
    
    func testZundamonImageFileURL() throws {
        let character = VoiceCharacter.zundamon
        let imageURL = character.imageFileURL()
        
        XCTAssertNotNil(imageURL, "ずんだもんの画像URLが取得できない")
        
        if let url = imageURL {
            // ファイルが実際に存在することを確認
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), 
                         "画像ファイルが実際には存在しない: \(url.path)")
        }
    }
    
    func testShikokuMetanImageFileURL() throws {
        let character = VoiceCharacter.shikokuMetan
        let imageURL = character.imageFileURL()
        
        // 四国めたんの画像は将来対応予定のため、現在は存在しなくても良い
        // URLが取得できない場合は適切にnilを返すことを確認
        if let url = imageURL {
            // URLが取得できた場合はファイルが実際に存在することを確認
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), 
                         "画像ファイルが実際には存在しない: \(url.path)")
        }
    }
    
    func testAccessibilityImageDescription() throws {
        let zundamon = VoiceCharacter.zundamon
        let shikokuMetan = VoiceCharacter.shikokuMetan
        
        XCTAssertFalse(zundamon.accessibilityImageDescription.isEmpty, 
                      "ずんだもんのアクセシビリティ説明が空")
        XCTAssertFalse(shikokuMetan.accessibilityImageDescription.isEmpty, 
                      "四国めたんのアクセシビリティ説明が空")
        
        XCTAssertTrue(zundamon.accessibilityImageDescription.contains("ずんだもん"), 
                     "ずんだもんの説明にキャラクター名が含まれていない")
        XCTAssertTrue(shikokuMetan.accessibilityImageDescription.contains("四国めたん"), 
                     "四国めたんの説明にキャラクター名が含まれていない")
    }
    
    // MARK: - CharacterImageError Tests
    
    func testCharacterImageErrorMessages() throws {
        let zundamon = VoiceCharacter.zundamon
        
        let imageNotFoundError = CharacterImageError.imageNotFound(character: zundamon)
        XCTAssertTrue(imageNotFoundError.localizedDescription.contains("ずんだもん"),
                     "エラーメッセージにキャラクター名が含まれていない")
        
        let invalidDataError = CharacterImageError.invalidImageData(character: zundamon)
        XCTAssertTrue(invalidDataError.localizedDescription.contains("ずんだもん"),
                     "エラーメッセージにキャラクター名が含まれていない")
        
        let bundleMissingError = CharacterImageError.bundleResourceMissing(path: "/test/path")
        XCTAssertTrue(bundleMissingError.localizedDescription.contains("/test/path"),
                     "エラーメッセージにパスが含まれていない")
    }
    
    // MARK: - ImageLoadResult Tests
    
    func testImageLoadResultSuccess() throws {
        // テスト用の1x1ピクセル画像を作成
        UIGraphicsBeginImageContext(CGSize(width: 1, height: 1))
        let context = UIGraphicsGetCurrentContext()
        context?.setFillColor(UIColor.red.cgColor)
        context?.fill(CGRect(x: 0, y: 0, width: 1, height: 1))
        let testImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        guard let image = testImage else {
            XCTFail("テスト用画像の作成に失敗")
            return
        }
        
        let result = ImageLoadResult.success(image)
        
        XCTAssertNotNil(result.image, "成功結果から画像が取得できない")
        XCTAssertNil(result.error, "成功結果でエラーが返された")
    }
    
    func testImageLoadResultFallback() throws {
        let error = CharacterImageError.memoryAllocationFailed
        let result = ImageLoadResult.fallback(error)
        
        XCTAssertNil(result.image, "エラー結果で画像が返された")
        XCTAssertNotNil(result.error, "エラー結果からエラーが取得できない")
    }
    
    // MARK: - Performance Tests
    
    func testImageURLPerformance() throws {
        let character = VoiceCharacter.zundamon
        
        measure {
            for _ in 0..<100 {
                _ = character.imageFileURL()
            }
        }
    }
    
    func testHasImagePerformance() throws {
        let character = VoiceCharacter.zundamon
        
        measure {
            for _ in 0..<100 {
                _ = character.hasImage
            }
        }
    }
}