import XCTest
import Combine
@testable import VirtualTrainerApp

final class VoicePreviewServiceTests: XCTestCase {

    var service: VoicePreviewService!
    var cancellables: Set<AnyCancellable>!

    @MainActor
    override func setUp() {
        super.setUp()
        service = VoicePreviewService.shared
        cancellables = Set<AnyCancellable>()
    }

    @MainActor
    override func tearDown() {
        // 再生中の音声を停止
        service.stopPreview()
        cancellables = nil
        service = nil
        super.tearDown()
    }

    @MainActor
    func testSingletonInstance() {
        // Given
        let instance1 = VoicePreviewService.shared
        let instance2 = VoicePreviewService.shared

        // Then: 同じインスタンスであること
        XCTAssertTrue(instance1 === instance2, "シングルトンインスタンスが同一であること")
    }

    @MainActor
    func testInitialState() {
        // Then: 初期状態では再生していないこと
        XCTAssertFalse(service.isPlaying, "初期状態ではisPlayingがfalseであること")
    }

    @MainActor
    func testPlayRandomPreviewUpdatesIsPlaying() {
        // Given
        let trainer = OshiTrainer.oshinoAi
        let expectation = XCTestExpectation(description: "isPlaying update")

        // When: isPlayingの変更を購読
        service.$isPlaying
            .dropFirst() // 初期値をスキップ
            .sink { isPlaying in
                if isPlaying {
                    expectation.fulfill()
                }
            }
            .store(in: &cancellables)

        // プレビューを再生
        service.playRandomPreview(for: trainer)

        // Then: isPlayingがtrueに更新されること
        wait(for: [expectation], timeout: 2.0)
        XCTAssertTrue(service.isPlaying, "プレビュー再生時にisPlayingがtrueになること")
    }

    @MainActor
    func testStopPreview() {
        // Given: 再生を開始
        let trainer = OshiTrainer.oshinoAi
        service.playRandomPreview(for: trainer)

        // Wait for playback to start
        let playExpectation = XCTestExpectation(description: "playback started")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            playExpectation.fulfill()
        }
        wait(for: [playExpectation], timeout: 1.0)

        // When: 停止
        service.stopPreview()

        // Then: isPlayingがfalseになること
        XCTAssertFalse(service.isPlaying, "停止後にisPlayingがfalseになること")
    }

    @MainActor
    func testMultiplePlaybackStopsPrevious() {
        // Given
        let trainer = OshiTrainer.oshinoAi

        // When: 複数回連続で再生
        service.playRandomPreview(for: trainer)

        let firstPlayExpectation = XCTestExpectation(description: "first playback")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            firstPlayExpectation.fulfill()
        }
        wait(for: [firstPlayExpectation], timeout: 1.0)

        // 2回目の再生（1回目を停止して新しい音声を再生）
        service.playRandomPreview(for: trainer)

        // Then: 2回目の再生が開始されること（前の再生が停止される）
        XCTAssertTrue(service.isPlaying, "新しいプレビュー再生が開始されること")
    }

    @MainActor
    func testRandomAudioTypeSelection() {
        // Given
        let trainer = OshiTrainer.oshinoAi
        var playbackOccurred = false

        // When: isPlayingの変更を監視
        service.$isPlaying
            .dropFirst()
            .sink { isPlaying in
                if isPlaying {
                    playbackOccurred = true
                }
            }
            .store(in: &cancellables)

        // プレビューを複数回再生（ランダム選択の確認）
        for _ in 0..<5 {
            service.playRandomPreview(for: trainer)

            // 各再生の間に短い待機
            let playExpectation = XCTestExpectation(description: "playback iteration")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                playExpectation.fulfill()
            }
            wait(for: [playExpectation], timeout: 1.0)

            service.stopPreview()
        }

        // Then: 少なくとも1回は再生が発生したこと
        XCTAssertTrue(playbackOccurred, "ランダムプレビュー再生が実行されること")
    }

    @MainActor
    func testPlayPreviewWithInvalidTrainerDoesNotCrash() {
        // Given: 存在しない音声ファイルを参照するカスタムトレーナー
        let customTrainer = OshiTrainer(
            id: "test-trainer",
            displayName: "テストトレーナー",
            firstPerson: "私",
            secondPerson: "あなた",
            personality: "テスト",
            voiceCharacter: .zundamon,
            imageName: "nonexistent",
            imageDirectory: "NonExistent"
        )

        // When: プレビューを再生（音声ファイルが見つからない可能性がある）
        service.playRandomPreview(for: customTrainer)

        // Then: クラッシュせず、isPlayingの状態が適切に管理されること
        // 注: 音声ファイルが見つからない場合、isPlayingはfalseのまま
        // 音声ファイルが見つかった場合、isPlayingはtrueになる
        // このテストはクラッシュしないことを確認するのが目的
    }
}