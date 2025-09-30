import XCTest
@testable import VirtualTrainerApp

final class ExerciseTypeTargetTests: XCTestCase {

    // MARK: - Target Duration Tests

    func testTargetDurationIsAlways60Seconds() {
        for exercise in ExerciseType.allCases {
            XCTAssertEqual(
                exercise.targetDuration,
                60,
                "\(exercise.displayName)の目標時間は60秒であるべき"
            )
        }
    }

    // MARK: - Target Reps Tests

    func testTargetRepsArePositiveOrNil() {
        for exercise in ExerciseType.allCases {
            if let reps = exercise.targetReps {
                XCTAssertGreaterThan(
                    reps,
                    0,
                    "\(exercise.displayName)の目標回数は正の整数であるべき"
                )
            }
        }
    }

    func testTargetRepsValues() {
        // 各種目の目標回数が正しいことを検証
        XCTAssertEqual(ExerciseType.overheadPress.targetReps, 10)
        XCTAssertEqual(ExerciseType.sideRaise.targetReps, 15)
        XCTAssertEqual(ExerciseType.squat.targetReps, 15)
        XCTAssertEqual(ExerciseType.pushUp.targetReps, 12)
    }

    // MARK: - Guidance Text Tests

    func testGuidanceTextIsNotEmpty() {
        for exercise in ExerciseType.allCases {
            XCTAssertFalse(
                exercise.guidanceText.isEmpty,
                "\(exercise.displayName)のガイダンステキストは空であるべきではない"
            )
        }
    }

    func testGuidanceTextContainsTargetInfo() {
        // ガイダンステキストに目標情報が含まれていることを検証
        XCTAssertTrue(ExerciseType.overheadPress.guidanceText.contains("10回"))
        XCTAssertTrue(ExerciseType.sideRaise.guidanceText.contains("15回"))
        XCTAssertTrue(ExerciseType.squat.guidanceText.contains("15回"))
        XCTAssertTrue(ExerciseType.pushUp.guidanceText.contains("12回"))
    }

    // MARK: - Target Display Text Tests

    func testTargetDisplayTextFormatIsCorrect() {
        // 回数ベースの種目
        let overheadPress = ExerciseType.overheadPress
        XCTAssertTrue(overheadPress.targetDisplayText.contains("10回"))
        XCTAssertTrue(overheadPress.targetDisplayText.contains("1分"))

        let sideRaise = ExerciseType.sideRaise
        XCTAssertTrue(sideRaise.targetDisplayText.contains("15回"))
        XCTAssertTrue(sideRaise.targetDisplayText.contains("1分"))
    }

    func testTargetDisplayTextAllExercises() {
        // 全種目の表示テキストフォーマットを検証
        for exercise in ExerciseType.allCases {
            let displayText = exercise.targetDisplayText
            XCTAssertTrue(
                displayText.contains("目標"),
                "\(exercise.displayName)の表示テキストには「目標」が含まれるべき"
            )
            XCTAssertTrue(
                displayText.contains("1分"),
                "\(exercise.displayName)の表示テキストには時間情報が含まれるべき"
            )
        }
    }

    // MARK: - Display Name Tests (省略防止)

    func testDisplayNameIsNotTruncated() {
        // 種目名が省略されていないことを確認
        for exercise in ExerciseType.allCases {
            let displayName = exercise.displayName
            XCTAssertFalse(displayName.isEmpty, "\(exercise)の表示名は空であるべきではない")
        }

        // 特に「オーバーヘッドプレス」が完全に保持されることを確認
        XCTAssertEqual(
            ExerciseType.overheadPress.displayName,
            "オーバーヘッドプレス",
            "オーバーヘッドプレスの名前は省略されるべきではない"
        )
    }

    // MARK: - Target Type Tests

    func testTargetTypeConsistency() {
        // 全種目が回数ベースであることを検証
        for exercise in ExerciseType.allCases {
            XCTAssertNotNil(exercise.targetReps, "\(exercise.displayName)は回数を持つべき")
            XCTAssertEqual(
                exercise.targetType,
                .reps,
                "\(exercise.displayName)は回数ベースであるべき"
            )
        }
    }

    // MARK: - Target Info Tests

    func testTargetInfoStructure() {
        let overheadPress = ExerciseType.overheadPress
        let targetInfo = overheadPress.targetInfo

        XCTAssertEqual(targetInfo.duration, 60)
        XCTAssertEqual(targetInfo.reps, 10)
        XCTAssertEqual(targetInfo.type, .reps)
        XCTAssertEqual(targetInfo.displayText, overheadPress.targetDisplayText)
        XCTAssertEqual(targetInfo.guidanceText, overheadPress.guidanceText)
    }

    func testTargetInfoForSideRaise() {
        let sideRaise = ExerciseType.sideRaise
        let targetInfo = sideRaise.targetInfo

        XCTAssertEqual(targetInfo.duration, 60)
        XCTAssertEqual(targetInfo.reps, 15)
        XCTAssertEqual(targetInfo.type, .reps)
        XCTAssertTrue(targetInfo.displayText.contains("15回"))
    }
}