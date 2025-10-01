# Claude Code Spec-Driven Development

Kiro-style Spec Driven Development implementation using claude code slash commands, hooks and agents.

## Project Context

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`
- Commands: `.claude/commands/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context  
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- `ios-exercise-form-detection`: AI_Model/main.pyのフォーム検知＋回数カウント機能をiPhoneアプリでも実行可能にする
- `exercise-selection-screen`: ユーザー体験改善のためのトレーニング種目選択画面の新規実装（現在はオーバーヘッドプレスのみ、将来の拡張対応）
- `speed-feedback-enhancement`: オーバーヘッドプレス動作速度に応じたずんだもん音声フィードバック機能（速すぎる警告・応援音声）
- `voice-character-shikoku-metan-integration`: 四国めたん音声キャラクター追加と設定画面での音声キャラ選択機能（回数カウント、フォームエラー、速度フィードバック対応）
- `camera-session-bug-fixes`: カメラビューからの退出時にカメラセッション・音声フィードバックが継続する問題とUI改善の修正
- `training-records-oshi-system`: トレーニング記録機能の追加と推し要素の統合（セッション記録・進捗追跡・オタク文化特化エンゲージメント機能）
- `character-image-selection-enhancement`: キャラクター選択画面への画像表示機能追加（ずんだもん画像表示による視覚的推しキャラクター選択体験の向上）
- `60-second-timer-session`: トレーニングセッションの終了方式を手動（バツボタン）から60秒タイマー式に変更し、自動終了・リザルト画面遷移機能を実装
- `oshi-trainer-selection-system`: 音声キャラクター画面を推しトレーナー選択画面に変更し、デフォルトトレーナー「推乃 藍」の実装と四国めたんボイス選択機能の追加
- `exercise-selection-ui-improvement`: エクササイズ選択画面のUI改善（種目追加・説明・ユーザビリティ向上）
- `google-calendar-training-notifications`: Googleカレンダー連携による推しトレーナーからの通知機能（隙間時間検知・トレーニング促進通知）
- `debug-notifications-calendar-visibility`: 開発・デバッグ用の通知とカレンダー可視化機能（テスト通知送信・カレンダー取得状況確認・予約通知一覧表示）
- `line-style-trainer-notification-api`: Communication Notifications APIを使用したLINEライクな推しトレーナー通知表示（左上にトレーナー画像、右下にアプリアイコン）
- Check `.kiro/specs/` for active specifications
- Use `/kiro:spec-status [feature-name]` to check progress

## Development Guidelines
- Think in English, but generate responses in Japanese (思考は英語、回答の生成は日本語で行うように)

## Workflow

### Phase 0: Steering (Optional)
`/kiro:steering` - Create/update steering documents
`/kiro:steering-custom` - Create custom steering for specialized contexts

**Note**: Optional for new features or small additions. Can proceed directly to spec-init.

### Phase 1: Specification Creation
1. `/kiro:spec-init [detailed description]` - Initialize spec with detailed project description
2. `/kiro:spec-requirements [feature]` - Generate requirements document
3. `/kiro:spec-design [feature]` - Interactive: "requirements.mdをレビューしましたか？ [y/N]"
4. `/kiro:spec-tasks [feature]` - Interactive: Confirms both requirements and design review

### Phase 2: Progress Tracking
`/kiro:spec-status [feature]` - Check current progress and phases

## Development Rules
1. **Consider steering**: Run `/kiro:steering` before major development (optional for new features)
2. **Follow 3-phase approval workflow**: Requirements → Design → Tasks → Implementation
3. **Approval required**: Each phase requires human review (interactive prompt or manual)
4. **No skipping phases**: Design requires approved requirements; Tasks require approved design
5. **Update task status**: Mark tasks as completed when working on them
6. **Keep steering current**: Run `/kiro:steering` after significant changes
7. **Check spec compliance**: Use `/kiro:spec-status` to verify alignment

## Steering Configuration

### Current Steering Files
Managed by `/kiro:steering` command. Updates here reflect command changes.

### Active Steering Files
- `product.md`: Always included - Product context and business objectives
- `tech.md`: Always included - Technology stack and architectural decisions
- `structure.md`: Always included - File organization and code patterns

### Custom Steering Files
<!-- Added by /kiro:steering-custom command -->
<!-- Format: 
- `filename.md`: Mode - Pattern(s) - Description
  Mode: Always|Conditional|Manual
  Pattern: File patterns for Conditional mode
-->

### Inclusion Modes
- **Always**: Loaded in every interaction (default)
- **Conditional**: Loaded for specific file patterns (e.g., `"*.test.js"`)
- **Manual**: Reference with `@filename.md` syntax
