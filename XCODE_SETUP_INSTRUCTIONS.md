# Notification Service Extension セットアップ手順

Communication Notificationsを有効化するため、Xcodeでの手動設定が必要です。

## 1. Notification Service Extensionターゲットの作成

1. Xcodeでプロジェクトを開く
2. **File** → **New** → **Target** を選択
3. **Notification Service Extension** を選択
4. 以下の設定を入力:
   - **Product Name**: `NotificationServiceExtension`
   - **Team**: あなたの開発者アカウント
   - **Bundle Identifier**: `com.yourcompany.VirtualTrainerApp.NotificationServiceExtension`
5. **Finish** をクリック
6. 「Activate "NotificationServiceExtension" scheme?」と聞かれたら **Cancel** を選択（メインアプリのスキームを維持）

## 2. NotificationService.swiftファイルの置き換え

1. Xcode左側のプロジェクトナビゲータで、自動生成された `NotificationServiceExtension/NotificationService.swift` を削除
2. Finderで `/Users/shinya/Documents/Virtual_Trainer/VirtualTrainerApp/NotificationServiceExtension/NotificationService.swift` を見つける
3. このファイルをXcodeの `NotificationServiceExtension` グループにドラッグ＆ドロップ
4. **Target Membership** で `NotificationServiceExtension` にチェックが入っていることを確認

## 3. Info.plistファイルの置き換え

1. Xcode左側のプロジェクトナビゲータで、自動生成された `NotificationServiceExtension/Info.plist` を削除
2. Finderで `/Users/shinya/Documents/Virtual_Trainer/VirtualTrainerApp/NotificationServiceExtension/Info.plist` を見つける
3. このファイルをXcodeの `NotificationServiceExtension` グループにドラッグ＆ドロップ

## 4. Entitlementsファイルの設定

1. プロジェクトナビゲータで `NotificationServiceExtension` ターゲットを選択
2. **Signing & Capabilities** タブを開く
3. **Code Signing Entitlements** に以下のパスを設定:
   ```
   NotificationServiceExtension/NotificationServiceExtension.entitlements
   ```

## 5. App Groups Capabilityの追加

### メインアプリ（VirtualTrainerApp）:

1. プロジェクトナビゲータで **VirtualTrainerApp** ターゲットを選択
2. **Signing & Capabilities** タブを開く
3. **+ Capability** ボタンをクリック
4. **App Groups** を追加
5. **+** ボタンをクリックして以下のグループIDを追加:
   ```
   group.com.yourcompany.VirtualTrainer
   ```

### NotificationServiceExtension:

1. プロジェクトナビゲータで **NotificationServiceExtension** ターゲットを選択
2. **Signing & Capabilities** タブを開く
3. **+ Capability** ボタンをクリック
4. **App Groups** を追加
5. **+** ボタンをクリックして以下のグループIDを追加（メインアプリと同じ）:
   ```
   group.com.yourcompany.VirtualTrainer
   ```

## 6. Communication Notifications Capabilityの追加（メインアプリのみ）

1. プロジェクトナビゲータで **VirtualTrainerApp** ターゲットを選択
2. **Signing & Capabilities** タブを開く
3. **+ Capability** ボタンをクリック
4. **Communication Notifications** を追加

## 7. モデルファイルの共有設定

以下のファイルを `NotificationServiceExtension` ターゲットにも含める必要があります:

1. `VirtualTrainerApp/Models/OshiTrainer.swift`
   - Xcodeでファイルを選択
   - 右側の **File Inspector** を開く
   - **Target Membership** セクションで `NotificationServiceExtension` にチェック

2. `VirtualTrainerApp/Models/VoiceCharacter.swift`
   - 同様に **Target Membership** で `NotificationServiceExtension` にチェック

3. `VirtualTrainerApp/Utilities/UserDefaultsKeys.swift`
   - 同様に **Target Membership** で `NotificationServiceExtension` にチェック

## 8. ビルドとテスト

1. Xcodeで **Product** → **Clean Build Folder** (Shift + Cmd + K)
2. **Product** → **Build** (Cmd + B) を実行
3. ビルドエラーがないことを確認
4. 実機またはシミュレータでアプリを実行
5. デバッグダッシュボードからテスト通知を送信して動作確認

## トラブルシューティング

### ビルドエラー「No such module 'Intents'」
- NotificationServiceExtensionターゲットのDeployment Targetが iOS 15.0以上になっていることを確認

### 画像が表示されない
- `Resources/Image/OshinoAi/normal.png` が正しく配置されていることを確認
- NotificationServiceExtensionターゲットの **Build Phases** → **Copy Bundle Resources** に画像が含まれていることを確認

### App Group共有が動作しない
- メインアプリとExtensionの両方で同じApp Group IDが設定されていることを確認
- 開発者アカウントでApp Groupが有効化されていることを確認

---

完了後、動作確認してフィードバックをお願いします。
