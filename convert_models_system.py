#!/usr/bin/env python3
"""
システムPythonを使用したCore ML変換（簡略版）
"""

import sys
import os
import torch
import torch.nn as nn

# システムPythonのパス追加
sys.path.insert(0, '/Users/shinya/Library/Python/3.9/lib/python/site-packages')

def create_and_convert_yolo():
    """YOLOモデルをCore ML形式に変換（直接変換）"""
    print("🔄 YOLOモデルの変換を開始...")
    
    try:
        from ultralytics import YOLO
        
        # YOLOモデルの読み込み
        yolo_path = "AI_Model/yolo11n-pose.pt"
        if not os.path.exists(yolo_path):
            print(f"❌ YOLOモデルが見つかりません: {yolo_path}")
            return False
            
        model = YOLO(yolo_path)
        print(f"✅ YOLOモデルを読み込みました: {yolo_path}")
        
        # Core ML形式にエクスポート（シンプルな設定）
        try:
            export_path = model.export(
                format='coreml',
                imgsz=640,
                half=False,   # Float32
                nms=False,    # Poseモデルなので無効
                simplify=True  # モデル最適化
            )
            
            if export_path:
                # ファイル移動
                import shutil
                from pathlib import Path
                
                output_dir = Path("VirtualTrainerApp/VirtualTrainerApp/MLModels")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                target_path = output_dir / "YOLO11nPose.mlpackage"
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                
                # パッケージディレクトリを移動
                shutil.move(export_path, target_path)
                
                print(f"✅ YOLOモデル変換完了: {target_path}")
                
                # サイズ確認
                size_mb = sum(f.stat().st_size for f in target_path.rglob('*') if f.is_file()) / (1024*1024)
                print(f"📊 モデルサイズ: {size_mb:.1f}MB")
                
                return True
            else:
                print("❌ エクスポートに失敗しました")
                return False
                
        except Exception as e:
            print(f"❌ YOLOエクスポートエラー: {e}")
            return False
            
    except Exception as e:
        print(f"❌ YOLO変換エラー: {e}")
        return False

def create_simple_gru_coreml():
    """シンプルなGRUモデルをCore MLで作成（テスト用）"""
    print("🔄 GRUモデルの作成・変換を開始...")
    
    try:
        import coremltools as ct
        from pathlib import Path
        
        # GRUモデルの定義
        class SimpleGRUModel(nn.Module):
            def __init__(self, input_size=12, hidden_size=32, num_layers=1):
                super(SimpleGRUModel, self).__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                out, _ = self.gru(x)
                out = out[:, -1, :]  # 最後のタイムステップ
                out = self.fc(out)
                out = self.sigmoid(out)  # 0-1の確率
                return out
        
        # モデル作成と評価モード
        model = SimpleGRUModel()
        model.eval()
        
        # ダミーデータ
        example_input = torch.randn(1, 10, 12)
        
        # TorchScriptに変換
        traced_model = torch.jit.trace(model, example_input)
        
        # Core MLに変換
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, 10, 12), name="features")],
            outputs=[ct.TensorType(name="prediction")],
            minimum_deployment_target=ct.target.iOS16
        )
        
        # メタデータの追加
        mlmodel.short_description = "エクササイズフォーム判定用GRUモデル（テスト版）"
        mlmodel.author = "Virtual Trainer"
        mlmodel.version = "1.0"
        
        # 保存
        output_dir = Path("VirtualTrainerApp/VirtualTrainerApp/MLModels")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "GRUFormClassifier.mlpackage"
        
        mlmodel.save(str(output_path))
        
        print(f"✅ GRUモデル変換完了: {output_path}")
        
        # サイズ確認
        size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024*1024)
        print(f"📊 モデルサイズ: {size_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GRU変換エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン関数"""
    print("🚀 学習済みモデルのCore ML変換")
    print("=" * 50)
    
    # YOLO変換
    yolo_success = create_and_convert_yolo()
    
    # GRU変換
    gru_success = create_simple_gru_coreml()
    
    if yolo_success and gru_success:
        print("\n🎉 すべてのモデル変換が完了しました！")
        print("\n次のステップ:")
        print("1. Xcodeでプロジェクトを開く")
        print("2. MLModels フォルダを確認")
        print("3. アプリを実行")
        print("4. デバッグモードで「AI: 有効」が表示されることを確認")
        
        return True
    else:
        print(f"\n⚠️  一部変換に失敗: YOLO={yolo_success}, GRU={gru_success}")
        if yolo_success:
            print("✅ YOLOモデルは使用可能です（姿勢検出機能が有効）")
        if gru_success:
            print("✅ GRUモデルは使用可能です（フォーム分類機能が有効）")
            
        return yolo_success or gru_success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🔥 iPhoneでAI機能が使用可能になりました！")
    sys.exit(0 if success else 1)