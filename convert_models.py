#!/usr/bin/env python3
"""
AI/MLモデルをCore ML形式に変換するスクリプト
- YOLO11n-poseモデルをCore MLに変換
- PyTorch GRUモデルをCore MLに変換
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def convert_yolo_model():
    """YOLO11n-poseモデルをCore ML形式に変換"""
    print("🔄 YOLO11n-poseモデルの変換を開始...")
    
    try:
        from ultralytics import YOLO
        
        # YOLOモデルの読み込み
        yolo_path = "AI_Model/yolo11n-pose.pt"
        if not os.path.exists(yolo_path):
            print(f"❌ YOLOモデルが見つかりません: {yolo_path}")
            return False
            
        model = YOLO(yolo_path)
        print(f"✅ YOLOモデルを読み込みました: {yolo_path}")
        
        # Core ML形式にエクスポート（pose modelはNMS不対応）
        export_path = model.export(
            format='coreml',
            imgsz=640,  # 入力画像サイズ
            half=True,   # Float16量子化
            nms=False,   # Poseモデルなのでnms=False
            int8=False   # INT8量子化は無効（精度のため）
        )
        
        # 出力ファイルをiOSプロジェクトに移動
        output_dir = Path("VirtualTrainerApp/VirtualTrainerApp/MLModels")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        target_path = output_dir / "YOLO11nPose.mlpackage"
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.move(export_path, target_path)
        
        print(f"✅ YOLOモデルをCore ML形式で保存: {target_path}")
        return True
        
    except Exception as e:
        print(f"❌ YOLO変換エラー: {e}")
        return False

def convert_gru_model():
    """PyTorch GRUモデルをCore ML形式に変換"""
    print("🔄 GRUモデルの変換を開始...")
    
    try:
        import coremltools as ct
        import sys
        
        # 量子化エンジンの設定（macOS用）
        if sys.platform == "darwin":
            torch.backends.quantized.engine = 'qnnpack'
        
        # GRUモデルの読み込み
        gru_path = "AI_Model/best_gru_model_v7_quantized.pth"
        if not os.path.exists(gru_path):
            print(f"❌ GRUモデルが見つかりません: {gru_path}")
            return False
            
        # モデル定義（main.pyから複製）
        class GRUModel(torch.nn.Module):
            def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout=0.2):
                super(GRUModel, self).__init__()
                self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.dropout = torch.nn.Dropout(dropout)
                self.fc = torch.nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                out, _ = self.gru(x)
                out = out[:, -1, :]  # 最後のタイムステップ
                out = self.dropout(out)
                out = self.fc(out)
                return out
        
        # モデルの読み込み（量子化モデルに対応）
        try:
            gru_model = torch.load(gru_path, map_location='cpu', weights_only=False)
        except RuntimeError as e:
            if "qengine" in str(e):
                print("⚠️  量子化モデルの読み込みに問題があるため、非量子化版を作成します")
                # 新しいモデルを作成して重みをロードしようとする
                gru_model = GRUModel()
                try:
                    state_dict = torch.load(gru_path, map_location='cpu', weights_only=True)
                    gru_model.load_state_dict(state_dict)
                except:
                    print("❌ モデルの重みをロードできません。元のPythonスクリプトを使用して非量子化モデルを作成してください")
                    return False
            else:
                raise e
                
        gru_model.eval()
        print(f"✅ GRUモデルを読み込みました: {gru_path}")
        
        # TorchScriptに変換
        example_input = torch.randn(1, 10, 12)  # バッチサイズ1, シーケンス長10, 特徴量12
        traced_model = torch.jit.script(gru_model)
        print("✅ TorchScriptに変換完了")
        
        # Core MLに変換
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, 10, 12), name="features")],
            outputs=[ct.TensorType(name="prediction")],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL  # Neural Engineを活用
        )
        
        # メタデータを追加
        mlmodel.short_description = "エクササイズフォーム判定用GRUモデル"
        mlmodel.author = "Virtual Trainer"
        mlmodel.license = "MIT"
        mlmodel.version = "1.0"
        
        # 入出力の説明を追加
        mlmodel.input_description["features"] = "正規化されたキーポイント特徴量 (10フレーム × 12次元)"
        mlmodel.output_description["prediction"] = "フォーム分類確率 (0: Normal, 1: Error)"
        
        # 出力ファイルの保存
        output_dir = Path("VirtualTrainerApp/VirtualTrainerApp/MLModels")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "GRUFormClassifier.mlpackage"
        
        mlmodel.save(str(output_path))
        print(f"✅ GRUモデルをCore ML形式で保存: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ GRU変換エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_models():
    """変換されたモデルの検証"""
    print("🔍 変換されたモデルを検証中...")
    
    models_dir = Path("VirtualTrainerApp/VirtualTrainerApp/MLModels")
    
    yolo_path = models_dir / "YOLO11nPose.mlpackage"
    gru_path = models_dir / "GRUFormClassifier.mlpackage"
    
    success = True
    
    if yolo_path.exists():
        size_mb = sum(f.stat().st_size for f in yolo_path.rglob('*') if f.is_file()) / (1024*1024)
        print(f"✅ YOLO11n-pose: {size_mb:.1f}MB")
    else:
        print("❌ YOLO11n-poseモデルが見つかりません")
        success = False
        
    if gru_path.exists():
        size_mb = sum(f.stat().st_size for f in gru_path.rglob('*') if f.is_file()) / (1024*1024)
        print(f"✅ GRUFormClassifier: {size_mb:.1f}MB")
    else:
        print("❌ GRUFormClassifierモデルが見つかりません")
        success = False
    
    return success

def main():
    """メイン関数"""
    print("🚀 AI/MLモデルのCore ML変換を開始します")
    print("=" * 50)
    
    # 必要なライブラリの確認
    required_libs = ['torch', 'ultralytics', 'coremltools']
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✅ {lib} インストール済み")
        except ImportError:
            missing_libs.append(lib)
            print(f"❌ {lib} が見つかりません")
    
    if missing_libs:
        print(f"\n⚠️  必要なライブラリが不足しています:")
        for lib in missing_libs:
            if lib == 'ultralytics':
                print(f"   pip install {lib}")
            elif lib == 'coremltools':
                print(f"   pip install {lib}")
        print()
        return False
    
    # モデル変換の実行
    yolo_success = convert_yolo_model()
    gru_success = convert_gru_model()
    
    if yolo_success and gru_success:
        print("\n" + "=" * 50)
        if verify_models():
            print("🎉 すべてのモデル変換が正常に完了しました！")
            print("\n次のステップ:")
            print("1. Xcodeプロジェクトを開く")
            print("2. MLModels フォルダをプロジェクトに追加")
            print("3. モデルがターゲットに含まれていることを確認")
            return True
        else:
            print("⚠️  一部のモデル変換に問題があります")
            return False
    else:
        print("\n❌ モデル変換が失敗しました")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)