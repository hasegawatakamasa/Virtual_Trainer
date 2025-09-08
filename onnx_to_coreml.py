#!/usr/bin/env python3
"""
ONNXモデルをCore ML形式に変換するスクリプト
"""

import os
import sys
from pathlib import Path

def convert_onnx_to_coreml():
    """ONNXモデルをCore ML形式に変換"""
    print("🔄 ONNXからCore MLへの変換を開始...")
    
    try:
        import coremltools as ct
        
        # YOLOモデルの変換
        print("📱 YOLOモデルの変換...")
        yolo_onnx_path = "AI_Model/yolo11n-pose.onnx"
        if os.path.exists(yolo_onnx_path):
            try:
                # ONNX to Core ML
                yolo_mlmodel = ct.convert(
                    yolo_onnx_path,
                    minimum_deployment_target=ct.target.iOS16,
                    compute_units=ct.ComputeUnit.ALL,
                    convert_to="mlprogram"
                )
                
                # メタデータの追加
                yolo_mlmodel.short_description = "YOLO11n-pose 姿勢検出モデル"
                yolo_mlmodel.author = "Ultralytics"
                yolo_mlmodel.license = "AGPL-3.0"
                yolo_mlmodel.version = "11.0"
                
                # 保存
                output_dir = Path("VirtualTrainerApp/VirtualTrainerApp/MLModels")
                output_dir.mkdir(parents=True, exist_ok=True)
                yolo_output_path = output_dir / "YOLO11nPose.mlpackage"
                
                yolo_mlmodel.save(str(yolo_output_path))
                print(f"✅ YOLOモデル変換完了: {yolo_output_path}")
                
            except Exception as e:
                print(f"❌ YOLOモデル変換エラー: {e}")
        
        # GRUモデルの変換
        print("🧠 GRUモデルの変換...")
        gru_onnx_path = "VirtualTrainerApp/VirtualTrainerApp/MLModels/GRUFormClassifier.onnx"
        if os.path.exists(gru_onnx_path):
            try:
                # ONNX to Core ML
                gru_mlmodel = ct.convert(
                    gru_onnx_path,
                    minimum_deployment_target=ct.target.iOS16,
                    compute_units=ct.ComputeUnit.ALL,
                    convert_to="mlprogram"
                )
                
                # メタデータの追加
                gru_mlmodel.short_description = "エクササイズフォーム判定用GRUモデル"
                gru_mlmodel.author = "Virtual Trainer"
                gru_mlmodel.license = "MIT"
                gru_mlmodel.version = "1.0"
                
                # 入出力の説明
                gru_mlmodel.input_description["features"] = "正規化されたキーポイント特徴量 (10フレーム × 12次元)"
                gru_mlmodel.output_description["prediction"] = "フォーム分類確率 (0: Normal, 1: Error)"
                
                # 保存
                gru_output_path = output_dir / "GRUFormClassifier.mlpackage"
                gru_mlmodel.save(str(gru_output_path))
                print(f"✅ GRUモデル変換完了: {gru_output_path}")
                
            except Exception as e:
                print(f"❌ GRUモデル変換エラー: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core ML変換エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_models():
    """変換されたCore MLモデルの検証"""
    print("🔍 変換されたCore MLモデルを検証中...")
    
    models_dir = Path("VirtualTrainerApp/VirtualTrainerApp/MLModels")
    
    yolo_path = models_dir / "YOLO11nPose.mlpackage"
    gru_path = models_dir / "GRUFormClassifier.mlpackage"
    
    success = True
    
    if yolo_path.exists():
        try:
            size_mb = sum(f.stat().st_size for f in yolo_path.rglob('*') if f.is_file()) / (1024*1024)
            print(f"✅ YOLO11n-pose: {size_mb:.1f}MB")
        except:
            print("⚠️  YOLO11n-poseモデルのサイズ計算に失敗")
    else:
        print("❌ YOLO11n-poseモデルが見つかりません")
        success = False
        
    if gru_path.exists():
        try:
            size_mb = sum(f.stat().st_size for f in gru_path.rglob('*') if f.is_file()) / (1024*1024)
            print(f"✅ GRUFormClassifier: {size_mb:.1f}MB")
        except:
            print("⚠️  GRUFormClassifierモデルのサイズ計算に失敗")
    else:
        print("❌ GRUFormClassifierモデルが見つかりません")
        success = False
    
    return success

def main():
    """メイン関数"""
    print("🚀 ONNX to Core ML 変換を開始します")
    print("=" * 50)
    
    success = convert_onnx_to_coreml()
    
    if success:
        print("\n" + "=" * 50)
        if verify_models():
            print("🎉 すべてのCore MLモデル変換が正常に完了しました！")
            print("\n次のステップ:")
            print("1. Xcodeプロジェクトを開く")
            print("2. MLModels フォルダを確認")
            print("3. アプリを実行してAI機能をテスト")
            print("4. デバッグ情報で「AI: 有効」が表示されることを確認")
            return True
        else:
            print("⚠️  一部のモデル変換に問題があります")
            return False
    else:
        print("\n❌ Core MLモデル変換が失敗しました")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)