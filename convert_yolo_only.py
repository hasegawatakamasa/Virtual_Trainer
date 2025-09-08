#!/usr/bin/env python3
"""
YOLO11n-poseモデルのみをCore ML形式に変換
"""

import os
import sys
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
        
        # 出力ディレクトリ作成
        output_dir = Path("VirtualTrainerApp/VirtualTrainerApp/MLModels")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Core ML形式にエクスポート（シンプルな設定）
        try:
            export_path = model.export(
                format='coreml',
                imgsz=640,
                half=False,   # Float32で試す
                nms=False,    # Poseモデルなので無効
                int8=False
            )
            print(f"✅ エクスポート完了: {export_path}")
            
            # ファイル移動
            import shutil
            target_path = output_dir / "YOLO11nPose.mlpackage"
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.move(export_path, target_path)
            
            print(f"✅ YOLOモデルをCore ML形式で保存: {target_path}")
            
            # サイズ確認
            size_mb = sum(f.stat().st_size for f in target_path.rglob('*') if f.is_file()) / (1024*1024)
            print(f"📊 モデルサイズ: {size_mb:.1f}MB")
            
            return True
            
        except Exception as export_error:
            print(f"❌ エクスポートエラー: {export_error}")
            # より基本的な設定で再試行
            try:
                print("🔄 基本設定で再試行中...")
                export_path = model.export(format='coreml')
                
                import shutil
                target_path = output_dir / "YOLO11nPose.mlpackage"
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.move(export_path, target_path)
                
                print(f"✅ 基本設定でエクスポート成功: {target_path}")
                return True
                
            except Exception as retry_error:
                print(f"❌ 再試行も失敗: {retry_error}")
                return False
        
    except Exception as e:
        print(f"❌ YOLO変換エラー: {e}")
        return False

if __name__ == "__main__":
    print("🚀 YOLO11n-poseモデルのCore ML変換")
    print("=" * 50)
    
    success = convert_yolo_model()
    
    if success:
        print("\n🎉 YOLOモデルの変換が正常に完了しました！")
        print("\n次のステップ:")
        print("1. Xcodeプロジェクトを開く")
        print("2. MLModels フォルダをプロジェクトに追加")
        print("3. モデルがターゲットに含まれていることを確認")
    else:
        print("\n❌ YOLOモデルの変換に失敗しました")
        print("\n対処方法:")
        print("1. venv環境を再作成")
        print("2. 古いバージョンのcoremltools/pytorchを試す")
        print("3. または、現在のiOSコードで模擬データを使用して開発を継続")
    
    sys.exit(0 if success else 1)