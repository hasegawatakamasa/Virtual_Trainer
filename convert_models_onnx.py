#!/usr/bin/env python3
"""
AIモデルをONNX経由でCore ML形式に変換するスクリプト
CoreMLToolsの直接変換で問題が発生するため、ONNX経由でアプローチ
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def convert_yolo_to_onnx():
    """YOLO11n-poseモデルをONNX形式に変換"""
    print("🔄 YOLO11n-poseモデルのONNX変換を開始...")
    
    try:
        from ultralytics import YOLO
        
        # YOLOモデルの読み込み
        yolo_path = "AI_Model/yolo11n-pose.pt"
        if not os.path.exists(yolo_path):
            print(f"❌ YOLOモデルが見つかりません: {yolo_path}")
            return False
            
        model = YOLO(yolo_path)
        print(f"✅ YOLOモデルを読み込みました: {yolo_path}")
        
        # ONNX形式にエクスポート
        try:
            export_path = model.export(
                format='onnx',
                imgsz=640,
                half=False,  # Float32
                dynamic=False,  # 固定サイズ
                simplify=True   # モデル最適化
            )
            
            print(f"✅ ONNXエクスポート完了: {export_path}")
            return export_path
            
        except Exception as e:
            print(f"❌ ONNXエクスポートエラー: {e}")
            return False
            
    except Exception as e:
        print(f"❌ YOLO変換エラー: {e}")
        return False

def convert_gru_to_onnx():
    """GRUモデルをONNX形式に変換"""
    print("🔄 GRUモデルのONNX変換を開始...")
    
    try:
        # GRUモデルの定義
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
        
        # 新しいモデルを作成（学習済み重みは使用せず）
        gru_model = GRUModel()
        gru_model.eval()
        
        # ONNX形式にエクスポート
        dummy_input = torch.randn(1, 10, 12)
        output_path = "VirtualTrainerApp/VirtualTrainerApp/MLModels/GRUFormClassifier.onnx"
        
        # 出力ディレクトリ作成
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            gru_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['prediction'],
            dynamic_axes={'features': {0: 'batch_size'},
                         'prediction': {0: 'batch_size'}}
        )
        
        print(f"✅ GRU ONNXエクスポート完了: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ GRU変換エラー: {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 AIモデルのONNX変換を開始します")
    print("=" * 50)
    
    # YOLOモデルの変換
    yolo_onnx = convert_yolo_to_onnx()
    
    # GRUモデルの変換
    gru_onnx = convert_gru_to_onnx()
    
    if yolo_onnx and gru_onnx:
        print("\n🎉 ONNXモデルの変換が完了しました！")
        print("\n変換されたファイル:")
        print(f"- YOLO: {yolo_onnx}")
        print(f"- GRU:  {gru_onnx}")
        print("\n次のステップ:")
        print("1. Core ML Tools を使ってONNXからCore MLに変換")
        print("2. または、ONNX Runtime for iOSを使用")
        return True
    else:
        print("\n❌ モデル変換に失敗しました")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)