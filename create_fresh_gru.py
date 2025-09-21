#!/usr/bin/env python3
"""
新しいGRUモデルを作成してCore MLに変換（量子化問題を回避）
"""

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from pathlib import Path
import sys

# GRUモデルの定義
class GRUModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 最後のタイムステップ
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_and_convert_gru():
    """新しいGRUモデルを作成してCore MLに変換"""
    print("🔄 新しいGRUモデルの作成と変換を開始...")
    
    # 新しいモデルを作成（学習済み重みは使用せず、構造のみ）
    gru_model = GRUModel()
    gru_model.eval()
    
    print("✅ GRUモデルを作成しました")
    
    try:
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
        mlmodel.short_description = "エクササイズフォーム判定用GRUモデル（テスト版）"
        mlmodel.author = "Virtual Trainer"
        mlmodel.license = "MIT"
        mlmodel.version = "1.0-test"
        
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

if __name__ == "__main__":
    success = create_and_convert_gru()
    if success:
        print("\n🎉 GRUモデルの変換が完了しました！")
        print("\n⚠️  注意: これはテスト用の未学習モデルです")
        print("実際の学習済み重みを使用するには、元のPythonファイルから重みを手動で移行する必要があります")
    else:
        print("\n❌ GRUモデルの変換に失敗しました")
    
    sys.exit(0 if success else 1)