import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import os

# --- --- --- 設定 (Configuration) --- --- ---

# ▼▼▼ 学習・量子化時と完全に同じパラメータを指定してください ▼▼▼
INPUT_SIZE = 22
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 3
DROPOUT = 0.5
# ▲▲▲ ここまで ▲▲▲

# --- モデルファイルとクラス名の設定 ---
YOLO_MODEL_PATH = "yolo11n-pose.pt"
QUANTIZED_MODEL_PATH = "side_raise_gru_model_quantized.pth"
CLASS_NAMES = ['Normal', 'Shrug Error', 'Elbow-Hand Error']

# --- 速度判定の閾値 ---
MIN_FRAMES_PER_REP = 10
MAX_FRAMES_PER_REP = 30

# --- --- --- GRUモデルの定義 (量子化モデルの読み込みに必要) --- --- ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# --- --- --- ヘルパー関数 (特徴量エンジニアリング) --- --- ---
def process_keypoints_for_inference(frame_keypoints_xy):
    if frame_keypoints_xy is None or len(frame_keypoints_xy) != 17:
        return None
        
    indices_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    points = np.array([frame_keypoints_xy[i] for i in indices_to_use], dtype=np.float32)

    if np.any(np.all(points == 0.0, axis=1)):
        return None

    left_shoulder = np.array(frame_keypoints_xy[5])
    right_shoulder = np.array(frame_keypoints_xy[6])
    center_point = (left_shoulder + right_shoulder) / 2.0
    relative_points = points - center_point
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

    if shoulder_dist < 1e-6:
        return None

    normalized_points = relative_points / shoulder_dist
    return normalized_points.flatten()

# --- --- --- メイン実行ブロック --- --- ---
if __name__ == '__main__':
    # --- モデルの読み込み ---
    print("モデルを読み込んでいます...")
    if not os.path.exists(YOLO_MODEL_PATH) or not os.path.exists(QUANTIZED_MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません。'{YOLO_MODEL_PATH}' と '{QUANTIZED_MODEL_PATH}' を確認してください。")
        exit()

    yolo_model = YOLO(YOLO_MODEL_PATH)
    gru_model = torch.load(QUANTIZED_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    gru_model.eval()
    print("モデルの読み込み完了。")

    # --- カメラの準備 ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        exit()

    # --- 変数の初期化 ---
    rep_counter = 0
    state = 'down'
    rep_keypoints_sequence = []
    last_verdict = "Ready"
    verdict_color = (255, 255, 255)

    print("リアルタイム判定を開始します... 'q'キーで終了します。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].keypoints and results[0].keypoints.shape[1] > 0:
            keypoints_xy = results[0].keypoints.xy[0].cpu().numpy().tolist()
            
            # --- 座標の取得 ---
            l_sh_xy = keypoints_xy[5];  r_sh_xy = keypoints_xy[6]
            l_el_xy = keypoints_xy[7];  r_el_xy = keypoints_xy[8]
            l_wr_xy = keypoints_xy[9];  r_wr_xy = keypoints_xy[10]

            # 必要なキーポイントがすべて検出されているかチェック
            if all(coord != [0.0, 0.0] for coord in [l_sh_xy, r_sh_xy, l_el_xy, r_el_xy, l_wr_xy, r_wr_xy]):
                
                # --- y座標の平均値を計算 ---
                # y座標は画面上部が0（小さいほど上）
                shoulder_y = (l_sh_xy[1] + r_sh_xy[1]) / 2.0
                elbow_y = (l_el_xy[1] + r_el_xy[1]) / 2.0
                wrist_y = (l_wr_xy[1] + r_wr_xy[1]) / 2.0
                
                # --- ★★★ 新しい回数カウントロジック ★★★ ---
                # 状態'down'の条件: 手首が肘より下にある
                is_down_position = wrist_y > elbow_y
                # 状態'up'の条件: 手首が肩より上にある
                is_up_position = wrist_y < shoulder_y

                if state == 'down' and is_up_position:
                    state = 'up'
                    rep_keypoints_sequence.clear() # レップ開始

                elif state == 'up' and is_down_position:
                    state = 'down'
                    rep_counter += 1
                    
                    # --- フォーム判定 ---
                    num_frames = len(rep_keypoints_sequence)
                    if num_frames < MIN_FRAMES_PER_REP:
                        last_verdict = "Too Fast"
                        verdict_color = (0, 255, 255)
                    elif num_frames > MAX_FRAMES_PER_REP:
                        last_verdict = "Too Slow"
                        verdict_color = (0, 165, 255)
                    else:
                        input_tensor = torch.tensor(rep_keypoints_sequence, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            output = gru_model(input_tensor)
                            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).item()
                            last_verdict = CLASS_NAMES[prediction]
                            if prediction == 0: verdict_color = (0, 255, 0)
                            else: verdict_color = (0, 0, 255)
                    
                    rep_keypoints_sequence.clear()

                # 状態が'up'の間だけキーポイントを記録
                if state == 'up':
                    features = process_keypoints_for_inference(keypoints_xy)
                    if features is not None:
                        rep_keypoints_sequence.append(features)

        cv2.putText(annotated_frame, f"Form: {last_verdict}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, verdict_color, 4)
        cv2.putText(annotated_frame, f"Reps: {rep_counter}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 4)
        
        cv2.imshow('Real-time Side Raise Coach', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()