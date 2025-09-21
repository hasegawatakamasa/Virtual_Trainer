import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import os

# --- --- --- 設定 (Configuration) --- --- ---

# ▼▼▼ 学習・量子化時と完全に同じパラメータを指定してください ▼▼▼
INPUT_SIZE = 4
HIDDEN_SIZE = 128
NUM_LAYERS = 3
NUM_CLASSES = 2
DROPOUT = 0.4
# ▲▲▲ ここまで ▲▲▲

# --- モデルファイルとクラス名の設定 ---
YOLO_MODEL_PATH = "yolo11n-pose.pt"
QUANTIZED_MODEL_PATH = "squat_gru_model_kie_quantized.pth"
CLASS_NAMES = ['Normal', 'Knees Inward']

# --- カウントと速度判定の閾値 ---
# 肩から腰までの距離を1とした時の、膝と腰の垂直方向の距離の比率で判定
DOWN_THRESHOLD_RATIO = -0.2 # この比率より腰が膝に近づいたら「しゃがんだ」
UP_THRESHOLD_RATIO = -0.6   # この比率より腰が膝から離れたら「立ち上がった」
MIN_FRAMES_PER_REP = 5
MAX_FRAMES_PER_REP = 50

# --- --- --- GRUモデルの定義 (量子化モデルの読み込みに必要) --- --- ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.gru(x); out = out[:, -1, :]; out = self.dropout(out); out = self.fc(out)
        return out

# --- --- --- ヘルパー関数 (特徴量エンジニアリング) --- --- ---
def process_keypoints_for_inference(frame_keypoints_xy):
    # この関数は学習時と完全に同じでなければならないため、変更しません
    if frame_keypoints_xy is None or len(frame_keypoints_xy) != 17: return None
    kps = {'left_shoulder': 5, 'right_shoulder': 6, 'left_hip': 11, 'right_hip': 12, 'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16}
    coords = {name: np.array(frame_keypoints_xy[idx]) for name, idx in kps.items()}
    if any(np.all(coord == 0.0) for coord in coords.values()): return None
    shoulder_center = (coords['left_shoulder'] + coords['right_shoulder']) / 2.0
    hip_center = (coords['left_hip'] + coords['right_hip']) / 2.0
    torso_dist = np.linalg.norm(shoulder_center - hip_center)
    hip_width = np.linalg.norm(coords['right_hip'] - coords['left_hip'])
    if torso_dist < 1e-6 or hip_width < 1e-6: return None
    knee_width = coords['right_knee'][0] - coords['left_knee'][0]
    feat1 = knee_width / hip_width
    feat2 = (coords['left_knee'][0] - coords['left_ankle'][0]) / hip_width
    feat3 = (coords['right_knee'][0] - coords['right_ankle'][0]) / hip_width
    feat4 = (hip_center[1] - shoulder_center[1]) / torso_dist
    return [feat1, feat2, feat3, feat4]

# --- --- --- メイン実行ブロック --- --- ---
if __name__ == '__main__':
    print("モデルを読み込んでいます...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    gru_model = torch.load(QUANTIZED_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    gru_model.eval()
    print("モデルの読み込み完了。")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): exit("エラー: カメラを開けませんでした。")

    rep_counter, state = 0, 'up'
    rep_keypoints_sequence = []
    last_verdict, verdict_color = "Ready", (255, 255, 255)
    feedback_message = ""
    depth_ratio_history = deque(maxlen=10)

    print("リアルタイム判定を開始します... 'q'キーで終了します。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        body_is_visible = False

        if results[0].keypoints and results[0].keypoints.shape[1] > 0:
            keypoints_xy = results[0].keypoints.xy[0].cpu().numpy().tolist()
            kps_to_check = [keypoints_xy[i] for i in [5, 6, 11, 12, 13, 14]] # 肩、腰、膝
            if all(coord != [0.0, 0.0] for coord in kps_to_check):
                body_is_visible = True
        
        if body_is_visible:
            feedback_message = ""
            # --- Y座標と基準距離を計算 ---
            shoulder_y = (keypoints_xy[5][1] + keypoints_xy[6][1]) / 2.0
            hip_y = (keypoints_xy[11][1] + keypoints_xy[12][1]) / 2.0
            knee_y = (keypoints_xy[13][1] + keypoints_xy[14][1]) / 2.0
            
            # ★★★ 変更点: 胴体の長さを基準に判定 ★★★
            vertical_torso_dist = abs(hip_y - shoulder_y)
            depth_ratio = (hip_y - knee_y) / vertical_torso_dist if vertical_torso_dist > 1e-6 else 0
            
            # --- 判定の安定化 ---
            depth_ratio_history.append(depth_ratio)
            smoothed_depth_ratio = np.mean(depth_ratio_history)

            # --- 状態遷移ロジック ---
            if state == 'up' and smoothed_depth_ratio > DOWN_THRESHOLD_RATIO:
                state = 'down'
                rep_keypoints_sequence.clear()
            elif state == 'down' and smoothed_depth_ratio < UP_THRESHOLD_RATIO:
                state = 'up'
                rep_counter += 1
                
                # ★★★ 変更点: 速度判定を復活 ★★★
                num_frames = len(rep_keypoints_sequence)
                if num_frames < MIN_FRAMES_PER_REP:
                    last_verdict = "Too Fast"
                    verdict_color = (0, 255, 255)
                elif num_frames > MAX_FRAMES_PER_REP:
                    last_verdict = "Too Slow"
                    verdict_color = (0, 165, 255)
                else:
                    input_tensor = torch.tensor([rep_keypoints_sequence], dtype=torch.float32)
                    with torch.no_grad():
                        output = gru_model(input_tensor)
                        prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).item()
                        last_verdict = CLASS_NAMES[prediction]
                        verdict_color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                rep_keypoints_sequence.clear()

            features = process_keypoints_for_inference(keypoints_xy)
            if features is not None:
                rep_keypoints_sequence.append(features)
        else:
            state = 'up'
            rep_keypoints_sequence.clear()
            depth_ratio_history.clear()
            feedback_message = "Please fit your body in the frame"

        cv2.putText(annotated_frame, f"Form: {last_verdict}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, verdict_color, 4)
        cv2.putText(annotated_frame, f"Reps: {rep_counter}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 4)
        if feedback_message:
            cv2.putText(annotated_frame, feedback_message, (50, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow('Real-time Squat Coach', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()