import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import os

# --- --- --- 設定 (Configuration) --- --- ---

# ▼▼▼ 学習・量子化時と完全に同じパラメータを指定してください ▼▼▼
INPUT_SIZE = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 5
NUM_CLASSES = 3
DROPOUT = 0.3
# ▲▲▲ ここまで ▲▲▲

# --- モデルファイルとクラス名の設定 ---
YOLO_MODEL_PATH = "yolo11n-pose.pt"
QUANTIZED_MODEL_PATH = "pushup_gru_model_quantized.pth"
CLASS_NAMES = ['Normal', 'Elbows Out Error', 'Hips Sag Error']

# --- カウントと速度判定の閾値 ---
DOWN_ANGLE_THRESHOLD = 100   # この角度より小さくなったら「体を下げた」と判定
UP_ANGLE_THRESHOLD = 160  # この角度より大きくなったら「体を上げた」と判定
MIN_FRAMES_PER_REP = 10   # これより速いと "Too Fast"
MAX_FRAMES_PER_REP = 90   # これより遅いと "Too Slow"

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

# --- --- --- ヘルパー関数 --- --- ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def process_keypoints_for_inference(frame_keypoints_xy):
    # 学習時と同じ特徴量エンジニアリング
    if frame_keypoints_xy is None or len(frame_keypoints_xy) != 17: return None
    kps = {
        'nose': 0, 'left_shoulder': 5, 'right_shoulder': 6, 
        'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10,
        'left_hip': 11, 'right_hip': 12
    }
    coords = {name: np.array(frame_keypoints_xy[idx]) for name, idx in kps.items()}
    if any(np.all(coord == 0.0) for coord in coords.values()): return None

    shoulder_width = np.linalg.norm(coords['right_shoulder'] - coords['left_shoulder'])
    if shoulder_width < 1e-6: return None

    shoulder_center = (coords['left_shoulder'] + coords['right_shoulder']) / 2.0
    hip_center = (coords['left_hip'] + coords['right_hip']) / 2.0
    
    elbow_width = np.linalg.norm(coords['right_elbow'] - coords['left_elbow'])
    wrist_width = np.linalg.norm(coords['right_wrist'] - coords['left_wrist'])
    feat1 = elbow_width / wrist_width if wrist_width > 1e-6 else 0.0

    left_forearm_rad = np.arctan2(coords['left_elbow'][1] - coords['left_wrist'][1], coords['left_elbow'][0] - coords['left_wrist'][0])
    right_forearm_rad = np.arctan2(coords['right_elbow'][1] - coords['right_wrist'][1], coords['right_elbow'][0] - coords['right_wrist'][0])
    feat2 = (np.abs(np.degrees(left_forearm_rad) - 90) + np.abs(np.degrees(right_forearm_rad) - 90)) / 2.0 / 90.0

    feat3 = (hip_center[1] - shoulder_center[1]) / shoulder_width
    feat4 = (shoulder_center[1] - coords['nose'][1]) / shoulder_width
    feat5 = (coords['left_elbow'][1] - coords['right_elbow'][1]) / shoulder_width
    feat6 = shoulder_center[1] / 480.0 # frame_heightを仮定

    return [feat1, feat2, feat3, feat4, feat5, feat6]

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
    angle_history = deque(maxlen=5) # 判定安定化のため

    print("リアルタイム判定を開始します... 'q'キーで終了します。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        body_is_visible = False

        if results[0].keypoints and results[0].keypoints.shape[1] > 0:
            keypoints_xy = results[0].keypoints.xy[0].cpu().numpy().tolist()
            kps_to_check = [keypoints_xy[i] for i in [5, 6, 7, 8, 9, 10, 11, 12]] # 肩, 肘, 手首, 腰
            if all(coord != [0.0, 0.0] for coord in kps_to_check):
                body_is_visible = True
        
        if body_is_visible:
            feedback_message = ""
            # --- 角度の計算 ---
            l_sh, r_sh = keypoints_xy[5], keypoints_xy[6]
            l_el, r_el = keypoints_xy[7], keypoints_xy[8]
            l_wr, r_wr = keypoints_xy[9], keypoints_xy[10]
            left_angle = calculate_angle(l_sh, l_el, l_wr)
            right_angle = calculate_angle(r_sh, r_el, r_wr)
            current_angle = (left_angle + right_angle) / 2
            
            angle_history.append(current_angle)
            smoothed_angle = np.mean(angle_history)

            # --- 状態遷移ロジック ---
            if state == 'up' and smoothed_angle < DOWN_ANGLE_THRESHOLD:
                state = 'down'
                rep_keypoints_sequence.clear()
            elif state == 'down' and smoothed_angle > UP_ANGLE_THRESHOLD:
                state = 'up'
                rep_counter += 1
                
                num_frames = len(rep_keypoints_sequence)
                if num_frames < MIN_FRAMES_PER_REP:
                    last_verdict = "Too Fast"
                    verdict_color = (0, 255, 255) # 黄色
                elif num_frames > MAX_FRAMES_PER_REP:
                    last_verdict = "Too Slow"
                    verdict_color = (0, 165, 255) # オレンジ
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
            angle_history.clear()
            feedback_message = "Please fit your upper body in the frame"

        cv2.putText(annotated_frame, f"Form: {last_verdict}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, verdict_color, 4)
        cv2.putText(annotated_frame, f"Reps: {rep_counter}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 4)
        if feedback_message:
            cv2.putText(annotated_frame, feedback_message, (50, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow('Real-time Push-up Coach', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
