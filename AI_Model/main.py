import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import os

# --- GRUモデルの定義 (量子化モデルを読み込むために必要) ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x); out = out[:, -1, :]; out = self.dropout(out); out = self.fc(out)
        return out

# --- ヘルパー関数 ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    if np.all(a==0) or np.all(b==0) or np.all(c==0): return 0.0
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

def process_keypoints(frame_keypoints):
    KP_MAPPING = {'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10}
    indices_to_use = sorted(list(KP_MAPPING.values()))
    if frame_keypoints is None or len(frame_keypoints) != 17: return None
    points = np.array([frame_keypoints[i] for i in indices_to_use], dtype=np.float32)
    if np.any(points == 0.0): return None
    left_shoulder = np.array(frame_keypoints[KP_MAPPING['left_shoulder']]); right_shoulder = np.array(frame_keypoints[KP_MAPPING['right_shoulder']])
    center_point = (left_shoulder + right_shoulder) / 2.0; relative_points = points - center_point
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist < 1e-6: return None
    normalized_points = relative_points / shoulder_dist
    return normalized_points.flatten()

# --- メイン実行ブロック ---
if __name__ == '__main__':
    print("モデルを読み込んでいます...")
    yolo_model = YOLO("yolo11n-pose.pt")
    
    quantized_model_path = "best_gru_model_v7_quantized.pth"
    if not os.path.exists(quantized_model_path):
        print(f"エラー: 量子化済みモデルが見つかりません: {quantized_model_path}"); exit()
    gru_model = torch.load(quantized_model_path, weights_only=False)
    gru_model.eval()
    print("モデルの読み込み完了。")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("エラー: カメラを開けませんでした。"); exit()
        
    class_names = ['Normal', 'Elbow Error']
    
    # --- 回数カウント用の設定 ---
    rep_counter = 0
    state = 'top'
    TOP_THRESHOLD = 130.0
    BOTTOM_THRESHOLD = 100.0
    # ★★★★★ 新しい変数：エクササイズゾーンの状態 ★★★★★
    exercise_active = False
    
    rep_keypoints = []
    last_verdict = "Ready"
    verdict_color = (255, 255, 255)
    current_elbow_angle = 0.0

    print("リアルタイム判定を開始します... 'q'キーで終了します。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].keypoints and results[0].keypoints.shape[1] > 0:
            keypoints_xy = results[0].keypoints.xy[0].cpu().numpy().tolist()
            
            # --- 座標の取得 ---
            l_sh_xy, r_sh_xy = keypoints_xy[5], keypoints_xy[6]
            l_el_xy, r_el_xy = keypoints_xy[7], keypoints_xy[8]
            l_wr_xy, r_wr_xy = keypoints_xy[9], keypoints_xy[10]

            # ★★★★★ 新しいロジック：エクササイズゾーンの判定 ★★★★★
            # y座標は画面上部が0のため、値が小さい方が上にある
            shoulder_y = (l_sh_xy[1] + r_sh_xy[1]) / 2
            wrist_y = (l_wr_xy[1] + r_wr_xy[1]) / 2
            
            # 手首が肩より上にある場合のみ、アクティブと判断
            if wrist_y < shoulder_y:
                exercise_active = True
            else:
                exercise_active = False
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

            current_elbow_angle = (calculate_angle(l_sh_xy, l_el_xy, l_wr_xy) + calculate_angle(r_sh_xy, r_el_xy, r_wr_xy)) / 2
            
            # --- 回数カウントのロジック ---
            if exercise_active:
                if state == 'top' and current_elbow_angle < BOTTOM_THRESHOLD:
                    state = 'bottom'
                    rep_keypoints.clear()
                
                # ★★★★★ 条件を追加：エクササイズゾーン内にいる時だけカウント ★★★★★
                elif state == 'bottom' and current_elbow_angle > TOP_THRESHOLD:
                    state = 'top'
                    rep_counter += 1
                    print(f"レップカウント: {rep_counter}")
                    
                    if len(rep_keypoints) > 10:
                        input_tensor = torch.tensor(rep_keypoints, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            output = gru_model(input_tensor)
                        prob = torch.sigmoid(output).item()
                        prediction = 1 if prob > 0.5 else 0
                        last_verdict = class_names[prediction]
                        verdict_color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                    else:
                        last_verdict = "Too Fast"
                        verdict_color = (0, 255, 255)

            features = process_keypoints(keypoints_xy)
            if features is not None:
                rep_keypoints.append(features)

        # --- 画面表示 ---
        cv2.putText(annotated_frame, f"Form: {last_verdict}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, verdict_color, 3)
        cv2.putText(annotated_frame, f"Reps: {rep_counter}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 3)
        
        debug_box_x = annotated_frame.shape[1] - 300
        cv2.putText(annotated_frame, "DEBUG INFO", (debug_box_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # ★★★★★ 新しいデバッグ情報を追加 ★★★★★
        active_text = "ACTIVE" if exercise_active else "INACTIVE"
        active_color = (0, 255, 0) if exercise_active else (100, 100, 100)
        cv2.putText(annotated_frame, f"Zone: {active_text}", (debug_box_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, active_color, 2)
        cv2.putText(annotated_frame, f"State: {state}", (debug_box_x, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(annotated_frame, f"Angle: {current_elbow_angle:.1f}", (debug_box_x, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.imshow('Real-time Form Coach', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
