import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import os
import sys
import threading
import queue
import pygame  # pygameを使って音声を再生

# --- --- --- 設定 (Configuration) --- --- ---

# ▼▼▼ モデルのパラメータとパスを、ご自身の学習済みモデルに合わせてください ▼▼▼
INPUT_SIZE = 12
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.5
QUANTIZED_MODEL_PATH = "best_gru_model_v7_quantized.pth"
CLASS_NAMES = ['Normal', 'Elbow Error']
# ▲▲▲ ここまで ▲▲▲

YOLO_MODEL_PATH = "yolo11n-pose.pt"

# ★★★ 音声ファイルのマッピング ★★★
AUDIO_FILE_MAP = {
    'Elbow Error': "audio/elbow_open.wav",
    'Too Fast': "audio/too_fast.wav",
}

# --- 回数カウント用の設定 ---
TOP_THRESHOLD = 130.0
BOTTOM_THRESHOLD = 100.0

# --- --- --- GRUモデルの定義 (量子化モデルの読み込みに必要) --- --- ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x); out = out[:, -1, :]; out = self.dropout(out); out = self.fc(out)
        return out

# --- --- --- ヘルパー関数 --- --- ---
def process_keypoints(frame_keypoints):
    KP_MAPPING = {'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10}
    indices_to_use = sorted(list(KP_MAPPING.values()))
    if frame_keypoints is None or len(frame_keypoints) != 17: return None
    points = np.array([frame_keypoints[i] for i in indices_to_use], dtype=np.float32)
    if np.any(np.all(points == 0.0, axis=1)): return None
    left_shoulder = np.array(frame_keypoints[KP_MAPPING['left_shoulder']]); right_shoulder = np.array(frame_keypoints[KP_MAPPING['right_shoulder']])
    center_point = (left_shoulder + right_shoulder) / 2.0
    relative_points = points - center_point
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist < 1e-6: return None
    normalized_points = relative_points / shoulder_dist
    return normalized_points.flatten()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.all(a==0) or np.all(b==0) or np.all(c==0): return 0.0
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- pygameを使った音声再生ワーカー ---
def audio_worker(q):
    pygame.mixer.init()
    while True:
        try:
            file_path = q.get()
            if file_path is None: break
            if os.path.exists(file_path):
                sound = pygame.mixer.Sound(file_path)
                sound.play()
                while pygame.mixer.get_busy():
                    pygame.time.Clock().tick(10)
            else:
                print(f"Warning: Audio file not found at {file_path}")
            q.task_done()
        except Exception as e:
            print(f"Audio worker error: {e}")
            break

# --- --- --- メイン実行ブロック --- --- ---
if __name__ == '__main__':
    if sys.platform == "darwin":
        torch.backends.quantized.engine = 'qnnpack'

    print("モデルを読み込んでいます...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    if not os.path.exists(QUANTIZED_MODEL_PATH):
        print(f"エラー: 量子化済みモデルが見つかりません: {QUANTIZED_MODEL_PATH}"); exit()
    gru_model = torch.load(QUANTIZED_MODEL_PATH, map_location='cpu', weights_only=False)
    gru_model.eval()
    print("モデルの読み込み完了。")

    # --- 音声再生キューとワーカースレッドを開始 ---
    audio_queue = queue.Queue()
    audio_thread = threading.Thread(target=audio_worker, args=(audio_queue,))
    audio_thread.daemon = True
    audio_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("エラー: カメラを開けませんでした。"); exit()
    
    rep_counter, state = 0, 'top'
    rep_keypoints_sequence = []
    last_verdict, verdict_color = "Ready", (255, 255, 255)
    
    print("リアルタイム判定を開始します... 'q'キーで終了します。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        body_is_visible = False

        if results[0].keypoints and results[0].keypoints.shape[1] > 0:
            keypoints_xy = results[0].keypoints.xy[0].cpu().numpy().tolist()
            kps_to_check = [keypoints_xy[i] for i in [5, 6, 7, 8, 9, 10]]
            if all(coord != [0.0, 0.0] for coord in kps_to_check):
                body_is_visible = True

        if body_is_visible:
            l_sh_xy, r_sh_xy = keypoints_xy[5], keypoints_xy[6]
            l_el_xy, r_el_xy = keypoints_xy[7], keypoints_xy[8]
            l_wr_xy, r_wr_xy = keypoints_xy[9], keypoints_xy[10]

            shoulder_y = (l_sh_xy[1] + r_sh_xy[1]) / 2.0
            wrist_y = (l_wr_xy[1] + r_wr_xy[1]) / 2.0
            
            exercise_active = wrist_y < shoulder_y
            
            if exercise_active:
                current_elbow_angle = (calculate_angle(l_sh_xy, l_el_xy, l_wr_xy) + calculate_angle(r_sh_xy, r_el_xy, r_wr_xy)) / 2.0
                
                if state == 'top' and current_elbow_angle < BOTTOM_THRESHOLD:
                    state = 'bottom'; rep_keypoints_sequence.clear()
                
                elif state == 'bottom' and current_elbow_angle > TOP_THRESHOLD:
                    state = 'top'; rep_counter += 1
                    
                    if len(rep_keypoints_sequence) > 10:
                        sequence_np = np.array(rep_keypoints_sequence, dtype=np.float32)
                        input_tensor = torch.from_numpy(sequence_np).unsqueeze(0)
                        with torch.no_grad():
                            output = gru_model(input_tensor)
                            prob = torch.sigmoid(output).item()
                            prediction = 1 if prob > 0.5 else 0
                            last_verdict = CLASS_NAMES[prediction]
                            verdict_color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                    else:
                        last_verdict = "Too Fast"; verdict_color = (0, 255, 255)

                    # --- 連続音声再生ロジック ---
                    count_to_play = min(rep_counter, 20) # 10回以上は10を再生
                    count_audio_file = f"audio/{count_to_play}.wav"
                    audio_queue.put(count_audio_file)

                    if last_verdict in AUDIO_FILE_MAP:
                        error_audio_file = AUDIO_FILE_MAP[last_verdict]
                        audio_queue.put(error_audio_file)

                    rep_keypoints_sequence.clear()
                
                features = process_keypoints(keypoints_xy)
                if features is not None:
                    rep_keypoints_sequence.append(features)
        else:
            state = 'top'; rep_keypoints_sequence.clear()
            if rep_counter == 0:
                last_verdict, verdict_color = "Ready", (255, 255, 255)

        cv2.putText(annotated_frame, f"Form: {last_verdict}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, verdict_color, 4)
        cv2.putText(annotated_frame, f"Reps: {rep_counter}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 4)
        
        cv2.imshow('Real-time Form Coach', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- 終了処理 ---
    audio_queue.put(None)
    audio_thread.join()
    cap.release()
    cv2.destroyAllWindows()
