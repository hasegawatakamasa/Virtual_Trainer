import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import threading
import queue
import pygame  # pygameを使って音声を再生

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

# ★★★ 音声ファイルのマッピング ★★★
AUDIO_FILE_MAP = {
    'Knees Inward': "audio/knees_inward.wav",
    'Too Fast': "audio/too_fast.wav",
    'Too Slow': "audio/too_slow.wav"
}

# --- カウントと速度判定の閾値 ---
DOWN_THRESHOLD_RATIO = -0.2
UP_THRESHOLD_RATIO = -0.6
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

# --- pygameを使った音声再生ワーカー ---
def audio_worker(q):
    """キューから音声ファイルのパスを受け取り、順番に再生する"""
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
    print("モデルを読み込んでいます...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    gru_model = torch.load(QUANTIZED_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    gru_model.eval()
    print("モデルの読み込み完了。")

    # --- 音声再生キューとワーカースレッドを開始 ---
    audio_queue = queue.Queue()
    audio_thread = threading.Thread(target=audio_worker, args=(audio_queue,))
    audio_thread.daemon = True
    audio_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): exit("エラー: カメラを開けませんでした。")

    rep_counter, state = 0, 'up'
    rep_keypoints_sequence = []
    last_verdict, verdict_color = "--", (255, 255, 255)
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
            kps_to_check = [keypoints_xy[i] for i in [5, 6, 11, 12, 13, 14]]
            if all(coord != [0.0, 0.0] for coord in kps_to_check):
                body_is_visible = True
        
        if body_is_visible:
            shoulder_y = (keypoints_xy[5][1] + keypoints_xy[6][1]) / 2.0
            hip_y = (keypoints_xy[11][1] + keypoints_xy[12][1]) / 2.0
            knee_y = (keypoints_xy[13][1] + keypoints_xy[14][1]) / 2.0
            vertical_torso_dist = abs(hip_y - shoulder_y)
            depth_ratio = (hip_y - knee_y) / vertical_torso_dist if vertical_torso_dist > 1e-6 else 0
            
            depth_ratio_history.append(depth_ratio)
            smoothed_depth_ratio = np.mean(depth_ratio_history)

            if state == 'up' and smoothed_depth_ratio > DOWN_THRESHOLD_RATIO:
                state = 'down'
                rep_keypoints_sequence.clear()
            elif state == 'down' and smoothed_depth_ratio < UP_THRESHOLD_RATIO:
                state = 'up'
                rep_counter += 1
                
                num_frames = len(rep_keypoints_sequence)
                if num_frames < MIN_FRAMES_PER_REP:
                    last_verdict = "Too Fast"; verdict_color = (0, 255, 255)
                elif num_frames > MAX_FRAMES_PER_REP:
                    last_verdict = "Too Slow"; verdict_color = (0, 165, 255)
                else:
                    sequence_np = np.array(rep_keypoints_sequence, dtype=np.float32)
                    input_tensor = torch.from_numpy(sequence_np).unsqueeze(0)
                    with torch.no_grad():
                        output = gru_model(input_tensor)
                        prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).item()
                        last_verdict = CLASS_NAMES[prediction]
                        verdict_color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                
                # --- 連続音声再生ロジック ---
                count_to_play = min(rep_counter, 20) # 20回まで対応
                count_audio_file = f"audio/{count_to_play}.wav"
                audio_queue.put(count_audio_file)

                if last_verdict in AUDIO_FILE_MAP:
                    error_audio_file = AUDIO_FILE_MAP[last_verdict]
                    audio_queue.put(error_audio_file)
                
                rep_keypoints_sequence.clear()

            features = process_keypoints_for_inference(keypoints_xy)
            if features is not None:
                rep_keypoints_sequence.append(features)
        else:
            state = 'up'
            rep_keypoints_sequence.clear()
            depth_ratio_history.clear()
            if rep_counter == 0:
                last_verdict, verdict_color = "--", (255, 255, 255)

        cv2.putText(annotated_frame, f"Form: {last_verdict}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, verdict_color, 4)
        cv2.putText(annotated_frame, f"Reps: {rep_counter}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 4)
        
        cv2.imshow('Real-time Squat Coach', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- 終了処理 ---
    audio_queue.put(None)
    audio_thread.join()
    cap.release()
    cv2.destroyAllWindows()
