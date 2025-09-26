import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import os
import threading
import queue
import pygame  # ★★★ playsoundの代わりに使用 ★★★

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

# ★★★ 音声ファイルのマッピング ★★★
AUDIO_FILE_MAP = {
    'Shrug Error': "audio/shrug_error.wav",
    'Elbow-Hand Error': "audio/elbow_error.wav",
    'Too Fast': "audio/too_fast.wav",
    'Too Slow': "audio/too_slow.wav"
    ""
}

# --- 速度判定の閾値 ---
MIN_FRAMES_PER_REP = 10
MAX_FRAMES_PER_REP = 60

# --- --- --- GRUモデルの定義 --- --- ---
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

# --- --- --- ヘルパー関数 --- --- ---
def process_keypoints_for_inference(frame_keypoints_xy):
    if frame_keypoints_xy is None or len(frame_keypoints_xy) != 17: return None
    indices_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    points = np.array([frame_keypoints_xy[i] for i in indices_to_use], dtype=np.float32)
    if np.any(np.all(points == 0.0, axis=1)): return None
    left_shoulder = np.array(frame_keypoints_xy[5]); right_shoulder = np.array(frame_keypoints_xy[6])
    center_point = (left_shoulder + right_shoulder) / 2.0
    relative_points = points - center_point
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist < 1e-6: return None
    normalized_points = relative_points / shoulder_dist
    return normalized_points.flatten()

# ★★★ 修正点: pygameを使った音声再生ワーカー ★★★
def audio_worker(q):
    """キューから音声ファイルのパスを受け取り、順番に再生する"""
    pygame.mixer.init() # このスレッドでミキサーを初期化
    while True:
        try:
            file_path = q.get()
            if file_path is None: # 終了シグナル
                break
            if os.path.exists(file_path):
                sound = pygame.mixer.Sound(file_path)
                sound.play()
                # 再生が終わるまで待機
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

    # ★★★ 音声再生キューとワーカースレッドを開始 ★★★
    audio_queue = queue.Queue()
    audio_thread = threading.Thread(target=audio_worker, args=(audio_queue,))
    audio_thread.daemon = True
    audio_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): exit("エラー: カメラを開けませんでした。")

    rep_counter, state = 0, 'start'
    rep_keypoints_sequence = []
    last_verdict, verdict_color = "Let's Start", (255, 255, 255)

    print("リアルタイム判定を開始します... 'q'キーで終了します。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        body_is_visible = False

        if results[0].keypoints and results[0].keypoints.shape[1] > 0:
            keypoints_xy = results[0].keypoints.xy[0].cpu().numpy().tolist()
            kps_to_check = [keypoints_xy[i] for i in [0, 5, 6, 7, 8, 9, 10, 11, 12]]
            if all(coord != [0.0, 0.0] for coord in kps_to_check):
                body_is_visible = True
        
        if body_is_visible:
            nose, l_sh, r_sh = keypoints_xy[0], keypoints_xy[5], keypoints_xy[6]
            l_el, r_el = keypoints_xy[7], keypoints_xy[8]
            l_wr, r_wr = keypoints_xy[9], keypoints_xy[10]
            l_hip, r_hip = keypoints_xy[11], keypoints_xy[12]
            shoulder_y = (l_sh[1] + r_sh[1]) / 2.0; elbow_y = (l_el[1] + r_el[1]) / 2.0
            wrist_y = (l_wr[1] + r_wr[1]) / 2.0; nose_y = nose[1]
            shoulder_width = abs(l_sh[0] - r_sh[0])
            is_up = wrist_y < shoulder_y
            is_out = abs(l_wr[0] - l_sh[0]) > shoulder_width * 0.7 and abs(r_wr[0] - r_sh[0]) > shoulder_width * 0.7
            is_down = wrist_y > elbow_y
            is_in = abs(l_wr[0] - l_hip[0]) < shoulder_width * 0.5 and abs(r_wr[0] - r_hip[0]) < shoulder_width * 0.5
            is_too_high = wrist_y < nose_y
            elbow_is_up = elbow_y < (shoulder_y + shoulder_width * 0.1)

            if state == 'start' and is_up and is_out and elbow_is_up:
                state = 'lifted'; rep_keypoints_sequence.clear()
            elif state == 'lifted' and is_down and is_in:
                state = 'start'; rep_counter += 1
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
                count_to_play = min(rep_counter, 10)
                count_audio_file = f"audio/{count_to_play}.wav"
                audio_queue.put(count_audio_file)

                if last_verdict in AUDIO_FILE_MAP:
                    error_audio_file = AUDIO_FILE_MAP[last_verdict]
                    audio_queue.put(error_audio_file)

                rep_keypoints_sequence.clear()
            elif state == 'lifted' and is_too_high:
                state = 'over'; rep_keypoints_sequence.clear()
            elif state == 'over' and is_down and is_in:
                state = 'start'

            if state == 'lifted':
                features = process_keypoints_for_inference(keypoints_xy)
                if features is not None: rep_keypoints_sequence.append(features)
        else:
            state = 'start'; rep_keypoints_sequence.clear()
            if rep_counter == 0:
                last_verdict, verdict_color = "Let's Start", (255, 255, 255)

        cv2.putText(annotated_frame, f"Form: {last_verdict}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, verdict_color, 4)
        cv2.putText(annotated_frame, f"Reps: {rep_counter}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 4)
        
        cv2.imshow('Real-time Side Raise Coach', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    # --- 終了処理 ---
    audio_queue.put(None)
    audio_thread.join()
    cap.release()
    cv2.destroyAllWindows()

