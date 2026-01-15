import cv2
import os
import numpy as np
import mediapipe as mp

# -------------------------
# KONFIGURASI
# -------------------------
VIDEO_DIR = "data/videos_raw"
OUTPUT_DIR = "data/templates"
MAX_FRAMES = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------
# FUNGSI EKSTRAK LANDMARK
# -------------------------
def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            sequence.append(landmarks)

    cap.release()

    if len(sequence) == 0:
        return None

    sequence = np.array(sequence)

    # Normalisasi panjang sequence
    if len(sequence) > MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]
    else:
        pad_len = MAX_FRAMES - len(sequence)
        sequence = np.pad(sequence, ((0, pad_len), (0, 0), (0, 0)))

    return sequence


# -------------------------
# PROSES SEMUA VIDEO
# -------------------------
for file in os.listdir(VIDEO_DIR):
    if not file.lower().endswith(".webm"):
        continue

    word = os.path.splitext(file)[0]
    video_path = os.path.join(VIDEO_DIR, file)

    print(f"Processing: {word}")

    landmarks = extract_landmarks_from_video(video_path)

    if landmarks is None:
        print(f"  ❌ GAGAL: {word} (tangan tidak terdeteksi)")
        continue

    output_path = os.path.join(OUTPUT_DIR, f"{word}.npy")
    np.save(output_path, landmarks)

    print(f"  ✅ BERHASIL → {output_path}")
