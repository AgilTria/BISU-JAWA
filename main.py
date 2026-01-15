import cv2
import numpy as np
import mediapipe as mp
import time

from src.similarity import load_templates, predict_gesture
from src.translate import translate
from src.tts import speak

# -------------------------
# KONFIGURASI
# -------------------------
MAX_FRAMES = 30
SCORE_THRESHOLD = 0.90
SPEAK_COOLDOWN = 2.5  # detik

# -------------------------
# LOAD TEMPLATE
# -------------------------
templates = load_templates()

# -------------------------
# MEDIAPIPE
# -------------------------
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

sequence = []
last_word = None
last_spoken_time = 0

print("🎥 Sistem aktif. Tekan 'q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = mp_hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        sequence.append(landmarks)

        if len(sequence) > MAX_FRAMES:
            sequence = sequence[-MAX_FRAMES:]

        if len(sequence) == MAX_FRAMES:
            input_seq = np.array(sequence)
            word, score = predict_gesture(input_seq, templates)

            cv2.putText(
                frame,
                f"{word} ({score:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            now = time.time()

            if (
                score >= SCORE_THRESHOLD and
                word != last_word and
                (now - last_spoken_time) > SPEAK_COOLDOWN
            ):
                result_id = translate(word, mode="ngoko")
                result_jv = translate(word, mode="krama")

                if result_id and result_jv:
                    # Indonesia
                    speak(result_id["indonesia"], lang="id")
                    # Jawa
                    speak(result_jv["jawa"], lang="jw")

                    last_word = word
                    last_spoken_time = now

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
