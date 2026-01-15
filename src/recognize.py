import cv2
import numpy as np
import mediapipe as mp
from similarity import load_templates, predict_gesture

MAX_FRAMES = 30

# Load template gesture
templates = load_templates()

# MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
sequence = []

print("🎥 Webcam aktif. Tekan 'q' untuk keluar.")

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

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
