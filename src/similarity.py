import numpy as np
import os
from numpy.linalg import norm

TEMPLATE_DIR = "data/templates"

# -------------------------
# LOAD SEMUA TEMPLATE
# -------------------------
def load_templates():
    templates = {}

    for file in os.listdir(TEMPLATE_DIR):
        if file.endswith(".npy"):
            word = os.path.splitext(file)[0]
            path = os.path.join(TEMPLATE_DIR, file)
            templates[word] = np.load(path)

    return templates


# -------------------------
# COSINE SIMILARITY
# -------------------------
def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()

    if norm(a) == 0 or norm(b) == 0:
        return 0.0

    return np.dot(a, b) / (norm(a) * norm(b))


# -------------------------
# PREDIKSI GESTURE
# -------------------------
def predict_gesture(input_sequence, templates):
    best_word = None
    best_score = -1

    for word, template in templates.items():
        score = cosine_similarity(input_sequence, template)

        if score > best_score:
            best_score = score
            best_word = word

    return best_word, best_score


# if __name__ == "__main__":
#     templates = load_templates()

#     # TEST: bandingkan template MAKAN dengan dirinya sendiri
#     test_input = templates["TIDUR"]

#     word, score = predict_gesture(test_input, templates)

#     print("Prediksi:", word)
#     print("Score:", score)
