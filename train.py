# train.py
import os
import cv2
import numpy as np
import json

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])
label_map = {label: idx for idx, label in enumerate(labels)}

faces = []
ids = []
for label, idx in label_map.items():
    folder = os.path.join(DATA_DIR, label)
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        faces.append(img)
        ids.append(idx)

faces = [cv2.resize(f, (200,200)) for f in faces]
ids = np.array(ids)

# create LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.train(faces, ids)
recognizer.write(os.path.join(MODEL_DIR, "lbph.yml"))

with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f)

print("Trained model saved to models/lbph.yml")
print("Label map:", label_map)