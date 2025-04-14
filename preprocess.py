import os
import cv2
import numpy as np
import pickle

data_dir = "data"
img_size = 100

X = []
y = []
labels = os.listdir(data_dir)

labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for label in labels:
    path = os.path.join(data_dir, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Save for later use
with open("dataset.pkl", "wb") as f:
    pickle.dump((X, y), f)

print(f"Saved {len(X)} images to dataset.pkl")
