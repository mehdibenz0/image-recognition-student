import cv2
import pickle
import numpy as np

# Load model and label encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("labels.pkl", "rb") as f:
    le = pickle.load(f)

img_size = 100
cap = cv2.VideoCapture(0)

print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (img_size, img_size))
    img_flat = img.reshape(1, -1)

    pred = model.predict(img_flat)
    label = le.inverse_transform(pred)[0]

    cv2.putText(frame, f"Prediction: {label}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)


    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
