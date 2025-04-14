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

# Load logo with transparency
logo = cv2.imread("images/techspark_logo.png", cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, (200, 100), interpolation=cv2.INTER_CUBIC)  # Use INTER_CUBIC for better quality


# Split into channels
b, g, r, a = cv2.split(logo)
overlay_color = cv2.merge((b, g, r))
mask = cv2.merge((a, a, a))


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

    # Overlay logo in bottom-left corner
    h_logo, w_logo = logo.shape[:2]
    x_offset, y_offset = 10, frame.shape[0] - h_logo - 40

    # Ensure ROI dimensions match
    roi = frame[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo]
    bg = cv2.bitwise_and(roi, 255 - mask)
    fg = cv2.bitwise_and(overlay_color, mask)
    combined = cv2.add(bg, fg)
    frame[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo] = combined

    # Add text below logo
    cv2.putText(frame, 
                "Powered by TechSpark Academy", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA)


    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
