import cv2
import os

# TODO 1: Ask the user to type the name of the object (e.g. 'pen', 'bottle')
label = 

# The folder will be called "data/object_name"
save_dir = os.path.join("data", label)
os.makedirs(save_dir, exist_ok=True)  # This creates the folder if it doesn't already exist

# TODO 2: Start the webcam
cap = 
print("Press SPACE to save an image. Press ESC to exit.")

count = 1  # This will be used to name each image file

# Start an infinite loop to capture frames from the webcam
while True:
    ret, frame = cap.read()  # ret = True if frame was read successfully
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Image Collector - " + label, frame)

    key = cv2.waitKey(1)

    # TODO 3 If the SPACE key is pressed, save the image
    if key == :
        filename = f"{label}_{count}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)  # Save the image
        print(f"Saved {filepath}")
        count += 1

    # TODO 4 If ESC key is pressed, exit the loop
    elif key == :
        break

cap.release()
cv2.destroyAllWindows()
