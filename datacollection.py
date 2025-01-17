import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import time

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Configuration
img_size = 300
offset = 20
labels = ['hello', 'thanks', 'yes', 'no', 'i love you']
folder = "Data"
os.makedirs(folder, exist_ok=True)
[label_folder := os.makedirs(os.path.join(folder, label), exist_ok=True) for label in labels]

# Initialize counters
counter, label_index = 0, 0

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    hands = detector.findHands(img, draw=True)[0]  # Detect hands without drawing
    if hands:
        x, y, w, h = hands[0]['bbox']
        cropped = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                      max(0, x - offset):min(img.shape[1], x + w + offset)]

        white_bg = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        aspect_ratio = h / w

        # Adjust cropped image to fit the white background
        if aspect_ratio > 1:
            scaled_width = int(img_size / h * w)
            resized = cv2.resize(cropped, (scaled_width, img_size))
            white_bg[:, (img_size - scaled_width) // 2:(img_size + scaled_width) // 2] = resized
        else:
            scaled_height = int(img_size / w * h)
            resized = cv2.resize(cropped, (img_size, scaled_height))
            white_bg[(img_size - scaled_height) // 2:(img_size + scaled_height) // 2, :] = resized

        # Display images
        cv2.imshow("Cropped Hand", cropped)
        cv2.imshow("Processed Image", white_bg)

    # Show the main webcam feed
    cv2.imshow("Webcam", img)

    # Handle key inputs
    key = cv2.waitKey(1)
    if key == ord("s"):  # Save image
        current_label = labels[label_index]
        save_path = os.path.join(folder, current_label, f"Image_{time.time()}.jpg")
        cv2.imwrite(save_path, white_bg)
        counter += 1
        print(f"Saved image {counter} for label: {current_label}")

        # Switch to the next label after saving 100 images
        if counter >= 100:
            counter = 0
            label_index = (label_index + 1) % len(labels)
            print(f"Switched to label: {labels[label_index]}")

    if key == ord("q"):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
