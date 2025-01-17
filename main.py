import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Initialize hand detector and load TensorFlow model
detector = HandDetector(maxHands=1)
model = tf.keras.models.load_model("Model/model.h5")  # Load the Keras model
labels = open("Model/labels.txt").read().strip().split("\n")
model_input_size = model.input_shape[1]  # Get the input size of the model (e.g., 224)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error: Could not open video source.")

# Image preprocessing function
def preprocess(img, bbox):
    x, y, w, h = bbox
    cropped = img[max(0, y - 20):min(img.shape[0], y + h + 20),
                  max(0, x - 20):min(img.shape[1], x + w + 20)]
    white_bg = np.ones((300, 300, 3), dtype=np.uint8) * 255
    aspect_ratio = h / w
    if aspect_ratio > 1:
        resized = cv2.resize(cropped, (int(300 / h * w), 300))
        white_bg[:, (300 - resized.shape[1]) // 2:(300 + resized.shape[1]) // 2] = resized
    else:
        resized = cv2.resize(cropped, (300, int(300 / w * h)))
        white_bg[(300 - resized.shape[0]) // 2:(300 + resized.shape[0]) // 2, :] = resized
    resized = cv2.cvtColor(cv2.resize(white_bg, (model_input_size, model_input_size)), cv2.COLOR_BGR2RGB)
    return np.expand_dims(resized, axis=0) / 255.0  # Normalize to [0, 1] range

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hands = detector.findHands(frame, flipType=True)[0]
    if hands:
        bbox = hands[0]['bbox']
        input_data = preprocess(frame, bbox)
        predictions = model.predict(input_data)  # Get model predictions
        label_idx = np.argmax(predictions[0])  # Get the class index with the highest probability
        label = labels[label_idx]  # Get the label name
        confidence = predictions[0][label_idx] * 100  # Convert to percentage

        # Display label and confidence
        cv2.rectangle(frame, (bbox[0], bbox[1] - 40), (bbox[0] + 150, bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, f"{label}: {confidence:.0f}%", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
