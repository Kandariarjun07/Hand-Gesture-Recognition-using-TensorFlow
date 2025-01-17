# Hand Gesture Recognition using TensorFlow

This project implements a real-time hand gesture recognition system using OpenCV, TensorFlow, and the CVZone HandTrackingModule. The system detects hand gestures through a webcam and classifies them using a pre-trained TensorFlow/Keras model.

## Features
- Real-time hand gesture detection using a webcam.
- Preprocessing of hand images to standardize input size.
- Hand gesture classification with a trained TensorFlow model.
- Display of predicted gesture label and confidence.

---

## Installation
### Requirements
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### Requirements File
Ensure the `requirements.txt` contains the following:
```plaintext
opencv-python
numpy
tensorflow
cvzone
mediapipe
```

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. Ensure the following files are present:
   - `Model/model.h5`: Trained TensorFlow/Keras model.
   - `Model/labels.txt`: Text file containing gesture labels (one per line).

3. Run the main script:
   ```bash
   python main.py
   ```

4. Use your webcam to display hand gestures. Press `q` to quit the application.

---

## Dataset Preparation
To train your own model:
1. Collect images for each hand gesture using the provided `datacollection.py` script.
   - Modify the `labels` array in `datacollection.py` to include your desired gestures.
   - Save at least 50 images per gesture for better accuracy.

2. Organize images into folders named after gesture labels.

3. Train the TensorFlow model using the provided training script (e.g., `train_model.py`).

---

## Code Breakdown
### Preprocessing Function
The `preprocess` function prepares images for model input:
- Crops and centers the detected hand.
- Resizes the image to match the model's input dimensions.
- Normalizes pixel values to the `[0, 1]` range.

### Gesture Prediction
The system uses a TensorFlow/Keras model:
1. Preprocesses the hand image.
2. Passes it to the model for prediction.
3. Displays the predicted label and confidence on the webcam feed.

---

## Key Files
- **`datacollection.py`**: Script to collect and save hand gesture images.
- **`train_model.py`**: Script to train the TensorFlow model.
- **`main.py`**: Main script for real-time gesture recognition.
- **`requirements.txt`**: Dependency file.

---

## Usage Notes
- Ensure proper lighting for accurate hand detection.
- Adjust the `offset` and `img_size` parameters in `main.py` and `datacollection.py` as needed for your webcam.

---

## Future Enhancements
- Add support for multi-hand detection.
- Use a larger dataset for better model accuracy.
- Extend to additional applications like sign language interpretation.

---

## License
This project is open-source and available under the MIT License.

---

## Acknowledgments
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [CVZone](https://github.com/cvzone/CVzone)
- [Mediapipe](https://google.github.io/mediapipe/)

---

For any issues or contributions, feel free to open a pull request or contact me.

