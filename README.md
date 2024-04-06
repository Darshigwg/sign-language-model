# Sign Language Detection Model

This project is aimed at developing a machine learning model for detecting sign language gestures. The model is trained to recognize various signs and interpret them into corresponding text or actions. This README provides an overview of the project and instructions for running it.

## Introduction

Sign language is a vital means of communication for individuals with hearing impairments. Developing a robust sign language detection model can facilitate better communication and accessibility for this community. This project utilizes machine learning techniques to recognize and interpret sign language gestures in real-time.

## Getting Started

To get started with the project, follow these steps:

### Prerequisites

- Python (3.10)
- Anaconda or Miniconda (optional but recommended)
- TensorFlow 
- OpenCV 

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/darshigwg/sign-language-model.git
2. Navigate to the project directory
   cd sign-language-model
### File-to-run series
## 1. Data Collection Script

The provided Python script (`data_collection.py`) facilitates the collection of data for training a sign language detection model. This script captures images from a webcam and saves them to the specified directory for each class of sign language gesture.

### Usage

1. **Prerequisites**: Ensure you have OpenCV installed (`pip install opencv-python`).

2. **Running the Script**:
   - Execute the script using Python (`python data_collection.py`).
   - Follow the on-screen instructions to capture images for each sign language gesture class.

### Description

- The script initializes a webcam capture using OpenCV.
- It prompts the user to press 'Q' when ready to start capturing images.
- For each class of sign language gesture (0 to 25, representing letters A to Z), it creates a directory in the specified data directory.
- It continuously captures frames from the webcam until the specified dataset size is reached for each class.
- Each captured frame is saved as an image file (`.jpg`) in the respective class directory.
- Finally, it releases the webcam capture and closes all OpenCV windows.

### Parameters

- `DATA_DIR`: The directory where the captured images will be saved.
- `number_of_classes`: The number of classes (sign language gestures) to capture data for (default is 26, representing letters A to Z).
- `dataset_size`: The number of images to capture for each class.

### Notes

- Ensure proper lighting and background for accurate data collection.
- Adjust `dataset_size` based on your dataset requirements.
- This script assumes a webcam is available and accessible.

## 2.Hand Landmarks Extraction Script

The provided Python script (`hand_landmarks_extraction.py`) extracts hand landmarks from images using the MediaPipe library. It processes images of sign language gestures to extract key hand landmarks, which can be used as features for training a sign language detection model.

### Usage

1. **Prerequisites**: Ensure you have MediaPipe installed (`pip install mediapipe`).

2. **Running the Script**:
   - Execute the script using Python (`python hand_landmarks_extraction.py`).
   - Ensure the `data` directory contains images of sign language gestures organized into subdirectories by class.

### Description

- The script initializes a MediaPipe Hands object for hand landmark detection.
- It iterates through each image in the `data` directory and its subdirectories.
- For each image, it reads the image and converts it to RGB format.
- It processes the image using MediaPipe to detect hand landmarks.
- If hand landmarks are detected, it extracts the x and y coordinates of each landmark and stores them in a list.
- The extracted landmark coordinates are appended to the `data` list, and the corresponding class label is appended to the `labels` list.
- Finally, it saves the extracted data and labels as a serialized pickle file (`data.pickle`).

### Parameters

- `DATA_DIR`: The directory containing the image data organized by class subdirectories.
- `hands`: MediaPipe Hands object for hand landmark detection.
- `results.multi_hand_landmarks`: Detected hand landmarks from MediaPipe.
- `data`: List to store extracted hand landmark data.
- `labels`: List to store corresponding class labels.

### Notes

- Ensure proper installation of MediaPipe and its dependencies.
- Adjust the `min_detection_confidence` parameter in `mp_hands.Hands()` based on your confidence threshold for hand landmark detection.
- The script assumes images are organized into class subdirectories within the `data` directory.


