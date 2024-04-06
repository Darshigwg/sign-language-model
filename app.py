import cv2
import mediapipe as mp
import numpy as np
import pickle
import streamlit as st

model_path = 'D:\personalsign\model.p'
model = pickle.load(open(model_path, 'rb'))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 0, 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L',
               13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W',
               24: 'X', 25: 'Y', 26: 'Z'}

# Define Streamlit app layout
st.title('Sign Language Detection Model')

# Placeholder for displaying webcam feed
frame_placeholder = st.empty()

# Placeholder for displaying predicted letter
predicted_letter_placeholder = st.empty()

# Main Streamlit app loop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Unable to open webcam.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to capture frame from webcam.")
            break

        frame_height, frame_width = frame.shape[:2]

        data_aux = np.zeros(21 * 2 * 2)  # Initialize with zeros to match the expected shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Add a bounding box around the hand
                x_min, y_min = int(hand_landmarks.landmark[0].x * frame_width), int(
                    hand_landmarks.landmark[0].y * frame_height)
                x_max, y_max = x_min, y_min
                for landmark in hand_landmarks.landmark[1:]:
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)  # Change color to black

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux[i * 2] = x
                    data_aux[i * 2 + 1] = y

            prediction = model.predict([data_aux])  # Use the modified data_aux array

            predicted_character = labels_dict[int(prediction[0])]

            predicted_letter_placeholder.text("Predicted letter is: " + predicted_character)

        # Display the frame with Streamlit
        frame_placeholder.image(frame, channels='RGB', use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
