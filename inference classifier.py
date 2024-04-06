import cv2
import mediapipe as mp
import numpy as np
import pickle

model = pickle.load(open('./model.p', 'rb'))

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 0, 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L',
               13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W',
               24: 'X', 25: 'Y', 26: 'Z'}

while True:
    ret, frame = cap.read()

    frame_height, frame_width = frame.shape[:2]

    # Draw the white box with black text at the top of the frame
    cv2.rectangle(frame, (0, 0), (frame_width, 50), (255, 255, 255), -1)
    cv2.putText(frame, "Sign Language Detection Model", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

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

        # Add a white box at the bottom with the predicted letter displayed
        cv2.rectangle(frame, (0, frame_height - 50), (frame_width, frame_height), (255, 255, 255), -1)
        cv2.putText(frame, "Predicted letter is: " + predicted_character, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
